"""Tilelang implementation of the NNUE sparse feature-transformer *backward*.

Computes, for each batch row b and each active feature k:
    weight_grad[indices[b, k], :] += output_grad[b, :] * values[b, k]
    bias_grad[:] += output_grad[b, :]       (once per row)

Shapes match the forward kernel:
    indices     : (B, K) int32   (-1 sentinel, sorted-active first)
    values      : (B, K) float32 (all-ones in production training)
    output_grad : (B, O) float32
    weight_grad : (N, O) float32 (accumulated)
    bias_grad   : (O,)   float32 (accumulated)

The existing CuPy reference uses atomicAdd from every thread into
global weight_grad and suffers heavy contention on hot features
(king pos, stm, etc.).  We compare against multiple alternatives:

    1) CuPy reference  (atomicAdd per (b, k, s))
    2) Torch eager     (index_add_ — same atomics, different launcher)
    3) Tilelang naive  (direct port of CuPy to tilelang)
    4) Tilelang sorted (sort by feature then segment-reduce, no atomic
       contention on hot features — this should be the big win)

Run via:
    ./scratch/envrun.sh python scratch/tl_sparse_ft_bwd.py
"""

from __future__ import annotations

import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tilelang
import tilelang.language as T

from model.modules.feature_transformer.kernel import (
    make_sparse_input_linear_backward_kernel,
)


# --- reference CuPy kernel --------------------------------------------------

def run_cupy_reference(
    indices: torch.Tensor,
    values: torch.Tensor,
    output_grad: torch.Tensor,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, K = indices.shape
    O = output_grad.shape[1]
    weight_grad = torch.zeros(N, O, dtype=torch.float32, device=indices.device)
    bias_grad = torch.zeros(O, dtype=torch.float32, device=indices.device)
    kernel = make_sparse_input_linear_backward_kernel(K, O)
    kernel(
        grid=(B,),
        args=(
            indices.data_ptr(),
            values.data_ptr(),
            weight_grad.data_ptr(),
            bias_grad.data_ptr(),
            output_grad.data_ptr(),
        ),
    )
    return weight_grad, bias_grad


# --- torch index_add baseline ----------------------------------------------

def run_torch_index_add(
    indices: torch.Tensor,
    values: torch.Tensor,
    output_grad: torch.Tensor,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent: flatten (B, K) -> (B*K,), mask out -1, then index_add_.
    Same math as CuPy, same atomic contention — just torch's launcher.
    """
    B, K = indices.shape
    O = output_grad.shape[1]
    # Expand grad to (B, K, O). We scale later only if needed (values==1 here).
    flat_idx = indices.reshape(-1)
    flat_val = values.reshape(-1)
    flat_grad = output_grad.unsqueeze(1).expand(B, K, O).reshape(B * K, O)
    mask = flat_idx != -1
    flat_idx = flat_idx[mask]
    flat_val = flat_val[mask]
    flat_grad = flat_grad[mask]
    flat_contrib = flat_grad * flat_val.unsqueeze(1)

    weight_grad = torch.zeros(N, O, dtype=torch.float32, device=indices.device)
    weight_grad.index_add_(0, flat_idx.long(), flat_contrib)
    bias_grad = output_grad.sum(dim=0)
    return weight_grad, bias_grad


# --- tilelang: direct port of CuPy (atomic per (b, k, s)) -------------------

_KERNEL_CACHE: dict = {}


@tilelang.jit
def _sparse_ft_backward_naive_factory(B, K, O, N_pad, threads, per_thread):
    @T.prim_func
    def kernel(
        indices: T.Tensor((B, K), "int32"),
        values: T.Tensor((B, K), "float32"),
        output_grad: T.Tensor((B, O), "float32"),
        weight_grad: T.Tensor((N_pad, O), "float32"),
        bias_grad: T.Tensor((O,), "float32"),
    ):
        _shape_capture = (B, K, O, N_pad, threads, per_thread)  # noqa: F841
        with T.Kernel(B, threads=threads) as bx:
            tid = T.get_thread_binding(0)
            og = T.alloc_fragment((per_thread,), "float32")

            # Load this row of output_grad into registers.
            for p in T.serial(per_thread):
                og[p] = output_grad[bx, p * threads + tid]

            # Accumulate bias_grad (one atomic per output element per row).
            for p in T.serial(per_thread):
                T.atomic_add(bias_grad[p * threads + tid], og[p])

            # For each active feature, scatter og * val into weight_grad row.
            for k in T.serial(K):
                idx = indices[bx, k]
                if idx != -1:
                    val = values[bx, k]
                    for p in T.serial(per_thread):
                        T.atomic_add(
                            weight_grad[idx, p * threads + tid], og[p] * val
                        )

    return kernel


def make_tl_sparse_ft_backward(
    B: int, K: int, O: int, N_pad: int, threads: int = 256
):
    assert O % threads == 0, f"O={O} must be divisible by threads={threads}"
    key = ("naive", B, K, O, N_pad, threads)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    per_thread = O // threads
    kernel = _sparse_ft_backward_naive_factory(
        B, K, O, N_pad, threads, per_thread
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


# --- tilelang: sort-and-segment-reduce (pre-sort done in torch) ------------

@tilelang.jit
def _sparse_ft_backward_sorted_factory(
    B, M, U, O, N_pad, threads, per_thread
):
    """Sorted segment-reduce backward.

    Inputs (pre-computed in torch with one sort):
        sorted_feat  : (M,) int32    — feature id for each non-empty entry,
                                       sorted ascending
        sorted_bidx  : (M,) int32    — corresponding batch row index
        sorted_val   : (M,) float32  — corresponding value
        output_grad  : (B, O) float32
        seg_start    : (U,) int32    — start offset of each unique-feature run
        seg_feat     : (U,) int32    — feature id of run u
        seg_count    : (U,) int32    — length of run u (>= 1)

    One block per unique feature `u`, `threads` threads per block.
    """

    @T.prim_func
    def kernel(
        sorted_feat: T.Tensor((M,), "int32"),
        sorted_bidx: T.Tensor((M,), "int32"),
        sorted_val: T.Tensor((M,), "float32"),
        output_grad: T.Tensor((B, O), "float32"),
        seg_start: T.Tensor((U,), "int32"),
        seg_feat: T.Tensor((U,), "int32"),
        seg_count: T.Tensor((U,), "int32"),
        weight_grad: T.Tensor((N_pad, O), "float32"),
    ):
        _shape_capture = (B, M, U, O, N_pad, threads, per_thread)  # noqa: F841
        with T.Kernel(U, threads=threads) as bu:
            tid = T.get_thread_binding(0)
            acc = T.alloc_fragment((per_thread,), "float32")

            for p in T.serial(per_thread):
                acc[p] = T.float32(0)

            start = seg_start[bu]
            count = seg_count[bu]
            feat = seg_feat[bu]

            # Dynamic-bound serial loop over the run for this feature.
            for j in T.serial(count):
                b = sorted_bidx[start + j]
                v = sorted_val[start + j]
                for p in T.serial(per_thread):
                    acc[p] += output_grad[b, p * threads + tid] * v

            for p in T.serial(per_thread):
                weight_grad[feat, p * threads + tid] = acc[p]

    return kernel


def _build_sorted_inputs(indices: torch.Tensor, values: torch.Tensor):
    """Sort (indices, values) by feature id and return the segment tables."""
    B, K = indices.shape
    flat_idx = indices.reshape(-1)
    flat_val = values.reshape(-1)
    flat_bid = (
        torch.arange(B, device=indices.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(B, K)
        .reshape(-1)
    )
    mask = flat_idx != -1
    flat_idx = flat_idx[mask]
    flat_val = flat_val[mask]
    flat_bid = flat_bid[mask]

    # Sort by feature id.
    sorted_idx, perm = torch.sort(flat_idx, stable=True)
    sorted_bid = flat_bid[perm]
    sorted_val = flat_val[perm]

    # Build segments: run-length encode.
    unique_feat, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
    seg_start = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=indices.device),
            torch.cumsum(counts, dim=0)[:-1],
        ]
    ).to(torch.int32)
    return (
        sorted_idx.to(torch.int32),
        sorted_bid.to(torch.int32),
        sorted_val.contiguous(),
        unique_feat.to(torch.int32),
        counts.to(torch.int32),
        seg_start,
    )


def make_tl_sorted_backward(B, M, U, O, N_pad, threads=256):
    assert O % threads == 0
    key = ("sorted", B, M, U, O, N_pad, threads)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    per_thread = O // threads
    k = _sparse_ft_backward_sorted_factory(B, M, U, O, N_pad, threads, per_thread)
    _KERNEL_CACHE[key] = k
    return k


# --- test + benchmark ------------------------------------------------------

def make_fake_batch(
    B: int, K: int, N: int, O: int, device="cuda", hot_frac: float = 0.0
):
    """hot_frac: fraction of the K active features per row that are drawn
    from a tiny hot set (size 64, mimicking NNUE king-position features)."""
    torch.manual_seed(0)
    active = torch.randint(20, K + 1, (B,), device="cpu")
    indices = torch.full((B, K), -1, dtype=torch.int32, device="cpu")
    hot_set = 64  # size of hot feature bucket
    for b in range(B):
        n = int(active[b].item())
        n_hot = int(round(n * hot_frac))
        n_cold = n - n_hot
        cold_part = torch.randint(hot_set, N, (n_cold,), device="cpu")
        hot_part = torch.randint(0, hot_set, (n_hot,), device="cpu")
        rnd = torch.cat([hot_part, cold_part]).to(torch.int32)
        indices[b, :n] = rnd
    values = torch.ones((B, K), dtype=torch.float32, device="cpu")
    output_grad = torch.randn((B, O), dtype=torch.float32, device="cpu") * 0.01
    return (
        indices.to(device),
        values.to(device),
        output_grad.to(device),
    )


def bench(fn, iters: int = 200, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    B, K, O = 16384, 32, 1024
    N = 60000
    # hot_frac ≈ 1/16 mimics NNUE: one king-position feature per row
    # chosen from a ~64-element hot bucket.
    hot_frac = 1.0 / K
    indices, values, output_grad = make_fake_batch(B, K, N, O, hot_frac=hot_frac)
    print(f"B={B} K={K} O={O} N={N}  hot_frac={hot_frac:.3f}")

    # Reference
    ref_wg, ref_bg = run_cupy_reference(indices, values, output_grad, N)

    # Torch index_add baseline
    torch_wg, torch_bg = run_torch_index_add(indices, values, output_grad, N)
    diff_w = (torch_wg - ref_wg).abs().max().item()
    diff_b = (torch_bg - ref_bg).abs().max().item()
    print(f"torch index_add max|diff| w={diff_w:.3e} b={diff_b:.3e}")

    # Tilelang naive
    N_pad = 1 << (N - 1).bit_length()
    tl_naive = make_tl_sparse_ft_backward(B, K, O, N_pad, threads=256)
    weight_grad = torch.zeros(N_pad, O, dtype=torch.float32, device="cuda")
    bias_grad = torch.zeros(O, dtype=torch.float32, device="cuda")
    tl_naive(indices, values, output_grad, weight_grad, bias_grad)
    diff_w = (weight_grad[:N] - ref_wg).abs().max().item()
    diff_b = (bias_grad - ref_bg).abs().max().item()
    print(f"tilelang naive  max|diff| w={diff_w:.3e} b={diff_b:.3e}")

    # Tilelang sorted segment-reduce
    (
        sorted_feat,
        sorted_bid,
        sorted_val,
        seg_feat,
        seg_count,
        seg_start,
    ) = _build_sorted_inputs(indices, values)
    M = sorted_feat.numel()
    U = seg_feat.numel()
    print(f"sorted: M={M} U={U}  (max seg count={int(seg_count.max().item())})")
    tl_sorted = make_tl_sorted_backward(B, M, U, O, N_pad, threads=256)
    weight_grad_s = torch.zeros(N_pad, O, dtype=torch.float32, device="cuda")
    tl_sorted(
        sorted_feat,
        sorted_bid,
        sorted_val,
        output_grad,
        seg_start,
        seg_feat,
        seg_count,
        weight_grad_s,
    )
    diff_w_s = (weight_grad_s[:N] - ref_wg).abs().max().item()
    print(f"tilelang sorted max|diff| w={diff_w_s:.3e}")

    # Benches
    def run_tl_naive():
        weight_grad.zero_()
        bias_grad.zero_()
        tl_naive(indices, values, output_grad, weight_grad, bias_grad)

    def run_tl_sorted_full():
        # Full path including the sort / segment build.
        prepared = _build_sorted_inputs(indices, values)
        sf, sb, sv, feat, count, start = prepared
        wg = torch.zeros(N_pad, O, dtype=torch.float32, device="cuda")
        tl_sorted(sf, sb, sv, output_grad, start, feat, count, wg)
        _ = output_grad.sum(dim=0)

    def run_tl_sorted_kernel_only():
        # Just the reduce kernel with pre-built sorted inputs (realistic
        # upper bound if the sort is amortized across multiple backward
        # passes, which it isn't, but useful to see the kernel itself).
        weight_grad_s.zero_()
        tl_sorted(
            sorted_feat,
            sorted_bid,
            sorted_val,
            output_grad,
            seg_start,
            seg_feat,
            seg_count,
            weight_grad_s,
        )

    def run_torch():
        run_torch_index_add(indices, values, output_grad, N)

    def run_cupy():
        run_cupy_reference(indices, values, output_grad, N)

    cupy_ms = bench(run_cupy)
    torch_ms = bench(run_torch)
    tl_naive_ms = bench(run_tl_naive)
    tl_sorted_full_ms = bench(run_tl_sorted_full)
    tl_sorted_kernel_ms = bench(run_tl_sorted_kernel_only)

    print()
    print(f"cupy backward            : {cupy_ms*1000:7.1f} us")
    print(f"torch index_add_         : {torch_ms*1000:7.1f} us  ({cupy_ms/torch_ms:.2f}x)")
    print(f"tilelang naive           : {tl_naive_ms*1000:7.1f} us  ({cupy_ms/tl_naive_ms:.2f}x)")
    print(f"tilelang sorted (full)   : {tl_sorted_full_ms*1000:7.1f} us  ({cupy_ms/tl_sorted_full_ms:.2f}x)")
    print(f"tilelang sorted (kernel) : {tl_sorted_kernel_ms*1000:7.1f} us  ({cupy_ms/tl_sorted_kernel_ms:.2f}x)")


if __name__ == "__main__":
    main()
