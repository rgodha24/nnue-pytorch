"""Tilelang implementation of the NNUE sparse feature-transformer forward op.

This is the GPU kernel that computes, for each row b in a batch:
    out[b, :] = bias[:] + sum_{k : indices[b, k] != -1}
                          weight[indices[b, k], :] * values[b, k]

Shapes:
    indices : (B, K) int32    (-1 sentinel means empty slot, sorted-active first)
    values  : (B, K) float32  (all-ones in the production training path)
    weight  : (N, O) float32
    bias    : (O,)   float32
    out     : (B, O) float32

For NNUE with Full_Threats+HalfKAv2_hm:
    B = 16384, K ~ 32 active features max, O = 1032 (L1 + num_psqt_buckets),
    N ~ 60k-ish rows in the feature table.

The reference CuPy kernel (model/modules/feature_transformer/kernel.py)
launches one block per row with 512 threads, each thread covering 2 output
elements.  That works fine because O is small.  We match that strategy in
tilelang but use T.Parallel to let the compiler pick vectorization, and
autotune block_size / threads.
"""

from __future__ import annotations

import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tilelang
import tilelang.language as T

# --- reference CuPy kernel (imported from the existing code path) -----------

from model.modules.feature_transformer.kernel import (
    make_sparse_input_linear_forward_kernel,
)


def run_cupy_reference(
    indices: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    B, K = indices.shape
    O = weight.shape[1]
    out = torch.empty(B, O, dtype=torch.float32, device=indices.device)
    kernel = make_sparse_input_linear_forward_kernel(K, O)
    kernel(
        grid=(B,),
        args=(
            indices.data_ptr(),
            values.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            out.data_ptr(),
        ),
    )
    return out


# --- tilelang kernel -------------------------------------------------------

_KERNEL_CACHE: dict = {}


@tilelang.jit(out_idx=[-1])
def _sparse_ft_forward_factory(B, K, O, N_pad, threads, per_thread):
    """One block per batch row, `threads` threads per block, plain serial K."""

    @T.prim_func
    def kernel(
        indices: T.Tensor((B, K), "int32"),
        values: T.Tensor((B, K), "float32"),
        weight: T.Tensor((N_pad, O), "float32"),
        bias: T.Tensor((O,), "float32"),
        out: T.Tensor((B, O), "float32"),
    ):
        _shape_capture = (B, K, O, N_pad, threads, per_thread)  # noqa: F841
        with T.Kernel(B, threads=threads) as bx:
            acc = T.alloc_fragment((per_thread,), "float32")
            tid = T.get_thread_binding(0)

            for p in T.serial(per_thread):
                acc[p] = bias[p * threads + tid]

            for k in T.serial(K):
                idx = indices[bx, k]
                if idx != -1:
                    val = values[bx, k]
                    for p in T.serial(per_thread):
                        acc[p] += weight[idx, p * threads + tid] * val

            for p in T.serial(per_thread):
                out[bx, p * threads + tid] = acc[p]

    return kernel


@tilelang.jit(out_idx=[-1])
def _sparse_ft_forward_factory_pipelined(B, K, O, N_pad, threads, per_thread):
    """Same as above but with the K loop marked `T.Pipelined(num_stages=2)`,
    which asks tilelang to software-pipeline the weight-row loads."""

    @T.prim_func
    def kernel(
        indices: T.Tensor((B, K), "int32"),
        values: T.Tensor((B, K), "float32"),
        weight: T.Tensor((N_pad, O), "float32"),
        bias: T.Tensor((O,), "float32"),
        out: T.Tensor((B, O), "float32"),
    ):
        _shape_capture = (B, K, O, N_pad, threads, per_thread)  # noqa: F841
        with T.Kernel(B, threads=threads) as bx:
            acc = T.alloc_fragment((per_thread,), "float32")
            tid = T.get_thread_binding(0)

            for p in T.serial(per_thread):
                acc[p] = bias[p * threads + tid]

            for k in T.Pipelined(K, num_stages=2):
                idx = indices[bx, k]
                if idx != -1:
                    val = values[bx, k]
                    for p in T.serial(per_thread):
                        acc[p] += weight[idx, p * threads + tid] * val

            for p in T.serial(per_thread):
                out[bx, p * threads + tid] = acc[p]

    return kernel


def make_tl_sparse_ft_forward(
    B: int, K: int, O: int, N_pad: int, threads: int = 256, pipelined: bool = False
):
    assert O % threads == 0, f"O={O} must be divisible by threads={threads}"
    key = (B, K, O, N_pad, threads, pipelined)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    per_thread = O // threads
    factory = (
        _sparse_ft_forward_factory_pipelined if pipelined else _sparse_ft_forward_factory
    )
    kernel = factory(B, K, O, N_pad, threads, per_thread)
    _KERNEL_CACHE[key] = kernel
    return kernel


# --- test + benchmark ------------------------------------------------------

def make_fake_batch(B: int, K: int, N: int, O: int, device="cuda"):
    torch.manual_seed(0)
    # Half-ish sparsity: 20..K active features per row, padded with -1.
    active = torch.randint(20, K + 1, (B,), device="cpu")
    indices = torch.full((B, K), -1, dtype=torch.int32, device="cpu")
    for b in range(B):
        n = int(active[b].item())
        rnd = torch.randperm(N, device="cpu")[:n].to(torch.int32)
        indices[b, :n] = rnd
    values = torch.ones((B, K), dtype=torch.float32, device="cpu")
    weight = torch.randn((N, O), dtype=torch.float32, device="cpu") * 0.01
    bias = torch.randn((O,), dtype=torch.float32, device="cpu") * 0.01
    return (
        indices.to(device),
        values.to(device),
        weight.to(device),
        bias.to(device),
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


def theoretical_bw_bound_us(
    B: int, K_active: float, O: int, bw_gbps: float
) -> float:
    """Rough lower bound for sparse FT forward runtime, assuming weight-row
    reads dominate (they do: ~2 GB of random loads vs ~66 MB of output).

    Returns microseconds.
    """
    # Per-row: K_active * O * 4 bytes of weight + O*4 bytes of output.
    bytes_total = B * (K_active * O * 4 + O * 4) + O * 4  # + bias read once
    gb = bytes_total / 1e9
    seconds = gb / bw_gbps
    return seconds * 1e6


def main():
    B, K, O = 16384, 32, 1024
    N = 60000
    indices, values, weight, bias = make_fake_batch(B, K, N, O)
    print(f"B={B} K={K} O={O} N={N}")

    # Reference
    ref_out = run_cupy_reference(indices, values, weight, bias)

    # Pad weight to a power-of-two N for tilelang (it bakes N into signature).
    N_pad = 1 << (N - 1).bit_length()
    weight_pad = torch.zeros((N_pad, O), dtype=torch.float32, device="cuda")
    weight_pad[:N].copy_(weight)

    # Correctness check with default config.
    tl_kernel = make_tl_sparse_ft_forward(B, K, O, N_pad, threads=256)
    tl_out = tl_kernel(indices, values, weight_pad, bias)
    max_diff = (tl_out - ref_out).abs().max().item()
    print(f"max|tl - cupy| = {max_diff:.3e}")
    assert max_diff < 1e-3, f"mismatch: {max_diff}"

    # Theoretical lower bound: RTX 5060 Ti has ~448 GB/s of HBM/GDDR BW.
    # Assume ~26 active features per row on average (our fake data draws
    # uniform in [20, 32]).
    k_mean = (20 + K) / 2
    for bw in (448.0, 300.0, 200.0):
        us = theoretical_bw_bound_us(B, k_mean, O, bw)
        print(f"  BW lower bound @ {bw:.0f} GB/s : {us:7.1f} us")

    # Bench reference once.
    ref_ms = bench(lambda: run_cupy_reference(indices, values, weight, bias))
    print()
    print(f"cupy (reference)           : {ref_ms*1000:7.1f} us")

    # Sweep threads. (Pipelined variant is broken: the `if idx != -1` inside
    # the K-loop body makes tilelang's PipelinePlanner bail out with
    # "Can't handle the body of the loop because it is not a SeqStmt, ...".)
    configs = [
        ("tilelang threads=128", 128, False),
        ("tilelang threads=256", 256, False),
        ("tilelang threads=512", 512, False),
    ]
    best = None
    for name, t, pipelined in configs:
        if O % t != 0:
            continue
        k = make_tl_sparse_ft_forward(B, K, O, N_pad, threads=t, pipelined=pipelined)
        # correctness
        out = k(indices, values, weight_pad, bias)
        diff = (out - ref_out).abs().max().item()
        assert diff < 1e-3, f"{name}: mismatch {diff}"
        ms = bench(lambda k=k: k(indices, values, weight_pad, bias))
        speedup = ref_ms / ms
        tag = ""
        if best is None or ms < best[1]:
            best = (name, ms)
            tag = " <- best"
        print(f"{name:27s}: {ms*1000:7.1f} us  ({speedup:.2f}x){tag}")
    print(f"\nbest tilelang: {best[0]} @ {best[1]*1000:.1f} us  ({ref_ms/best[1]:.2f}x over cupy)")


if __name__ == "__main__":
    main()
