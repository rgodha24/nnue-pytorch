"""Smoke test for tilelang on this machine."""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def add(N, block=256):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                C[idx] = A[idx] + B[idx]
    return kernel


def main():
    k = add(1024)
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    c = k(a, b)
    print("ok", torch.allclose(c, a + b))


if __name__ == "__main__":
    main()
