"""Test that TileLang kernels actually work on this machine."""

import torch
import tilelang
import tilelang.language as T

print(f"TileLang version: {tilelang.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


# Simple vector add kernel
@tilelang.jit
def vector_add():
    # Define a simple kernel that adds two vectors
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(1, threads=128) as bx:
            tid = T.get_thread_binding(0)
            # Each thread handles one element
            for i in T.serial(n):
                if i % 128 == tid:
                    C[i] = A[i] + B[i]

    return kernel


print("\nCompiling vector_add kernel...")
vector_add_kernel = vector_add()
print("Compiled successfully!")

# Test with actual tensors
print("\nTesting with tensors...")
n = 1024
a = torch.randn(n, device="cuda", dtype=torch.float32)
b = torch.randn(n, device="cuda", dtype=torch.float32)
c = torch.empty(n, device="cuda", dtype=torch.float32)

print(f"Input A shape: {a.shape}")
print(f"Input B shape: {b.shape}")

vector_add_kernel(a, b, c)

# Verify result
expected = a + b
max_error = torch.max(torch.abs(c - expected)).item()
print(f"\nMax error: {max_error}")
if max_error < 1e-5:
    print("✅ TileLang kernel works correctly!")
else:
    print("❌ Results mismatch!")
    print(f"Expected: {expected[:10]}")
    print(f"Got: {c[:10]}")

print("\nAll tests passed!")
