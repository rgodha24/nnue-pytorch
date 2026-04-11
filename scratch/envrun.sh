#!/usr/bin/env bash
# Wrapper that sets all the env vars required to run nnue-pytorch + cupy +
# tilelang on this NixOS box without going through `nix develop` each time.
# Usage: ./scratch/envrun.sh python scratch/something.py
set -e

CUDA12_ROOT=${CUDA12_ROOT:-/nix/store/1mps3cdd4jmzsxcy1nnr18riy62wslsr-cuda-merged-12.9}
CUDA_NVRTC=${CUDA_NVRTC:-/nix/store/8byskxb9lcbsags64m1q0ms8xmw5qsx0-cuda12.9-cuda_nvrtc-12.9.86-lib}
GCC_LIB=${GCC_LIB:-/nix/store/ab3753m6i7isgvzphlar0a8xb84gl96i-gcc-15.2.0-lib}

export LD_LIBRARY_PATH="$CUDA12_ROOT/lib:$CUDA_NVRTC/lib:/run/opengl-driver/lib:$GCC_LIB/lib:${LD_LIBRARY_PATH:-}"
export CUDA_PATH="$CUDA12_ROOT"
export CUDA_HOME="$CUDA12_ROOT"
export CUDA_ROOT="$CUDA12_ROOT"
export CPATH="$CUDA12_ROOT/include:${CPATH:-}"
export NVCC_PREPEND_FLAGS="-I$CUDA12_ROOT/include -L$CUDA12_ROOT/lib"
export PATH="$CUDA12_ROOT/bin:$PATH"

exec uv run --no-sync "$@"
