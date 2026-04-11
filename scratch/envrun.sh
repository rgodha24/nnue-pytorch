#!/usr/bin/env bash
# Wrapper that sets all the env vars required to run nnue-pytorch + cupy +
# tilelang on this NixOS box without going through `nix develop` each time.
# Usage: ./scratch/envrun.sh python scratch/something.py
set -e
CUDA12_ROOT=/nix/store/1mps3cdd4jmzsxcy1nnr18riy62wslsr-cuda-merged-12.9
CUDA_NVCC=/nix/store/8j41syz9cbh1l74k2283q14ghpap7nfx-cuda12.9-cuda_nvcc-12.9.86
CUDA_NVRTC=/nix/store/8byskxb9lcbsags64m1q0ms8xmw5qsx0-cuda12.9-cuda_nvrtc-12.9.86-lib

export LD_LIBRARY_PATH=$CUDA12_ROOT/lib:$CUDA_NVRTC/lib:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}
export CUDA_PATH=$CUDA12_ROOT
export CUDA_HOME=$CUDA12_ROOT
export CPATH=$CUDA12_ROOT/include:${CPATH:-}
export NVCC_PREPEND_FLAGS="-I$CUDA12_ROOT/include -L$CUDA12_ROOT/lib"
export PATH=$CUDA_NVCC/bin:$CUDA12_ROOT/bin:$PATH

exec uv run --no-sync "$@"
