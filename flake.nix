{
  description = "nnue-pytorch CUDA development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        cudaPackages = pkgs.cudaPackages_12;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            uv

            rustc
            cargo
            maturin

            cudaPackages.cudatoolkit
            cudaPackages.cuda_nvrtc
            cudaPackages.cuda_cudart
            cudaPackages.cudnn
            cudaPackages.libcublas
            cudaPackages.libcufft
            cudaPackages.libcurand
            cudaPackages.libcusparse
            cudaPackages.libcusolver
            cudaPackages.nccl

            stdenv.cc.cc.lib
            glibc

            gcc
            cmake
            pkg-config

            cacert
            openssl
          ];

          shellHook = ''
            export CUDA_PATH=${cudaPackages.cudatoolkit}
            export CUDA_HOME=$CUDA_PATH
            export CUDA_ROOT=$CUDA_PATH
            export CC=/run/current-system/sw/bin/gcc
            export CXX=/run/current-system/sw/bin/g++
            export CUDAHOSTCXX=/run/current-system/sw/bin/g++
            export CPATH=$CUDA_PATH/include:${pkgs.glibc.dev}/include:$CPATH
            export C_INCLUDE_PATH=${pkgs.glibc.dev}/include:$C_INCLUDE_PATH
            export CPLUS_INCLUDE_PATH=${pkgs.glibc.dev}/include:$CPLUS_INCLUDE_PATH
            export NVCC_PREPEND_FLAGS="-I$CUDA_PATH/include -L$CUDA_PATH/lib"
            export NVCC_APPEND_FLAGS="--compiler-options -idirafter,${pkgs.glibc.dev}/include"

            if [ -d "/run/opengl-driver/lib" ]; then
              export LD_LIBRARY_PATH=$CUDA_PATH/lib:${cudaPackages.cuda_nvrtc.lib}/lib:/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            else
              export LD_LIBRARY_PATH=$CUDA_PATH/lib:${cudaPackages.cuda_nvrtc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            fi

            export PATH=/run/current-system/sw/bin:$CUDA_PATH/bin:$PATH

            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            export NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt

            echo "nnue-pytorch CUDA dev shell loaded"
          '';
        };
      }
    );
}
