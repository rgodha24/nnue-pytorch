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

            if [ -d "/run/opengl-driver/lib" ]; then
              export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${cudaPackages.cuda_nvrtc.lib}/lib:/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            else
              export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${cudaPackages.cuda_nvrtc.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            fi

            export PATH=$CUDA_PATH/bin:$PATH

            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            export NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt

            echo "nnue-pytorch CUDA dev shell loaded"
          '';
        };
      }
    );
}
