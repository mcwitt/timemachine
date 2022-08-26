let
  systemPkgs = import <nixpkgs> { };

  sources = import ./nix/sources.nix;

  overlay = _: prev: {
    python3 = prev.python3.override {
      packageOverrides = final: _: {
        openmm = final.callPackage "${sources.NixOS-QChem}/pkgs/apps/openmm" {
          inherit (cudaPackages) cudatoolkit;
          enableCuda = false;
        };
      };
    };
  };

  pkgs = import sources.nixpkgs { overlays = [ overlay ]; };

  cudaPackages = pkgs.cudaPackages_11_6;

  python = pkgs.python3.withPackages (ps:
    with ps; [
      grpcio
      grpcio-tools
      jax
      jaxlib
      matplotlib
      openmm
      pip
      rdkit
      scipy
    ]);

in
pkgs.mkShell {

  buildInputs = [
    python

    cudaPackages.cudatoolkit
    cudaPackages.cuda_memcheck
    cudaPackages.cuda_sanitizer_api

    pkgs.bear
    pkgs.black
    pkgs.clang-tools
    pkgs.cmake
    pkgs.hadolint
    pkgs.pyright
    pkgs.shellcheck
    pkgs.stdenv.cc
  ];

  shellHook = ''
    export CUDACXX=${cudaPackages.cudatoolkit}/bin/nvcc
    export LD_LIBRARY_PATH=${systemPkgs.linuxPackages.nvidia_x11}/lib

    # put packages into $PIP_PREFIX instead of the usual locations.
    # https://nixos.wiki/wiki/Python
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${python.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}
