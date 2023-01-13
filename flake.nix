{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    timemachine-flake.url = "github:mcwitt/timemachine-flake";
    timemachine-flake.inputs.nixpkgs.follows = "nixpkgs";
    pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  };

  outputs = { self, nixpkgs, timemachine-flake, pre-commit-hooks, ... }:
    let
      system = "x86_64-darwin";

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ timemachine-flake.overlays.default ];
      };
    in
    {
      devShells.${system}.default = timemachine-flake.devShells.${system}.timemachine.override (old: {
        shellHook = ''
          ${old.shellHook}
          ${self.checks.${system}.pre-commit-check.shellHook}
        '';
      });

      checks.${system}.pre-commit-check = pre-commit-hooks.lib.${system}.run {
        src = ./.;

        excludes = [
          "\\.pdb"
          "\\.sdf"
          "\\.proto"
          "\\.xml"
          "/vendored/"
          "^attic/"
          "^timemachine/ff/params/"
          "^timemachine/_vendored/"
          "^versioneer\\.py$"
          "^timemachine/_version\\.py$"
          "^timemachine/lib/custom_ops\\.pyi$"
        ];

        hooks = {
          check-yaml = {
            enable = true;
            name = "check yaml";
            description = "checks yaml files for parseable syntax.";
            entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-yaml";
            types = [ "yaml" ];
          };

          end-of-file-fixer = {
            enable = true;
            name = "fix end of files";
            description = "ensures that a file is either empty, or ends with one newline.";
            entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/end-of-file-fixer";
            types = [ "text" ];
          };

          trailing-whitespace = {
            enable = true;
            name = "trim trailing whitespace";
            description = "trims trailing whitespace.";
            entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/trailing-whitespace-fixer";
            types = [ "text" ];
          };

          black.enable = true;
          flake8.enable = true;
          isort.enable = true;

          mypy = {
            enable = true;
            types_or = [ "python" "pyi" ];
            files = nixpkgs.lib.mkForce "^timemachine.*\.pyi?$";
            excludes = [ "^timemachine/lib/custom_ops.py$" ];
          };

          clang-format = {
            enable = true;
            files = "^timemachine/cpp/src/";
            types_or = [ "c" "c++" "cuda" ];
          };

          nixpkgs-fmt.enable = true;
        };

        settings = {
          mypy.binPath =
            let
              python = pkgs.python3.withPackages (ps: with ps; [
                mypy
                numpy
                types-pyyaml
              ]);
            in
            "${python}/bin/mypy";
        };
      };
    };
}
