{
  description = "Julia2Nix development environment";
  nixConfig = {
    allowUnfree = true;
    extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.nixpkgs.follows = "nixpkgs";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    devshell.url = "github:numtide/devshell";
    devshell.inputs.nixpkgs.follows = "nixpkgs";

    julia2nix.url = "github:JuliaCN/Julia2Nix.jl";
  };

  outputs = inputs@{ self, julia2nix, ... }:
    (inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
          config.cudaVersion = "12.4";
          overlays = [ inputs.devshell.overlays.default self.overlays.default ];
        };
        julia-wrapped = inputs.julia2nix.lib.${system}.julia-wrapped {
          # package = pkgs.julia_17-bin;
          package = julia2nix.packages.${system}.julia_19-bin;
          enable = {
            # only x86_64-linux is supported
            GR = true;
            python = pkgs.python3.buildEnv.override {
              extraLibs = with pkgs.python3Packages; [ xlrd matplotlib pyqt5 ];
              # ignoreCollisions = true;
            };
          };
        };

        # run this command in your project: nix run github:JuliaCN/Julia2Nix.jl#packages.x86_64-linux.julia2nix
        # we need to generate the julia2nix.toml first
        project = inputs.julia2nix.lib.${system}.buildProject {
          src = ./.;
          name = "your julia project";
          package = julia-wrapped;
        };
      in {
        packages = {
          # make sure you have generated the julia2nix.toml
          # default = project;
        };
        devShells.default = pkgs.devshell.mkShell {
          packages = with pkgs; [
            cudatoolkit
            cudaPackages.cudnn
            git
            gitRepo
            pdf2svg
            gnupg
            autoconf
            curl
            procps
            gnumake
            util-linux
            m4
            gperf
            unzip
            libGLU
            libGL
            xorg.libXi
            xorg.libXmu
            freeglut
            xorg.libXext
            xorg.libX11
            xorg.libXv
            xorg.libXrandr
            zlib
            ncurses5
            stdenv.cc
            binutils
          ];
          env = [
            {
              name = "NIX_LD_LIBRARY_PATH";
              value = pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc
                # ...
              ];
            }
            {
              name = "NIX_LD";
              value = "${pkgs.glibc}/lib/ld-linux-x86-64.so.2";
            }
            {
              name = "CUDA_PATH";
              value = "${pkgs.cudatoolkit}";
            }
            {
              name = "LD_LIBRARY_PATH";
              value =
                "${pkgs.linuxPackages.nvidia_x11_production}/lib:${pkgs.ncurses5}/lib";
            }
            {
              name = "EXTRA_LDFLAGS";
              value =
                "-L/lib -L${pkgs.linuxPackages.nvidia_x11_production}/lib";
            }
            {
              name = "EXTRA_CCFLAGS";
              value = "-I/usr/include";
            }
            {
              name = "JULIA_NUM_THREADS";
              value = "auto";
            }
          ];
          imports = [
            # you can keep either one of them devshellProfiles.packages or julia-wrapped
            #inputs.julia2nix.${pkgs.system}.julia2nix.devshellProfiles.packages

            # add dev-tools in your devshell
            inputs.julia2nix.${pkgs.system}.julia2nix.devshellProfiles.dev

            # add nightly julia
            # inputs.julia2nix.${pkgs.system}.julia2nix.devshellProfiles.nightly
          ];
          commands = [{
            package = julia-wrapped;
            help =
              julia2nix.packages.${pkgs.system}.julia_19-bin.meta.description;
          }];
        };
      })) // {
        overlays.default = final: prev: { };
      };
}
