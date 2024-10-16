{
  description = "Operon development environment";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nur.url = "github:nix-community/NUR";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/master";
  inputs.pratt-parser.url = "github:foolnotion/pratt-parser-calculator";
  inputs.vstat.url = "github:heal-research/vstat/cpp20-eve";

  outputs = { self, flake-utils, nixpkgs, nur, pratt-parser, vstat }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nur.overlay ];
        };
        repo = pkgs.nur.repos.foolnotion;

      in rec {
        defaultPackage = pkgs.gcc11Stdenv.mkDerivation {
          name = "operon";
          src = self;

          cmakeFlags = [
            "-DBUILD_CLI_PROGRAMS=ON"
            "-DBUILD_SHARED_LIBS=ON"
            "-DBUILD_TESTING=OFF"
            "-DCMAKE_BUILD_TYPE=Release"
            "-DUSE_OPENLIBM=ON"
            "-DUSE_SINGLE_PRECISION=ON"
            "-DCMAKE_CXX_FLAGS=${if pkgs.targetPlatform.isx86_64 then "-march=haswell" else ""}"
          ];

          nativeBuildInputs = with pkgs; [ cmake pkg-config ];

          buildInputs = with pkgs; [
            cxxopts
            doctest
            eigen
            fmt
            git
            openlibm
            # flakes
            pratt-parser.defaultPackage.${system}
            vstat.defaultPackage.${system}
            # Some dependencies are provided by a NUR repo
            repo.aria-csv
            repo.eve
            repo.fast_float
            repo.robin-hood-hashing
            repo.scnlib
            repo.taskflow
            repo.xxhash
          ];
        };

        devShell = pkgs.gcc11Stdenv.mkDerivation {
          name = "operon-env";

          nativeBuildInputs = with pkgs; [
            bear
            cmake
            clang_14
            clang-tools
            cppcheck
            include-what-you-use
          ];
          buildInputs = defaultPackage.buildInputs ++ (with pkgs; [
            gdb
            hotspot
            hyperfine
            valgrind
            linuxPackages.perf
          ]);

          shellHook = ''
            LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib ]
            };
          '';
        };
      });
}
