#!/usr/bin/env python3
"""The AD comparison, end to end.

ddx, the C++ harness against it, the Python module, and then both runs.  An
optional variable count to sweep to, then any flag the C++ harness takes.

Usage:
    python3 scripts/compare.py                    # to n = 64, the harness default
    python3 scripts/compare.py 128                # to n = 128
    python3 scripts/compare.py 128 --absolute     # ns per call, not ratios
    python3 scripts/compare.py --sizes=8,16,32    # those sizes and no sweep
    python3 scripts/compare.py 2048 --nojit       # no compiled arms: sweep far
    python3 scripts/compare.py --nopython         # C++ only; --nocpp the other
    python3 scripts/compare.py --gpu --stack=rocm     # the GPU libraries
    python3 scripts/compare.py --batch --stack=rocm   # batches, CPU and GPU

--nocpp and --nopython are this script's: they skip that half, its build
included.  C++ harness flags: --absolute, --nodes, --scratch, --nojit, --trend,
--sizes=a,b,c.
Both halves write their cells to build_compare/cells_*.csv and, when the
Python environment is there to draw them, cells_*.png.
--max is set from the count above, so pass the count rather than --max.  The
Python stage is given the same count and the flags it spells the same way --
--sizes, --nojit, --trend, --absolute, --nodes -- and is documented in
compare/python/main.py.

--gpu compares the GPU libraries: ddx's OpenCL device against torch and JAX,
each on the GPU, over batches of points -- a gradient at every point per call.
--batch is the same batches with every column, CPU and GPU: ddx's sweep and
JIT, torch and JAX on the CPU beside the GPU columns.  Both are Python only,
since no C++ AD library runs on a GPU, and write build_compare/cells_gpu.csv
or cells_batch.csv.  --points=, --cells=, --budget= and --arms= narrow them.

--stack=rocm|cuda|cpu syncs that build of torch and JAX into .venv; --gpu
needs one.  The sync is exact, so a run without --stack removes what a previous
run put there.  An RDNA2 APU such as the Radeon 680M wants
HSA_OVERRIDE_GFX_VERSION=10.3.0 in the environment before torch or JAX sees it.

Standard library only, like build_llvm.py: this runs before uv has made the
environment the Python stage needs.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DDX_BUILD = ROOT / "build" / "release_jit"
COMPARE_BUILD = ROOT / "build_compare"
PYTHON_BUILD = ROOT / "build" / "python"
VENV_PYTHON = (
    ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)
HARNESS = COMPARE_BUILD / ("benchmarks_compare" + (".exe" if os.name == "nt" else ""))
MAIN_PY = ROOT / "compare" / "python" / "main.py"
BATCH_FLAGS = ("--points=", "--cells=", "--budget=", "--arms=")


def run(*args: str, env: dict[str, str] | None = None) -> None:
    """Echo and run a command from the ddx root, stopping on failure."""
    print("+", shlex.join(args), flush=True)
    subprocess.run(args, check=True, cwd=ROOT, env=env)


def build_cpp() -> None:
    """Build libddx with the JIT, then the C++ harness against it."""
    # The C++ harness consumes ddx through find_package, so it needs the export
    # that the configure writes and the libddx.so that the build produces --
    # both, not either.  Configuring again is a no-op when nothing moved.
    run("cmake", "--preset", "release_jit")
    run("cmake", "--build", "--preset", "release_jit")
    # DDX_BUILD_DIR is a find_package hint and has to be absolute; a relative
    # one resolves against the harness's binary dir, not this one.
    run(
        "cmake",
        "-S",
        str(ROOT / "compare"),
        "-B",
        str(COMPARE_BUILD),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DDDX_BUILD_DIR={DDX_BUILD}",
    )
    run("cmake", "--build", str(COMPARE_BUILD))


def build_python(stack: str | None) -> None:
    """Build the Python module and install the comparison libraries."""
    # The Python module, always: the stage below imports what this checkout
    # just built, not whatever is installed in the virtualenv.
    run("cmake", "--preset", "python")
    run("cmake", "--build", "--preset", "python")
    # --no-install-project matters.  uv.lock records ddx as an editable
    # install, and an editable install registers a sys.meta_path finder that
    # beats PYTHONPATH -- it would shadow the build tree above.  This installs
    # the comparison libraries and leaves ddx alone, which is what
    # python/tests/conftest.py guards for too.
    groups = ["--group", "compare"]
    if stack:
        groups += ["--group", f"stack-{stack}"]
    run("uv", "sync", *groups, "--no-install-project")


def banner(title: str) -> None:
    """Head a stage's output."""
    print(f"\n{f' {title} ':=^68}", flush=True)


def main() -> None:
    """Parse the count and this script's flags; the rest goes to the harnesses."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="anything else is passed to the C++ harness untouched",
        allow_abbrev=False,
    )
    parser.add_argument(
        "max",
        nargs="?",
        type=int,
        default=64,
        help="largest n to sweep to (default 64)",
    )
    parser.add_argument("--nocpp", action="store_true", help="skip the C++ half")
    parser.add_argument("--nopython", action="store_true", help="skip the Python half")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="the GPU libraries against each other: ddx's device, torch, JAX",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="the same batches with every column, CPU and GPU",
    )
    parser.add_argument(
        "--stack",
        choices=("rocm", "cuda", "cpu"),
        help="sync that build of torch and JAX into .venv",
    )
    args, harness_args = parser.parse_known_args()
    batches = args.gpu or args.batch
    if args.nocpp and args.nopython:
        parser.error("--nocpp and --nopython together leave nothing to run")
    if args.gpu and args.batch:
        parser.error("--gpu is the GPU columns of --batch; ask for one")
    if batches and args.nopython:
        parser.error("--gpu and --batch are Python only; --nopython leaves nothing")
    if args.gpu and not args.stack:
        parser.error("--gpu needs --stack=rocm or --stack=cuda: torch and JAX")
    if not args.nopython and shutil.which("uv") is None:
        sys.exit(
            "compare: uv is needed for the Python stage -- https://docs.astral.sh/uv/"
        )

    cpp = not args.nocpp and not batches
    if cpp:
        build_cpp()
    if not args.nopython:
        build_python(args.stack)
    python_env = {**os.environ, "PYTHONPATH": str(PYTHON_BUILD / "python")}

    if batches:
        suite = "gpu" if args.gpu else "batch"
        banner("GPU libraries" if args.gpu else "Batches, CPU and GPU")
        COMPARE_BUILD.mkdir(exist_ok=True)
        run(
            str(VENV_PYTHON),
            str(MAIN_PY),
            f"--{suite}",
            f"--csv={COMPARE_BUILD / f'cells_{suite}.csv'}",
            *(a for a in harness_args if a.startswith(BATCH_FLAGS) or a == "--nojit"),
            env=python_env,
        )
        return

    if cpp:
        banner("C++")
        cells = COMPARE_BUILD / "cells_cpp.csv"
        run(str(HARNESS), f"--max={args.max}", f"--csv={cells}", *harness_args)
        if not args.nopython:
            run(
                str(VENV_PYTHON),
                str(ROOT / "compare" / "python" / "plot.py"),
                str(cells),
                f"--out={COMPARE_BUILD / 'cells_cpp.png'}",
                "--title=C++: ddx against Adept, CppAD, XAD and autodiff",
            )

    if not args.nopython:
        # Only the flags both halves spell the same way.
        shared = [
            a
            for a in harness_args
            if a.startswith("--sizes=") or a in ("--nojit", "--trend", "--absolute", "--nodes")
        ]
        banner("Python")
        run(
            str(VENV_PYTHON),
            str(MAIN_PY),
            f"--max={args.max}",
            f"--csv={COMPARE_BUILD / 'cells_python.csv'}",
            f"--png={COMPARE_BUILD / 'cells_python.png'}",
            *shared,
            env=python_env,
        )


if __name__ == "__main__":
    main()
