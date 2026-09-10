#!/usr/bin/env python3
"""Stage the native NuGet package tree from two install prefixes.

The layout `nuget pack contrib/nuget/ddx.nuspec -BasePath <stage>` expects:

    <stage>/
        README.md, LICENSE.txt, THIRD-PARTY-NOTICES.txt
        build/native/ddx.targets
        build/native/include/**                      headers + the Boost subset
        build/native/lib/x64/{Release,Debug}/ddx.lib
        build/native/bin/x64/{Release,Debug}/ddx.dll

Headers come from the Release prefix; the two prefixes differ only in the
binaries.  Missing pieces stop the script — a package staged short would pack
and publish without complaint.

Usage:
    python3 pack_nuget.py --release <prefix> --debug <prefix> --stage <dir>
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True, help="Release install prefix")
    ap.add_argument("--debug", type=Path, required=True, help="Debug install prefix")
    ap.add_argument("--stage", type=Path, required=True, help="output directory (recreated)")
    args = ap.parse_args()

    shutil.rmtree(args.stage, ignore_errors=True)
    native = args.stage / "build" / "native"
    native.mkdir(parents=True)

    shutil.copy2(ROOT / "contrib" / "nuget" / "ddx.targets", native)
    shutil.copy2(ROOT / "README.md", args.stage)
    shutil.copy2(ROOT / "LICENSE.txt", args.stage)
    shutil.copy2(ROOT / "THIRD-PARTY-NOTICES.txt", args.stage)
    shutil.copytree(args.release / "include", native / "include")

    missing = []
    for config, prefix in (("Release", args.release), ("Debug", args.debug)):
        for sub, name in (("lib", "ddx.lib"), ("bin", "ddx.dll")):
            src = prefix / sub / name
            if not src.is_file():
                missing.append(src)
                continue
            dst = native / sub / "x64" / config
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 1
    print(f"staged {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
