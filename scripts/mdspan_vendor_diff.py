#!/usr/bin/env python3
"""Diff the vendored mdspan against upstream, ignoring formatting.

Usage:
    python3 mdspan_vendor_diff.py                  # against the pinned commit
    python3 mdspan_vendor_diff.py --ref single-header   # against the branch tip
    python3 mdspan_vendor_diff.py --find-base      # which upstream commit is it?

Expected output today: eight tokens.  Six are formatting (a sorted #include, a
pair of swapped using-declarations); two are the LOCAL PATCH guard that keeps
at() out of a -fno-exceptions build.
"""
import argparse
import json
import re
import sys
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

REPO = "kokkos/mdspan"
BRANCH = "single-header"
# The upstream commit the vendored mdspan.hpp came from; update on re-vendor.
PINNED = "c3fc07b607db2a43cd65aabc12ffda9328833d95"
VENDORED = Path(__file__).resolve().parent.parent / "include/md/third_party/mdspan.hpp"

TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[A-Za-z_]\w*|\d[\w.\']*|\S')
def strip_comments(src: str) -> str:
    """Drop // and /* */ comments without tripping over // inside a literal."""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            out.append(c)
            i += 1
            while i < n:
                if src[i] == "\\":
                    out.append(src[i : i + 2])
                    i += 2
                    continue
                out.append(src[i])
                i += 1
                if src[i - 1] == c:
                    break
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
            continue
        out.append(c)
        i += 1
    return "".join(out)
def normalize(src: str) -> list[str]:
    """One token per line, except directives: those stay whole, since a
    preprocessor directive ends at its newline and the line break is meaning."""
    text = strip_comments(src.replace("\\\n", ""))
    toks: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:             continue
        if s.startswith("#"): toks.append("#DIRECTIVE " + " ".join(TOKEN.findall(s)))
        else:                 toks.extend(TOKEN.findall(s))
    # Merge adjacent string literals, since clang-format moves the split freely.
    merged: list[str] = []
    for t in toks:
        if (t.startswith('"') and
                t.endswith('"') and
                merged[-1:] and
                merged[-1].startswith('"') and
                merged[-1].endswith('"')
        ):
            merged[-1] = merged[-1][:-1] + t[1:]
        else:
            merged.append(t)
    return merged


def diff(a: list[str], b: list[str], context: int = 3) -> list[str]:
    """Unified diff over token lists.  difflib.unified_diff would do, but its
    autojunk heuristic treats ';' and '::' -- thousands of them here -- as junk
    and stops finding the minimal edit; SequenceMatcher with autojunk off does."""
    ops = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    out: list[str] = []
    for i, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal":
            head = i == 0
            tail = i == len(ops) - 1
            if i2 - i1 <= 2 * context and not head and not tail:
                out += [" " + l for l in a[i1:i2]]
            else:
                if not head:
                    out += [" " + l for l in a[i1 : i1 + context]]
                if not tail:
                    out.append(f"@@ {i2 - context + 1} @@")
                    out += [" " + l for l in a[max(i1, i2 - context) : i2]]
            continue
        out += ["-" + l for l in a[i1:i2]]
        out += ["+" + l for l in b[j1:j2]]
    return out

def fetch(ref: str, cache: Path) -> str:
    path = cache / f"{ref}.hpp"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/{REPO}/{ref}/mdspan.hpp"
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_text()

def commits(limit: int) -> list[tuple[str, str]]:
    url = f"https://api.github.com/repos/{REPO}/commits?sha={BRANCH}&per_page={limit}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return [(c["sha"], c["commit"]["committer"]["date"][:10]) for c in json.load(r)]

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=PINNED, help="upstream commit or branch (default: the pinned commit)")
    ap.add_argument("--file", type=Path, default=VENDORED, help="the vendored copy to check")
    ap.add_argument("--find-base", type=int, nargs="?", const=40, metavar="N",
                    help="rank the last N upstream commits by how close they are")
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/mdspan-upstream"))
    args = ap.parse_args()

    ours = normalize(args.file.read_text())
    if args.find_base:
        ranked = []
        for sha, date in commits(args.find_base):
            n = sum(1 for l in diff(normalize(fetch(sha, args.cache_dir)), ours, context=0)
                    if l[:1] in "-+")
            ranked.append((n, date, sha))
            print(f"  {n:6d} tokens  {date}  {sha}", file=sys.stderr)
        n, date, sha = min(ranked)
        print(f"\nclosest: {sha} ({date}), {n} differing tokens")
        return 0

    theirs = normalize(fetch(args.ref, args.cache_dir))
    hunks = diff(theirs, ours)
    changed = sum(1 for l in hunks if l[:1] in "-+")
    print(f"--- upstream {args.ref[:10]}\n+++ {args.file}")
    print("\n".join(hunks))
    print(f"--- {len(theirs)} upstream tokens, {len(ours)} ours, {changed} differing")
    return 1 if changed else 0

if __name__ == "__main__":
    sys.exit(main())
