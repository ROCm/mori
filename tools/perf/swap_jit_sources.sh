#!/bin/bash
# Swap JIT kernel sources of the installed mori package between the PR's and a base
# commit's versions, for an A/B perf run without reinstalling.
#
# Usage: swap_jit_sources.sh <base|head> <dir>
#   <dir>/files.txt  repo-relative paths, one per line (e.g. src/ops/dispatch_combine/x.cpp)
#   <dir>/base/<path> the base commit's copy of each path
#
# "base" saves the installed copy as <file>.head and installs the base copy; "head"
# restores it. Kernels are JIT-compiled from mori/_jit-sources with a content-hashed
# cache, so the next run recompiles whichever version is in place.
set -euo pipefail

MODE="$1"
DIR="$2"
JIT=$(python3 -c 'import mori, os; print(os.path.join(os.path.dirname(mori.__file__), "_jit-sources"))')

while read -r f; do
  [[ -z "$f" ]] && continue
  case "$MODE" in
    base)
      [[ -f "$JIT/$f.head" ]] || cp "$JIT/$f" "$JIT/$f.head"
      cp "$DIR/base/$f" "$JIT/$f"
      ;;
    head)
      [[ -f "$JIT/$f.head" ]] && mv "$JIT/$f.head" "$JIT/$f"
      ;;
    *) echo "usage: $0 <base|head> <dir>"; exit 1 ;;
  esac
  echo "[$MODE] $JIT/$f"
done < "$DIR/files.txt"
