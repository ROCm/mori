#!/usr/bin/env bash
# Runs test_ionic_mr_limit on this host: the 4 KiB tier always, the 2 MiB
# hugetlb tier (the UMBP DRAM pool's case) unless SKIP_HUGE=1. Pass the
# firmware cap through FW_MAX_ENTRIES (0 = only check the driver cap).
#
# The hugetlb tier temporarily raises vm.nr_hugepages to cover the largest case
# (~1 TiB) and restores the original count on exit; it needs passwordless
# sudo and that much free RAM. Set FULL_REPRO=1 to also register the exact
# 1536 GiB pool that failed in production (needs ~1.5 TiB of hugepages); it
# fails in the driver (ENOMEM) rather than in firmware (EINVAL).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dev="${DEV:-}"
out="${OUT_DIR:-$(mktemp -d)}"
mkdir -p "$out"
bin="$out/test_ionic_mr_limit"

g++ -O2 -std=c++17 -o "$bin" "$here/test_ionic_mr_limit.cpp" -libverbs
dev_args=(--fw-max-entries "${FW_MAX_ENTRIES:-502528}")
[[ -n "$dev" ]] && dev_args+=(--dev "$dev")

echo "== host $(hostname -s) kernel $(uname -r) ionic_rdma $(modinfo -F version ionic_rdma 2>/dev/null || echo '?')"
echo "== tier 1: 4 KiB pages (fw ceiling 1963 MiB, driver ceiling 2 GiB)"
rc=0
"$bin" "${dev_args[@]}" --page 4k --sizes 1G,1963M,1964M,2G+64K,3G || rc=$?

if [[ "${SKIP_HUGE:-0}" != 1 ]]; then
  sizes="512G,980G,1005056M,1005058M,1023G"
  [[ "${FULL_REPRO:-0}" == 1 ]] && sizes="$sizes,1536G"
  largest_gib=$([[ "${FULL_REPRO:-0}" == 1 ]] && echo 1536 || echo 1023)
  need_pages=$(( largest_gib * 512 + 512 ))

  nr_file=/proc/sys/vm/nr_hugepages
  orig_pages=$(cat "$nr_file")
  restore() { echo "$orig_pages" | sudo -n tee "$nr_file" >/dev/null; echo "== restored nr_hugepages=$(cat $nr_file)"; }
  trap restore EXIT

  echo "== reserving $need_pages x 2 MiB hugepages (was $orig_pages)"
  echo "$need_pages" | sudo -n tee "$nr_file" >/dev/null
  got_pages=$(awk '/HugePages_Total/{print $2}' /proc/meminfo)
  if (( got_pages < need_pages )); then
    echo "== only got $got_pages hugepages; not enough contiguous free memory for the 2 MiB tier"
    exit 1
  fi

  echo "== tier 2: 2 MiB hugetlb pages (fw ceiling 981.5 GiB, driver ceiling 1 TiB)"
  echo "   ulimit -l: $(ulimit -l)"
  "$bin" "${dev_args[@]}" --page 2m --sizes "$sizes" || rc=$?
fi
exit $rc
