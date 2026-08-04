#!/usr/bin/env bash
# Per-kernel VGPR / SGPR / LDS / SCRATCH straight out of the JIT'd code object. No GPU run, so this
# is always safe to do while something else owns the cards.
#
# private_segment_fixed_size is the one to read first: anything above 0 means the kernel spills to
# scratch, which is off-chip, and a loop that spills runs several times slower than the same source
# compiled without the pressure -- the usual reason a kernel is far slower than a microbenchmark of
# what looks like the same inner loop.
#
# The .hsaco is a clang OFFLOAD BUNDLE, not an ELF: llvm-readelf on it says "not recognized as a
# valid object file", which reads like a missing file rather than a wrapped one. Unbundle first.
#
# N=      how many of the most recently written JIT dirs to look at (default 2, i.e. an A/B pair;
#         the cache holds hundreds and their names encode the gate set)
# KERN=   substring filter on kernel names, empty for all
# DIS=1   also disassemble the FIRST matching kernel and locate its scratch traffic against the
#         fold's ds_read_b128s. A spill count alone cannot tell you whether it costs anything: 192 B
#         saved and reloaded once per token in setup is free, the same 192 B touched inside the
#         element loop is not, and only the instruction addresses separate the two.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
docker exec -e KERN="${KERN:-}" -e N="${N:-2}" -e DIS="${DIS:-0}" \
            -e DIRPAT="${DIRPAT:-*}" -e PAT="${PAT:-}" -e SKIPM="${SKIPM:-0}" "$CTR" bash -lc '
BN=/opt/rocm/llvm/bin
RE=$(command -v llvm-readelf || echo $BN/llvm-readelf)
BUND=$(command -v clang-offload-bundler || echo $BN/clang-offload-bundler)
i=0
# DIRPAT selects which cache entries to look at. The names carry the gate set, so the newest dirs
# after a deletion sweep are the DELETED builds -- disassembling those answers the wrong question.
for f in $(ls -1t /root/.mori/jit/$DIRPAT/latest/ep_intranode.hsaco 2>/dev/null); do
  i=$((i+1)); [ "$i" -gt "$N" ] && break
  d=$(basename "$(dirname "$(dirname "$(readlink -f "$f")")")")
  echo "==== $d"
  echo "     $(date -r "$(readlink -f "$f")" +%F\ %T)"
  tgt=$("$BUND" --type=o --list --input="$(readlink -f "$f")" 2>/dev/null | grep -i gfx | head -1)
  if [ -z "$tgt" ]; then echo "     (no gfx target in bundle)"; continue; fi
  elf=/tmp/hs_$i.elf
  "$BUND" --type=o --unbundle --input="$(readlink -f "$f")" --targets="$tgt" --output="$elf" 2>/dev/null
  # The metadata map is emitted in ALPHABETICAL key order, which is why this cannot just print on
  # the last field it cares about: .group_segment_fixed_size comes BEFORE .name, and .sgpr_count /
  # .vgpr_count come after .private_segment_fixed_size. Latch each one and print at .wavefront_size,
  # which sorts last. Getting this wrong prints blank columns rather than failing.
  "$RE" --notes "$elf" 2>/dev/null | awk -v k="$KERN" "
    /\.group_segment_fixed_size/   { gp = \$NF }
    /\.name:/ { nm = \$0; sub(/.*\.name: */, \"\", nm); keep = (k == \"\" || index(nm, k) > 0); g = gp }
    keep && /\.private_segment_fixed_size/ { p = \$NF }
    keep && /\.sgpr_count/                 { s = \$NF }
    keep && /\.sgpr_spill_count/           { ss = \$NF }
    keep && /\.vgpr_count/                 { v = \$NF }
    keep && /\.vgpr_spill_count/           { vs = \$NF }
    keep && /\.wavefront_size/ {
      printf \"     scratch=%-5s spill(v/s)=%-6s vgpr=%-4s sgpr=%-4s lds=%-7s %s\n\", \
             p, vs \"/\" ss, v, s, g, substr(nm, 1, 58)
      keep = 0 }"

  [ "${DIS:-0}" = 1 ] || continue
  # Exact "<name>:" match on the disassembly itself, not a grep over the symbol table: KERN is a
  # substring, and _bf16_nop2p is a prefix of _bf16_nop2p_fp8cast, so picking a symbol by grep lands
  # on whichever sorts first. No --symbolize-operands either -- it rewrites the header lines this
  # keys on, which is how an earlier attempt "found" a 23-instruction kernel.
  "$BN/llvm-objdump" -d "$elf" 2>/dev/null > /tmp/dis_all.txt
  echo "     objdump lines: $(wc -l < /tmp/dis_all.txt)"
  sym=$(grep -oE "^[0-9a-f]+ <${KERN}>:" /tmp/dis_all.txt | head -1)
  if [ -z "$sym" ]; then
    echo "     no exact <$KERN>: header. Headers present:"
    grep -oE "<[A-Za-z0-9_]+>:" /tmp/dis_all.txt | head -8
    continue
  fi
  awk -v k="$KERN" "
    /^[0-9a-f]+ <[A-Za-z0-9_]+>:/ { ink = (\$0 ~ (\"<\" k \">:\")) }
    ink" /tmp/dis_all.txt > /tmp/dis.txt
  echo "     -- $KERN : $(wc -l < /tmp/dis.txt) lines --"
  echo "     raw sample:"; sed -n "40,44p" /tmp/dis.txt | sed "s/^/       |/"
  # Histogram the mnemonics actually present instead of grepping for a fixed list. gfx1250 renamed
  # them against CDNA -- LDS access is ds_load_*, not ds_read_*, and s_waitcnt split into per-
  # counter waits -- so a fixed list silently reports 0 for the very instructions being looked for.
  echo "     top mnemonics:"
  sed -E "s@//.*@@" /tmp/dis.txt \
    | grep -oE "\b(s|v|ds|global|scratch|buffer|image|flat)_[a-z0-9_]+" \
    | sort | uniq -c | sort -rn | head -22 | sed "s/^/       /"
  # The inner loop of the fold is the densest run of ds_read_b128. Print around the first one so
  # the actual issue order -- reads batched or interleaved with the accumulate, and where the waits
  # land -- is visible rather than inferred from the source.
  # (No apostrophes anywhere below: this whole body is a single-quoted argument to bash -lc, and one
  # in a comment ends the string, which surfaces as a syntax error tens of lines later.)
  # PAT locates the region to print. Default is the packed fp32 add, which is the accumulate at the
  # heart of the fold -- a big kernel has many ds_load sites and most of them are other phases.
  # SKIPM skips the first M matches, for walking between several candidate regions.
  n=$(grep -nE "${PAT:-v_pk_add_f32}" /tmp/dis.txt | sed -n "$(( ${SKIPM:-0} + 1 ))p" | cut -d: -f1)
  if [ -n "$n" ]; then
    echo "     -- around match $(( ${SKIPM:-0} + 1 )) of ${PAT:-v_pk_add_f32} (line $n) --"
    sed -E "s@ +// [0-9A-F]+:.*@@" /tmp/dis.txt \
      | sed -n "$((n>20?n-20:1)),$((n+44))p" | sed "s/^/       /"
  fi
done
echo "===DONE==="
'
echo "===OUTER DONE==="
