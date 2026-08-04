#!/usr/bin/env bash
# The script _ct_bg.sh launches. Its whole job is to hold the run's parameters, because _ct_bg.sh
# starts the work with `bash $WORK` and passes no environment -- so -Envp on the send cannot reach
# _ct_aa1.sh through it. EDIT THE EXPORTS BELOW for each run rather than adding another file; one
# named launcher per experiment is how tools/ got to several hundred scripts.
#
# Send with:  _send_ct.ps1 -Script tools/_ct_bg.sh -Aux tools/_ct_bgwork.sh,tools/_ct_aa1.sh \
#               -Envp "WORK=/tmp/_ct_bgwork.sh TAG=<tag>"
# Read with:  _send_ct.ps1 -Script tools/_ct_bgread.sh -Envp "TAG=<tag>"
set -uo pipefail

# --- this run: which baseline is PUSH actually supposed to beat? ---------------------------------
# MORI_COMB_PULL picks who stages a caller-owned input (host / kernel / off), NOT the transport.
# Every run in this session used BASE=off, which HANDOFF §18.2 records as the SLOWEST of the three
# (318.8us) -- so the 314.7 -> 287.9 win is real but sits on the wrong line. Re-measure all of them
# on this HEAD rather than quoting that table.
# Price the ZC=0 staging copy. Both legs on ONE head, because the 233.9 / 167.6 pair quoted so far
# was taken on 1e3729d, before FOLDB, and FOLDB touches the fold loop that BOTH transports run.
#
# ZC is per-run, not per-spec, so this is two sweeps rather than two specs. They run back to back,
# which the one-job-at-a-time rule allows only because both configurations are ones we have already
# run repeatedly -- do NOT extend this pattern to an untried gate combination.
#
# Cannot time the copy in-band: the bench captures combine into a cuda graph, and synchronize
# during capture raises hipErrorStreamCaptureUnsupported. Subtraction gives the duration (the copy
# is in the ZC=0 graph and absent from the ZC=1 one); MORI_COMB_CPTIME prints the byte count, which
# is the half that was being assumed. Size print is once, so both combine numbers stay usable.
# MEASURED 1e08978: ZC=0 host 263.2 / ZC=1 197.5, copy = 65.7us on 16384 x 7168 bf16 = 224 MiB,
# so 3575 GB/s of payload. The DIFFERENCE matches the 66.3 and 67.7 on record, but both ABSOLUTE
# numbers are ~20% off the history (ZC=1 was 167.6 combine / 173.6 disp, this run 197.5 / 212.5)
# and dispatch moved too -- dispatch contains no staging copy and was not touched, so whatever
# this is, it is not the change under test. The GPUs are idle (no KFD compute pids, VRAM at the
# 175MB floor), so the remaining candidates are the node itself and one config difference: the
# history ran the spec with '!' (check armed, SKIPCHECK=0) and the run above ran it without.
#
# This sweep isolates that: same ZC, same head, one spec each way. If both land near 197 the
# config is innocent and the node has drifted; if the checked one returns to ~167 then SKIPCHECK
# changes what is being timed and every unchecked number this session needs re-reading.
# MEASURED: check armed 197.5 / not armed 198.2, so SKIPCHECK is innocent. Remaining split is
# node drift vs a regression in 1e3729d..1e08978, which contains FOLDB and b45a9d99 (blockwise
# into the pipelined gather) -- both kernel changes. Go back to the commit the 167.6/173.6 pair
# was taken on and re-run it: same node, same hour, only the code differs.
#
# CHECK THE HEAD LINE. This clone is shallow and single-branch, so fetching a bare sha can fail
# while leaving the old checkout in place, which would print ~197 and read as "node drift" when
# nothing was actually rolled back.
# Is the node back? 1e3729d gave 167.6 at ~12:53 and 197.2 at ~14:46 -- same commit, same
# container, and the only event in between was a neighbour running four-rank tests at 13:06-13:34.
# Current HEAD was measured equal to 1e3729d today (197.5 vs 197.2), so HEAD is the useful thing
# to re-run: back near 167 means the node recovered, still 197 means something persists.
#
# Snapshot the environment INTO THE SAME LOG as the numbers. Every figure in HANDOFF §20 is bare
# -- no load, no neighbours, no container age -- which is exactly why deciding whether 197 was a
# regression or a dirty node took an afternoon and three wrong answers.
echo "=== environment at start ==="
date -u '+  utc %F %T'
uptime | sed 's/^/  /'
docker ps --format '  ctr {{.Names}} {{.Status}}'
ps -eo pcpu,pid,etime,comm --sort=-pcpu --no-headers 2>/dev/null | head -5 | sed 's/^/  cpu /'
rocm-smi --showuse 2>/dev/null | grep -iE "GPU\[" | sed 's/^/  /'
rocm-smi --showmeminfo vram 2>/dev/null | grep -iE "Used" | sed 's/^/  /'
echo "=== end environment ==="

export REV=debug-aa
export BASE=""
export CBN=64 CWPB=8
export ZC=1
export SPECS="zc1now!="

exec bash /tmp/_ct_aa1.sh
