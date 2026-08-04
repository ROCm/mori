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
#   host!    ZC=0 default: one d2d on the caller stream, then the zero-copy PULL path (was 236.7)
#   push!    ZC=0 PUSH with FOLDB, i.e. what this session built            (was 287.9)
# MEASURED on 1e3729d: host 233.9 / push 288.2, so PUSH is 54.3us behind PULL in its OWN scenario.
#
# Now the ZC=1 leg, which is where the 169.0/1255.9 reference lives. ZC is per-run, not per-spec,
# hence a separate sweep. Worth pinning down because dispatch in the ZC=0 run above reads
# 173.6us/1223.6 GB/s, close enough to "170us, 1220GB" that the target could be either number, and
# they imply completely different work.
export REV=debug-aa
export BASE=""
export ZC=1
export CBN=64 CWPB=8
export SPECS="zc1!="

exec bash /tmp/_ct_aa1.sh
