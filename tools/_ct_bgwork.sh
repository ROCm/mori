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
export REV=debug-aa
export BASE=""
export CBN=64 CWPB=8

export ZC=0
export SPECS="zc0host=MORI_COMB_CPTIME=1"
bash /tmp/_ct_aa1.sh

echo "=== second leg: ZC=1, same head, no staging copy ==="
export ZC=1
export SPECS="zc1="
exec bash /tmp/_ct_aa1.sh
