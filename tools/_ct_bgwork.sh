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

# --- this run: where are the 35.5us the fold still has over the simulator? -----------------------
# FOLDB got the fold to 133.6 against tdm_redsim's 98.10 at the same geometry. Rather than guess at
# the next branch to delete, let [CSPLIT] split the kernel: cSetup is the per-token routing, cIssue
# the TDM launch, cWait the tensorcnt wait, cRed the fold loop itself. The simulator models only
# cRed, so whichever of the others is large IS the unmodelled gap, and it is measured rather than
# inferred by subtraction.
# Caveat on reading it: the buckets are atomicMax over warps, and only the first 12 calls print, so
# early rows include the cold launch -- read the last rows. Neither spec carries '!' because the
# printf itself perturbs the timing, so these are for APPORTIONING, not for headline numbers.
export REV=debug-aa
export BASE="MORI_COMB_PULL=off"
export CBN=64 CWPB=8
export SPECS="basetm=MORI_COMB_TIMING=1,foldbtm=MORI_COMB_TIMING=1 MORI_COMB_FOLDB=1"

exec bash /tmp/_ct_aa1.sh
