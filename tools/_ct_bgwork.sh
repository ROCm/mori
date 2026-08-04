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

# --- this run: how much of cWait is actually exposed, now that the fold is fast? -----------------
# [CSPLIT] puts the fold loop at 105.7us against the simulator's 98.10 -- only 7.6 left there -- and
# cWait at ~30. PUSH runs issue -> wait -> fold strictly in order (MORI_COMB_PIPE only ever wired
# itself to the PULL side, see the UseP2PRead guards), so none of that wait is hidden. NOWAIT
# deletes the s_wait_tensorcnt to price the whole of it before spending LDS on a double buffer.
#   foldb        287.9us on record, re-run in the same batch so the delta is same-batch
#   foldbnowait  WRONG RESULTS by construction: folds a tile the engine may still be writing.
#                Upper bound on what any amount of overlap could ever recover.
export REV=debug-aa
export BASE="MORI_COMB_PULL=off"
export CBN=64 CWPB=8
export SPECS="foldb=MORI_COMB_FOLDB=1,foldbnowait=MORI_COMB_FOLDB=1 MORI_COMB_NOWAIT=1"

exec bash /tmp/_ct_aa1.sh
