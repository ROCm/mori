#!/usr/bin/env bash
# Read back a detached run's log, and say whether it is still going.
# TAG names the run; LINES caps the output (0 = all).
set -uo pipefail
TAG="${TAG:-bg}"
LOG="$HOME/bg_$TAG.log"
PIDF="$HOME/bg_$TAG.pid"
[ -f "$LOG" ] || { echo "NO_LOG $LOG"; ls "$HOME"/bg_*.log 2>/dev/null; exit 0; }
p=$(cat "$PIDF" 2>/dev/null)
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo "STATE=running pid=$p"; else echo "STATE=finished"; fi
echo "LINES=$(wc -l < "$LOG")  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))s since last write"
echo "----"
LINES="${LINES:-400}"
if [ "$LINES" = 0 ]; then cat "$LOG"; else tail -n "$LINES" "$LOG"; fi
echo "BGREAD_DONE"
