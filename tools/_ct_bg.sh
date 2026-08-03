#!/usr/bin/env bash
# Start a work script on the node DETACHED, and return immediately.
#
# WHY. This node stops answering ssh partway through a four-rank GPU run -- TCP still connects, the
# SSH banner never arrives -- and stays that way for 15 to 60 minutes. It has now done it three
# times. When the ssh session dies it takes the remote bash with it, so the run is lost as well as
# the window, and the next attempt starts from nothing.
#
# Detaching separates the two failures. setsid puts the work in its own session with no controlling
# terminal, so a dropped connection cannot signal it; the log is on the host filesystem, which
# survives whatever the ssh daemon is doing; and every line is flushed as it is produced, so a run
# that dies halfway still leaves the rows it had already printed.
#
# Usage: -Aux the work script to /tmp, then send this with WORK=/tmp/<name> TAG=<tag>.
# Read it back with _ct_bgread.sh TAG=<tag>.
set -uo pipefail
WORK="${WORK:?WORK=/tmp/script.sh required}"
TAG="${TAG:-bg}"
LOG="$HOME/bg_$TAG.log"
[ -f "$WORK" ] || { echo "ABORT: $WORK not on the node"; exit 1; }

# One at a time. Two four-rank jobs on the same four GPUs is the failure that cost a node reboot
# before, and detaching makes it EASIER to do by accident, not harder.
if pgrep -f "bg_marker_$TAG" >/dev/null 2>&1; then echo "ABORT: $TAG already running"; exit 1; fi
for t in $(ls "$HOME"/bg_*.pid 2>/dev/null); do
  p=$(cat "$t" 2>/dev/null)
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo "ABORT: $t still alive (pid $p)"; exit 1; fi
done

: > "$LOG"
setsid nohup env BG_MARKER="bg_marker_$TAG" stdbuf -oL -eL bash "$WORK" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$HOME/bg_$TAG.pid"
sleep 2
echo "STARTED tag=$TAG pid=$(cat "$HOME/bg_$TAG.pid") log=$LOG"
head -3 "$LOG" 2>/dev/null
echo "BG_DONE"
