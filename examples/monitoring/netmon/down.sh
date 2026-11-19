#!/usr/bin/env bash
# Stop the stack. The data directory is kept; delete it yourself to start from scratch.
set -euo pipefail
cd "$(dirname "$0")"
name="netmon-$(id -un)"
for c in "$name-vm" "$name-grafana"; do
  if [ -n "$(docker ps -aq --filter "name=^${c}\$")" ]; then
    docker rm -f "$c" >/dev/null && echo "removed $c"
  fi
done
# The earlier docker-compose version of this stack (same data directory layout): stop it too,
# but only if it runs as you (its names do not say whose it is).
for c in netmon-victoriametrics-1 netmon-grafana-1; do
  if [ -n "$(docker ps -aq --filter "name=^${c}\$")" ] &&
    [ "$(docker inspect -f '{{.Config.User}}' "$c")" = "$(id -u):$(id -g)" ]; then
    docker rm -f "$c" >/dev/null && echo "removed $c (docker-compose stack)"
  fi
done
