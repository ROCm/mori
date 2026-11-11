#!/usr/bin/env bash
# Stop the stack. The data directory is kept; delete it yourself to start from scratch.
set -euo pipefail
cd "$(dirname "$0")"
docker compose down
