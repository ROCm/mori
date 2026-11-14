#!/usr/bin/env bash
# Stop and remove the containers; the data directory is kept.
p="xgmi-netmon-$(id -un)"
docker rm -f "$p-vm" "$p-grafana"
