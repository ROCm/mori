#!/usr/bin/env bash
# How many GPUs does this host actually have, and which four is the container mapped to?
#
# Asked before reaching for a reboot: /dev/dri has card0..card19, so if the wedge is confined to
# the four the container holds there may be another four to move to. A reboot on a node with other
# users on it is the last option, not the first.
set -uo pipefail
CTR="${CTR:-MORI-EPV2}"
echo "== host gpu list =="
timeout 60 rocm-smi --showid 2>&1 | head -30
echo "== container device mapping =="
docker inspect "$CTR" --format '{{json .HostConfig.Devices}}' 2>&1 | head -c 1200; echo
docker inspect "$CTR" --format 'env={{json .Config.Env}}' 2>&1 | tr ',' '\n' | grep -iE 'VISIBLE|HIP|ROCR' | head -6
echo "== hung tasks now =="
(sudo -n dmesg -T 2>/dev/null || dmesg -T) | tail -6
echo "QGPUS_DONE"
