#!/bin/bash
# Install Broadcom's RoCE userspace provider (libbnxt_re) for Thor/Thor2 NICs.
#
# The inbox bnxt_re provider that ships with rdma-core only speaks kernel ABI 1
# (still true as of rdma-core 50 on noble). Hosts running the 237.x bnxt_re
# driver expose ABI 8, so without this package libibverbs rejects every device
# and the container sees no RDMA hardware at all.
#
# Usage: install_bnxt_rocelib.sh <bnxt-rocelib version>   e.g. 235.2.86.0
# Pick the version matching the host's bnxt_re driver; list the available ones
# with `apt-cache madison bnxt-rocelib`.
set -euo pipefail

VERSION="${1:?usage: install_bnxt_rocelib.sh <bnxt-rocelib version>}"
CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://packages.broadcom.com/artifactory/api/security/keypair/PackagesKey/public \
    -o /etc/apt/keyrings/broadcom-nic.asc
chmod a+r /etc/apt/keyrings/broadcom-nic.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/broadcom-nic.asc] \
https://packages.broadcom.com/artifactory/ethernet-nic-debian-public ${CODENAME} main" \
    > /etc/apt/sources.list.d/broadcom-nic.list

apt-get update
apt-get install -y --no-install-recommends "bnxt-rocelib=${VERSION}"

# The package installs under /usr/local/lib/x86_64-linux-gnu; mori's env_check
# looks for libbnxt_re-<ver>.so directly under /usr/local/lib.
cp -a /usr/local/lib/x86_64-linux-gnu/libbnxt_re* /usr/local/lib/.
ldconfig

rm -rf /var/lib/apt/lists/*
