#!/usr/bin/env bash
# What the kernel says about amdgpu since the reset. A ring timeout or a failed reset here is the
# difference between "restart the container" and "reboot the node".
set -uo pipefail
(sudo -n dmesg -T 2>/dev/null || dmesg -T 2>/dev/null) | grep -iE 'amdgpu|kfd|ring|reset|timeout|fault' | tail -40
echo "--- uptime ---"
uptime
echo "QDMESG_DONE"
