#!/bin/bash
echo "========================================"
echo "ROCm Environment Diagnostics"
echo "========================================"
echo ""

echo "1. ROCm Version:"
rocm-smi --showproductname 2>/dev/null || echo "rocm-smi not found"
hipcc --version 2>/dev/null | head -5 || echo "hipcc not found"
echo ""

echo "2. HSA Runtime Version:"
find /opt -name "libhsa-runtime64.so*" 2>/dev/null | xargs ls -l 2>/dev/null
echo ""

echo "3. HSA Environment Variables:"
env | grep -E "HSA|ROCR|HIP" || echo "No HSA/ROCR/HIP env vars set"
echo ""

echo "4. Docker IPC Settings:"
cat /proc/sys/kernel/shmmax 2>/dev/null && echo "(shmmax)"
cat /proc/sys/kernel/shmmni 2>/dev/null && echo "(shmmni)"
echo ""

echo "5. Device Files Permissions:"
ls -la /dev/kfd /dev/dri/render* 2>/dev/null
echo ""

echo "6. GPU Topology (P2P Links):"
rocm-smi --showtopo 2>/dev/null || echo "Cannot show topology"
echo ""

echo "7. Kernel Version:"
uname -r
echo ""

echo "8. Docker Capabilities:"
grep Cap /proc/self/status 2>/dev/null || echo "Cannot read capabilities"
echo ""

echo "9. Check if running in privileged mode:"
if [ -e /sys/fs/cgroup/devices/devices.list ]; then
    if grep -q "a \*:\* rwm" /sys/fs/cgroup/devices/devices.list 2>/dev/null; then
        echo "✅ Running in privileged mode"
    else
        echo "⚠️  Not running in privileged mode"
    fi
fi
echo ""

echo "10. ROCm Driver Version:"
cat /sys/module/amdgpu/version 2>/dev/null || echo "Cannot read amdgpu version"
echo ""

