# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Shared helpers for the ops layer.

GPU model detection reads KFD sysfs (torch/HIP-free): PCI device id first, gfx
arch as fallback. CU count is deliberately NOT used to identify a model -- it
cannot separate MI300X from MI325X (both 304 CU) or MI350X from MI355X (both
256 CU), and it shifts with compute partitioning (SPX/DPX/CPX) and CU masking.
"""

import functools
import glob

# KFD `device_id` (PCI DID) -> model. MI300X/MI325X (both 304 CU) and
# MI350X/MI355X (both 256 CU) are only separable here, not by CU count.
_DID_TO_MODEL = {
    0x74A1: "mi300x",  # MI300X (gfx942, 304 CU)
    0x74A5: "mi325x",  # MI325X (gfx942, 304 CU)
    0x74A2: "mi308x",  # MI308X (gfx942, 80 CU)
    0x75A0: "mi350x",  # MI350X (gfx950, 256 CU)
    0x75A3: "mi355x",  # MI355X (gfx950, 256 CU)
}

# gfx_target_version -> model, for parts whose DID is not enumerated above.
_ARCH_TO_MODEL = {
    90500: "mi355x",  # gfx950
}


@functools.lru_cache(maxsize=1)
def topology():
    """(cu_count, gfx_target_version, device_id) of the first GPU node from KFD
    sysfs. gfx_target_version is e.g. 90402 (gfx942), 90500 (gfx950); device_id
    is the PCI DID (e.g. 0x74a2 = MI308X). Homogeneous host assumed. Returns
    (0, 0, 0) if sysfs is unavailable (no KFD mounted)."""
    for props in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            vals = {}
            with open(props) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        vals[parts[0]] = int(parts[1])
            simd = vals.get("simd_count", 0)
            if simd <= 0:  # CPU / non-GPU node
                continue
            spc = vals.get("simd_per_cu", 0) or 1
            return (
                simd // spc,
                vals.get("gfx_target_version", 0),
                vals.get("device_id", 0),
            )
        except Exception:
            continue
    return 0, 0, 0


def cu_count():
    return topology()[0]


def detect_model():
    """Model name for the local GPU (e.g. 'mi300x'), or None if unknown."""
    _, gfx, did = topology()
    model = _DID_TO_MODEL.get(did)
    return model if model is not None else _ARCH_TO_MODEL.get(gfx)
