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
import logging

# KFD `device_id` (PCI DID) -> model. MI300X/MI325X (both 304 CU) and
# MI350X/MI355X (both 256 CU) are only separable here, not by CU count.
_DID_TO_MODEL = {
    0x74A1: "mi300x",  # MI300X (gfx942, 304 CU)
    0x74A9: "mi300x",  # MI300X HF (gfx942, 304 CU)
    0x74A5: "mi325x",  # MI325X (gfx942, 304 CU)
    0x74B5: "mi300x",  # MI300X VF
    0x74BD: "mi300x",  # MI300X HF
    0x74B9: "mi325x",  # MI325X VF
    0x74A2: "mi308x",  # MI308X (gfx942, 80 CU)
    0x74A8: "mi308x",  # MI308X HF (gfx942, 80 CU)
    0x75A0: "mi350x",  # MI350X (gfx950, 256 CU)
    0x75A3: "mi355x",  # MI355X (gfx950, 256 CU)
}

# gfx_target_version -> model, for parts whose DID is not enumerated above.
_ARCH_TO_MODEL = {
    90500: "mi355x",  # gfx950
}


@functools.lru_cache(maxsize=1)
def _first_gpu_node():
    """KFD sysfs properties of the first GPU node, or {} without KFD."""
    for props in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            vals = {}
            with open(props) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        vals[parts[0]] = int(parts[1])
            if vals.get("simd_count", 0) > 0:  # skip CPU / non-GPU nodes
                return vals
        except Exception:
            continue
    return {}


def topology():
    """(cu_count, gfx_target_version, device_id) of the first GPU node from KFD
    sysfs. gfx_target_version is e.g. 90402 (gfx942), 90500 (gfx950); device_id
    is the PCI DID (e.g. 0x74a2 = MI308X). Homogeneous host assumed. Returns
    (0, 0, 0) if sysfs is unavailable (no KFD mounted)."""
    vals = _first_gpu_node()
    if not vals:
        return 0, 0, 0
    spc = vals.get("simd_per_cu", 0) or 1
    return (
        vals["simd_count"] // spc,
        vals.get("gfx_target_version", 0),
        vals.get("device_id", 0),
    )


def asic_revision():
    """The first GPU's ASIC revision, or None without KFD. KFD capability bits 25:22,
    what HIP reports as hipDeviceAttributeAsicRevision."""
    cap = _first_gpu_node().get("capability")
    return None if cap is None else (cap >> 22) & 0xF


def cu_count():
    return topology()[0]


logger = logging.getLogger(__name__)


def fit_blocks_to_cus(t: dict) -> dict:
    """Cap a tuning entry's block counts (single-shot and schedule) at the CU count,
    in place. A gfx125x EP kernel holds a CU per block and waits for its whole grid,
    so a grid wider than the GPU never finishes."""
    cu = cu_count()
    blocks = [t["dispatch_block_num"], t["combine_block_num"]]
    blocks += [max(b[1], b[3]) for b in t["schedule"] or ()]
    if cu <= 0 or max(blocks) <= cu:
        return t
    _warn_capped(max(blocks), cu)
    t["dispatch_block_num"] = min(t["dispatch_block_num"], cu)
    t["combine_block_num"] = min(t["combine_block_num"], cu)
    if t["schedule"]:
        t["schedule"] = tuple(
            (edge, min(db, cu), dw, min(cb, cu), cw)
            for edge, db, dw, cb, cw in t["schedule"]
        )
    return t


@functools.lru_cache(maxsize=None)
def _warn_capped(blocks: int, cu: int) -> None:
    logger.warning("tuned EP grid of %d blocks capped at this GPU's %d CUs", blocks, cu)


def detect_model():
    """Model name for the local GPU (e.g. 'mi300x'), or None if unknown."""
    _, gfx, did = topology()
    model = _DID_TO_MODEL.get(did)
    return model if model is not None else _ARCH_TO_MODEL.get(gfx)


def arch_name():
    """Arch name for the local GPU (e.g. 'gfx950'), or None if unavailable.

    Decoded from KFD's gfx_target_version (90402 -> gfx942, 120500 -> gfx1250),
    so it costs a sysfs read. mori.jit.config.detect_gpu_arch answers the same
    question by shelling out to rocm_agent_enumerator/rocminfo, which is the
    right thing when the answer has to match what the compiler will target; this
    is for labelling a tuning table, where a subprocess per import is not.
    """
    version = topology()[1]
    if not version:
        return None
    return f"gfx{version // 10000}{version // 100 % 100}{version % 100:x}"
