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
"""UMBP (Unified Memory/Bandwidth Pool) Python bindings via mori."""
import os
import stat
from pathlib import Path


def _configure_packaged_spdk_proxy() -> None:
    if os.environ.get("UMBP_SPDK_PROXY_BIN"):
        return

    proxy_path = Path(__file__).resolve().parents[1] / "spdk_proxy"
    if not proxy_path.is_file():
        return

    try:
        mode = proxy_path.stat().st_mode
        exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        if (mode & exec_bits) != exec_bits:
            proxy_path.chmod(mode | exec_bits)
    except OSError:
        # Best effort only; if chmod fails we fall back to the existing
        # discovery logic in LocalStorageManager.
        pass

    if os.access(proxy_path, os.X_OK):
        os.environ["UMBP_SPDK_PROXY_BIN"] = str(proxy_path)


def _configure_packaged_umbp_master() -> None:
    if os.environ.get("UMBP_MASTER_BIN"):
        return

    master_path = Path(__file__).resolve().parents[1] / "umbp_master"
    if not master_path.is_file():
        return

    try:
        mode = master_path.stat().st_mode
        exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        if (mode & exec_bits) != exec_bits:
            master_path.chmod(mode | exec_bits)
    except OSError:
        pass

    if os.access(master_path, os.X_OK):
        os.environ["UMBP_MASTER_BIN"] = str(master_path)


_configure_packaged_spdk_proxy()
_configure_packaged_umbp_master()

from mori.cpp import (
    UMBPClient,
    UMBPConfig,
    UMBPCopyPipelineConfig,
    UMBPDistributedConfig,
    UMBPDramConfig,
    UMBPDurabilityMode,
    UMBPHostBufferBacking,
    UMBPHostBufferHandle,
    UMBPHostMemAllocator,
    UMBPIoBackend,
    UMBPIoConfig,
    UMBPRole,
    UMBPSsdConfig,
    UMBPEvictionConfig,
    UMBPDurabilityConfig,
    UMBPTierType,
    UMBPExternalKvMatch,
)

__all__ = [
    "UMBPClient",
    "UMBPConfig",
    "UMBPCopyPipelineConfig",
    "UMBPDistributedConfig",
    "UMBPDramConfig",
    "UMBPDurabilityConfig",
    "UMBPDurabilityMode",
    "UMBPHostBufferBacking",
    "UMBPHostBufferHandle",
    "UMBPHostMemAllocator",
    "UMBPIoBackend",
    "UMBPIoConfig",
    "UMBPRole",
    "UMBPSsdConfig",
    "UMBPEvictionConfig",
    "UMBPTierType",
    "UMBPExternalKvMatch",
]
