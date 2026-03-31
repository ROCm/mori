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
"""EP Dispatch/Combine tuning configuration management.

Provides a unified dtype registry, JSON config file loading/saving,
and runtime launch parameter lookup for tuned kernel configurations.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified dtype registry
# ---------------------------------------------------------------------------
# Single source of truth for all dtype string representations.
# Add new dtypes here — all derived mappings update automatically.
#
# Fields: (torch_dtype, config_str, file_short_name, kernel_suffix)
#   config_str      — JSON fields & CLI args (matches benchmark _DATA_TYPE_MAP keys)
#   file_short_name — compact form for filenames
#   kernel_suffix   — HIP kernel function name suffix (matches _DTYPE_SUFFIX)

_DTYPE_REGISTRY: list[tuple] = [
    (torch.bfloat16, "bf16", "bf16", "bf16"),
]

try:
    _DTYPE_REGISTRY.append(
        (torch.float8_e4m3fnuz, "fp8_e4m3_fnuz", "fp8fnuz", "fp8_fnuz")
    )
except AttributeError:
    pass

try:
    _DTYPE_REGISTRY.append((torch.float8_e4m3fn, "fp8_e4m3", "fp8ocp", "fp8_ocp"))
except AttributeError:
    pass

try:
    _DTYPE_REGISTRY.append((torch.float4_e2m1fn_x2, "fp4", "fp4", "fp4"))
except AttributeError:
    pass

DTYPE_TO_CONFIG_STR: dict[torch.dtype, str] = {r[0]: r[1] for r in _DTYPE_REGISTRY}
CONFIG_STR_TO_DTYPE: dict[str, torch.dtype] = {r[1]: r[0] for r in _DTYPE_REGISTRY}
CONFIG_STR_TO_SHORT_NAME: dict[str, str] = {r[1]: r[2] for r in _DTYPE_REGISTRY}

_QUANT_SHORT_NAME: dict[str, str] = {"fp8_direct_cast": "fp8cast"}

_KERNEL_TYPE_NAMES = frozenset(
    {
        "IntraNode",
        "InterNode",
        "InterNodeV1",
        "InterNodeV1LL",
        "AsyncLL",
    }
)

_QUANT_TYPE_CONFIG_STRS = {"none", "fp8_direct_cast"}


def kernel_type_to_config_str(kernel_type) -> str:
    """Normalize kernel_type (str or enum) to a config string like 'IntraNode'."""
    name = getattr(kernel_type, "name", None) or str(kernel_type)
    if name in _KERNEL_TYPE_NAMES:
        return name
    raise ValueError(
        f"Unknown kernel_type: {kernel_type!r}, expected one of {sorted(_KERNEL_TYPE_NAMES)}"
    )


def quant_type_to_config_str(quant_type) -> str:
    """Normalize quant_type (str or enum) to a config string like 'none' or 'fp8_direct_cast'."""
    if isinstance(quant_type, str):
        s = quant_type.strip().lower()
        if s in _QUANT_TYPE_CONFIG_STRS:
            return s
        if s == "none_":
            return "none"
    else:
        name = getattr(quant_type, "name", str(quant_type))
        if name == "None_":
            return "none"
        if name == "Fp8DirectCast":
            return "fp8_direct_cast"
    raise ValueError(f"Unknown quant_type: {quant_type!r}")


_SUPPORTED_VERSION = "1.0"

_TUNING_CONFIGS_DIR = Path(__file__).parent / "tuning_configs"

_PHASE_RULE_REQUIRED = frozenset(
    {
        "dtype",
        "num_tokens",
        "hidden_dim",
        "block_num",
        "rdma_block_num",
        "warp_per_block",
    }
)


def dtype_to_config_str(dtype: torch.dtype) -> str:
    """Convert a torch dtype to its config string representation."""
    return DTYPE_TO_CONFIG_STR[dtype]


# ---------------------------------------------------------------------------
# File naming helpers — dtype is NOT in the filename
# ---------------------------------------------------------------------------


def build_config_filename(
    gpu_arch: str,
    kernel_type: str,
    ep_size: int,
    quant_type: str,
) -> str:
    """Build the JSON config filename (no dtype — dtype lives inside rules)."""
    name = f"{gpu_arch}_{kernel_type}_ep{ep_size}"
    if quant_type != "none":
        quant_short = _QUANT_SHORT_NAME.get(quant_type, quant_type)
        name += f"_{quant_short}"
    return name + ".json"


def config_path_for(
    gpu_arch: str,
    kernel_type: str,
    ep_size: int,
    quant_type: str,
) -> Path:
    """Resolve the config file path, honoring MORI_EP_TUNING_CONFIG override."""
    env_override = os.environ.get("MORI_EP_TUNING_CONFIG")
    if env_override:
        return Path(env_override)
    filename = build_config_filename(gpu_arch, kernel_type, ep_size, quant_type)
    return _TUNING_CONFIGS_DIR / filename


# ---------------------------------------------------------------------------
# LaunchParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchParams:
    block_num: int
    rdma_block_num: int
    warp_per_block: int


# ---------------------------------------------------------------------------
# TuningConfigManager
# ---------------------------------------------------------------------------


class TuningConfigManager:
    """Loads, caches, and queries per-file tuning configurations.

    Each file contains separate dispatch_rules and combine_rules lists.
    Rules carry their own dtype field, so dispatch and combine can use
    different dtypes independently.
    """

    _cache: ClassVar[dict[str, "TuningConfigManager"]] = {}

    def __init__(self, dispatch_rules: list[dict], combine_rules: list[dict]):
        self.dispatch_rules: list[dict] = dispatch_rules
        self.combine_rules: list[dict] = combine_rules

    @classmethod
    def get_instance(
        cls,
        gpu_arch: str,
        kernel_type: str,
        ep_size: int,
        quant_type: str,
    ) -> "TuningConfigManager":
        cache_key = build_config_filename(gpu_arch, kernel_type, ep_size, quant_type)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        path = config_path_for(gpu_arch, kernel_type, ep_size, quant_type)
        dispatch_rules, combine_rules = cls._load_rules(path, gpu_arch)
        instance = cls(dispatch_rules, combine_rules)
        cls._cache[cache_key] = instance
        return instance

    @classmethod
    def _load_rules(
        cls, path: Path, expected_gpu_arch: str
    ) -> tuple[list[dict], list[dict]]:
        if not path.is_file():
            logger.debug("Tuning config not found: %s", path)
            return [], []
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load tuning config %s: %s", path, exc)
            return [], []

        version = data.get("version")
        if version != _SUPPORTED_VERSION:
            logger.warning(
                "Unsupported tuning config version %r in %s (expected %s)",
                version,
                path,
                _SUPPORTED_VERSION,
            )
            return [], []

        file_gpu_arch = data.get("gpu_arch")
        if file_gpu_arch and file_gpu_arch != expected_gpu_arch:
            logger.warning(
                "GPU arch mismatch in %s: file says %s, detected %s",
                path,
                file_gpu_arch,
                expected_gpu_arch,
            )

        dispatch_rules = cls._validate_and_sort(
            data.get("dispatch_rules", []), path, "dispatch"
        )
        combine_rules = cls._validate_and_sort(
            data.get("combine_rules", []), path, "combine"
        )
        return dispatch_rules, combine_rules

    @classmethod
    def _validate_and_sort(cls, raw_rules: list, path: Path, phase: str) -> list[dict]:
        valid: list[dict] = []
        for i, rule in enumerate(raw_rules):
            missing = _PHASE_RULE_REQUIRED - rule.keys()
            if missing:
                logger.warning(
                    "Skipping %s rule %d in %s: missing fields %s",
                    phase,
                    i,
                    path,
                    missing,
                )
                continue
            valid.append(rule)
        valid.sort(key=lambda r: (r["dtype"], r["hidden_dim"], r["num_tokens"]))
        logger.debug("Loaded %d %s rules from %s", len(valid), phase, path)
        return valid

    # ------------------------------------------------------------------
    # Runtime lookup
    # ------------------------------------------------------------------

    @staticmethod
    def lookup(
        sorted_rules: list[dict],
        dtype: torch.dtype,
        num_tokens: int,
        hidden_dim: int,
    ) -> LaunchParams | None:
        """Find the best matching launch params.

        Exact-matches dtype and hidden_dim, then picks the tightest
        (smallest) rule num_tokens >= actual num_tokens (ceiling match).
        """
        dtype_str = DTYPE_TO_CONFIG_STR.get(dtype)
        if dtype_str is None:
            return None
        for rule in sorted_rules:
            if rule["dtype"] != dtype_str:
                continue
            if rule["hidden_dim"] != hidden_dim:
                continue
            if num_tokens <= rule["num_tokens"]:
                return LaunchParams(
                    block_num=rule["block_num"],
                    rdma_block_num=rule["rdma_block_num"],
                    warp_per_block=rule["warp_per_block"],
                )
        return None

    # ------------------------------------------------------------------
    # Tuning result persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_tuning_result(
        path: str | Path,
        metadata: dict,
        dispatch_entry: dict,
        combine_entry: dict,
    ) -> None:
        """Save or merge tuning results into a JSON config file.

        dispatch_entry / combine_entry format:
            {"dtype": "bf16", "num_tokens": 128, "hidden_dim": 7168,
             "block_num": 64, "rdma_block_num": 0, "warp_per_block": 16,
             "bandwidth_gbps": 312.5}

        Uses keep-best strategy per phase: only overwrites if new
        bandwidth is strictly higher. Writes atomically.
        """
        path = Path(path)

        if path.is_file():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = None
        else:
            data = None

        if data is None:
            data = {
                "version": _SUPPORTED_VERSION,
                **metadata,
                "dispatch_rules": [],
                "combine_rules": [],
            }

        for section_key, entry in [
            ("dispatch_rules", dispatch_entry),
            ("combine_rules", combine_entry),
        ]:
            rules: list[dict] = data.setdefault(section_key, [])
            merge_key = (entry["dtype"], entry["num_tokens"], entry["hidden_dim"])

            matched_idx = None
            for i, rule in enumerate(rules):
                if (
                    rule.get("dtype"),
                    rule.get("num_tokens"),
                    rule.get("hidden_dim"),
                ) == merge_key:
                    matched_idx = i
                    break

            if matched_idx is not None:
                old_bw = rules[matched_idx].get("bandwidth_gbps", 0)
                new_bw = entry.get("bandwidth_gbps", 0)
                if new_bw > old_bw:
                    rules[matched_idx] = entry
                    logger.info(
                        "Updated %s rule for %s (%.2f -> %.2f GB/s)",
                        section_key,
                        merge_key,
                        old_bw,
                        new_bw,
                    )
                else:
                    logger.info(
                        "Kept existing %s rule for %s "
                        "(existing %.2f >= new %.2f GB/s)",
                        section_key,
                        merge_key,
                        old_bw,
                        new_bw,
                    )
            else:
                rules.append(entry)
                logger.info("Added %s rule: %s", section_key, merge_key)

            rules.sort(key=lambda r: (r["dtype"], r["hidden_dim"], r["num_tokens"]))

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=path.stem
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
