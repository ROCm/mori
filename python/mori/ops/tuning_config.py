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

Tuning configs are stored as two separate JSON files per (gpu_arch,
gpu_model, kernel_type, ep_size) combination:

  {arch}_{model}_{kernel}_ep{n}_dispatch.json
  {arch}_{model}_{kernel}_ep{n}_combine.json

Dispatch files contain rules keyed by (dtype, num_tokens, hidden_dim, topk).
Combine files contain rules keyed by (dtype, num_tokens, hidden_dim, topk,
zero_copy, quant_type).  This separation allows dispatch rules to be
shared across different quant_type / zero_copy configurations.

``topk`` (num_experts_per_token) is optional: rules written before it was
introduced have no ``topk`` field and act as wildcards, so old configs keep
working.  It matters because two models can share a hidden_dim yet route a
very different number of experts per token (DeepSeek-V4-Pro top-6 vs
Kimi-K3 top-16 at hidden 7168), which changes the per-rank traffic and thus
the best block/warp geometry.
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

from mori.ops import utils as gpu_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified dtype registry
# ---------------------------------------------------------------------------

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

_KERNEL_TYPE_NAMES = frozenset(
    {"IntraNode", "InterNode", "InterNodeV1", "InterNodeV1LL", "AsyncLL", "IntraNodeLL"}
)

# Single source for the quant-type mapping. Keyed by EpDispatchCombineQuantType
# member name; the accepted config strings are derived rather than restated,
# because holding the two in separate literals is what let them drift apart
# (fp4_blockwise reached dispatch_combine.py but never the enum branch here).
_QUANT_ENUM_TO_CONFIG_STR = {
    "None_": "none",
    "Fp8DirectCast": "fp8_direct_cast",
    "Fp8BlockwiseQuant": "fp8_blockwise",
    "Fp4BlockwiseQuant": "fp4_blockwise",
}

_QUANT_TYPE_CONFIG_STRS = frozenset(_QUANT_ENUM_TO_CONFIG_STR.values())


def kernel_type_to_config_str(kernel_type) -> str:
    """Normalize kernel_type (str or enum) to a config string like 'IntraNode'."""
    name = getattr(kernel_type, "name", None) or str(kernel_type)
    if name in _KERNEL_TYPE_NAMES:
        return name
    raise ValueError(
        f"Unknown kernel_type: {kernel_type!r}, "
        f"expected one of {sorted(_KERNEL_TYPE_NAMES)}"
    )


def quant_type_to_config_str(quant_type) -> str:
    """Normalize quant_type (str or enum) to a combine quant config string."""
    if isinstance(quant_type, str):
        s = quant_type.strip().lower()
        if s in _QUANT_TYPE_CONFIG_STRS:
            return s
        if s == "none_":
            return "none"
    else:
        name = getattr(quant_type, "name", str(quant_type))
        if name in _QUANT_ENUM_TO_CONFIG_STR:
            return _QUANT_ENUM_TO_CONFIG_STR[name]
    raise ValueError(f"Unknown quant_type: {quant_type!r}")


def dtype_to_config_str(dtype: torch.dtype) -> str:
    """Convert a torch dtype to its config string representation."""
    return DTYPE_TO_CONFIG_STR[dtype]


# ---------------------------------------------------------------------------
# GPU model detection
# ---------------------------------------------------------------------------

_SUPPORTED_VERSION = "1.0"
_TUNING_CONFIGS_DIR = Path(__file__).parent / "tuning_configs"

_gpu_model_cache: str | None = None
_gpu_model_detected: bool = False


def detect_gpu_model() -> str | None:
    """Detect GPU model, e.g. 'mi300x', 'mi308x'. KFD-sysfs PCI DID first (exact,
    needs no HIP context); the torch device name is only parsed for parts the DID
    table doesn't know yet."""
    global _gpu_model_cache, _gpu_model_detected
    if _gpu_model_detected:
        return _gpu_model_cache
    _gpu_model_detected = True

    _gpu_model_cache = gpu_utils.detect_model()
    if _gpu_model_cache is not None:
        return _gpu_model_cache

    import re

    try:
        name = torch.cuda.get_device_properties(0).name.lower()
        m = re.search(r"\bmi\d+\w*", name)
        if m:
            _gpu_model_cache = m.group(0)
    except Exception:
        pass
    return _gpu_model_cache


# ---------------------------------------------------------------------------
# Rule validation — dispatch and combine have different required fields
#
# ``topk`` is deliberately NOT required: rules predating it are treated as
# topk wildcards (see _match_rules).
# ---------------------------------------------------------------------------

_DISPATCH_RULE_REQUIRED = frozenset(
    {
        "dtype",
        "num_tokens",
        "hidden_dim",
        "block_num",
        "rdma_block_num",
        "warp_per_block",
    }
)

_COMBINE_RULE_REQUIRED = frozenset(
    {
        "dtype",
        "num_tokens",
        "hidden_dim",
        "zero_copy",
        "quant_type",
        "block_num",
        "rdma_block_num",
        "warp_per_block",
    }
)

# ---------------------------------------------------------------------------
# File naming — files split by phase (dispatch / combine), no quant in name
# ---------------------------------------------------------------------------


def build_config_filename(
    gpu_arch: str,
    kernel_type: str,
    ep_size: int,
    gpu_model: str | None = None,
    phase: str = "dispatch",
) -> str:
    """Build config filename: ``{arch}_{model}_{kernel}_ep{n}_{phase}.json``."""
    parts = [gpu_arch]
    if gpu_model:
        parts.append(gpu_model)
    parts.append(f"{kernel_type}_ep{ep_size}_{phase}")
    return "_".join(parts) + ".json"


def _find_fallback_config(
    gpu_arch: str,
    kernel_type: str,
    ep_size: int,
    phase: str,
) -> Path | None:
    """Find any config file matching the same (arch, kernel, ep, phase)
    but different gpu_model."""
    suffix = f"_{kernel_type}_ep{ep_size}_{phase}.json"
    prefix = f"{gpu_arch}_"
    if not _TUNING_CONFIGS_DIR.is_dir():
        return None
    for f in sorted(_TUNING_CONFIGS_DIR.iterdir()):
        if f.name.startswith(prefix) and f.name.endswith(suffix) and f.is_file():
            return f
    return None


def config_path_for(
    gpu_arch: str,
    kernel_type: str,
    ep_size: int,
    gpu_model: str | None = None,
    phase: str = "dispatch",
) -> Path:
    """Resolve config file path for a given phase.

    Priority:
    1. MORI_EP_TUNING_CONFIG env var (exact path override, ignores phase)
    2. Exact match with gpu_model
    3. Fallback: any file with same (arch, kernel, ep, phase) but different model
    """
    env_override = os.environ.get("MORI_EP_TUNING_CONFIG")
    if env_override:
        return Path(env_override)
    exact = _TUNING_CONFIGS_DIR / build_config_filename(
        gpu_arch,
        kernel_type,
        ep_size,
        gpu_model,
        phase,
    )
    if exact.is_file():
        return exact
    fallback = _find_fallback_config(gpu_arch, kernel_type, ep_size, phase)
    if fallback is not None:
        logger.info(
            "No tuning config for gpu_model=%s phase=%s, using fallback: %s",
            gpu_model,
            phase,
            fallback.name,
        )
        return fallback
    return exact


# ---------------------------------------------------------------------------
# LaunchParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchParams:
    block_num: int
    rdma_block_num: int
    warp_per_block: int


# ---------------------------------------------------------------------------
# Save reporting
#
# BANDWIDTH_METRIC_KEY records how a rule's bandwidth_gbps was computed, so a
# re-tune can tell whether the old value is comparable with the new one. Rules
# written before this existed carry no key at all, which reads as "unknown" and
# is correctly treated as not comparable with anything.
#
# A tuning run overwrites the matching rule unconditionally -- see
# save_tuning_result. Two tuning runs are rarely comparable (different host,
# ROCm version, machine load, time), so no numeric rule can reliably decide
# "is this new result good enough to keep", and a wrong decision in the
# *keep the old rule* direction is invisible: you get a stale config and no
# indication your run was discarded. Overwriting instead makes the change
# reviewable in the JSON's git diff, which is where it can actually be judged.
#
# These print rather than logger.info deliberately. This library's logger sits
# at WARNING by default, so an info-level line here would never reach the
# person running the tuning -- exactly the silent-discard failure the gate
# used to have.
# ---------------------------------------------------------------------------


BANDWIDTH_METRIC_KEY = "bandwidth_metric"

#: Value for BANDWIDTH_METRIC_KEY when bandwidth_gbps is the mean over all
#: ranks and rounds -- the "Average" row of the table tuning prints.
BANDWIDTH_METRIC_GRAND_MEAN = "grand_mean"


def _rule_geometry(rule: dict) -> str:
    """block/warp/rdma of a rule, in the order tuning prints them."""
    return (
        f"block={rule.get('block_num')} "
        f"warp={rule.get('warp_per_block')} "
        f"rdma={rule.get('rdma_block_num')}"
    )


def _format_delta(
    old: float | None, new: float | None, unit: str, comparable: bool = True
) -> str:
    """`old -> new (+x.x%)`, or a bare value when there is nothing to compare.

    ``comparable=False`` still shows both numbers but drops the percentage:
    a percentage between two differently-defined quantities reads as a
    performance change when it is only a change of definition.
    """
    if new is None:
        return "n/a"
    if not old:
        return f"{new:.2f}{unit}"
    if not comparable:
        return (
            f"{old:.2f} -> {new:.2f}{unit} "
            f"(old value used a different metric, not comparable)"
        )
    pct = (new - old) / old * 100.0
    return f"{old:.2f} -> {new:.2f}{unit} ({pct:+.1f}%)"


def _rule_latency(rule: dict) -> float | None:
    """A rule's latency, whichever of the two spellings it uses.

    The inter-node tuner writes avg_latency_us, the intra-node one
    latency_us. Reading only the first left every intra-node save reporting
    "latency: n/a" -- dropping the field that matters most on the one line
    that exists to make the change reviewable.
    """
    for key in ("avg_latency_us", "latency_us"):
        value = rule.get(key)
        if value is not None:
            return value
    return None


def _report_rule_change(phase: str, merge_key, old_rule: dict, new_rule: dict) -> None:
    old_geo, new_geo = _rule_geometry(old_rule), _rule_geometry(new_rule)
    geo = (
        f"{old_geo}  ->  {new_geo}" if old_geo != new_geo else f"{new_geo} (unchanged)"
    )
    # Inter-node bandwidth_gbps changed meaning (slowest-rank mean -> grand
    # mean); a rule written before that carries no marker. Percentages across
    # the two are meaningless and read as a big win -- grand mean >=
    # slowest-rank mean by construction -- which is exactly backwards for a
    # report meant to help judge whether to keep the new result.
    #
    # Both sides missing the marker means neither side changed definition, so
    # they *are* comparable: that is the intra-node path, whose bandwidth_gbps
    # was never redefined. Only a mismatch indicates a redefinition.
    old_metric = old_rule.get(BANDWIDTH_METRIC_KEY)
    new_metric = new_rule.get(BANDWIDTH_METRIC_KEY)
    bw_comparable = old_metric == new_metric
    print(
        f"[mori-tuning] REPLACED {phase} rule {merge_key}\n"
        f"               config:  {geo}\n"
        f"               latency: "
        f"{_format_delta(_rule_latency(old_rule), _rule_latency(new_rule), 'us')}\n"
        f"               bw:      "
        f"{_format_delta(old_rule.get('bandwidth_gbps'), new_rule.get('bandwidth_gbps'), ' GB/s', bw_comparable)}",
        flush=True,
    )


def _report_rule_added(phase: str, merge_key, rule: dict) -> None:
    print(
        f"[mori-tuning] ADDED {phase} rule {merge_key}\n"
        f"               config:  {_rule_geometry(rule)}\n"
        f"               latency: {_format_delta(None, _rule_latency(rule), 'us')}\n"
        f"               bw:      {_format_delta(None, rule.get('bandwidth_gbps'), ' GB/s')}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# TuningConfigManager
# ---------------------------------------------------------------------------


class TuningConfigManager:
    """Loads, caches, and queries tuning configurations.

    Each instance holds dispatch_rules and combine_rules loaded from
    two separate JSON files (one per phase).
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
        gpu_model: str | None = None,
    ) -> "TuningConfigManager":
        cache_key = f"{gpu_arch}_{gpu_model or ''}_{kernel_type}_ep{ep_size}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        dispatch_path = config_path_for(
            gpu_arch,
            kernel_type,
            ep_size,
            gpu_model,
            "dispatch",
        )
        combine_path = config_path_for(
            gpu_arch,
            kernel_type,
            ep_size,
            gpu_model,
            "combine",
        )
        dispatch_rules = cls._load_phase_rules(
            dispatch_path,
            gpu_arch,
            "dispatch",
            _DISPATCH_RULE_REQUIRED,
        )
        combine_rules = cls._load_phase_rules(
            combine_path,
            gpu_arch,
            "combine",
            _COMBINE_RULE_REQUIRED,
        )
        instance = cls(dispatch_rules, combine_rules)
        cls._cache[cache_key] = instance
        return instance

    @classmethod
    def _load_phase_rules(
        cls,
        path: Path,
        expected_gpu_arch: str,
        phase: str,
        required_fields: frozenset[str],
    ) -> list[dict]:
        if not path.is_file():
            logger.debug("Tuning config not found: %s", path)
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load tuning config %s: %s", path, exc)
            return []

        version = data.get("version")
        if version != _SUPPORTED_VERSION:
            logger.warning(
                "Unsupported tuning config version %r in %s (expected %s)",
                version,
                path,
                _SUPPORTED_VERSION,
            )
            return []

        file_gpu_arch = data.get("gpu_arch")
        if file_gpu_arch and file_gpu_arch != expected_gpu_arch:
            logger.warning(
                "GPU arch mismatch in %s: file says %s, detected %s",
                path,
                file_gpu_arch,
                expected_gpu_arch,
            )

        raw_rules = data.get("rules", [])
        valid: list[dict] = []
        for i, rule in enumerate(raw_rules):
            missing = required_fields - rule.keys()
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

        if phase == "dispatch":
            valid.sort(
                key=lambda r: (
                    r["dtype"],
                    r["hidden_dim"],
                    r.get("topk") or 0,
                    r["num_tokens"],
                )
            )
        else:
            valid.sort(
                key=lambda r: (
                    r["dtype"],
                    r["hidden_dim"],
                    r.get("topk") or 0,
                    r["zero_copy"],
                    r["quant_type"],
                    r["num_tokens"],
                )
            )
        logger.debug("Loaded %d %s rules from %s", len(valid), phase, path)
        return valid

    # ------------------------------------------------------------------
    # Runtime lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _match_rules(
        sorted_rules: list[dict],
        dtype_str: str,
        num_tokens: int,
        hidden_dim: int,
        zero_copy: bool | None,
        quant_type: str | None,
        topk: int | None,
        *,
        require_hidden_match: bool,
        require_topk_match: bool,
    ) -> LaunchParams | None:
        """Inner lookup over *sorted_rules* with optional hidden_dim/topk filtering.

        Rules are sorted by num_tokens ascending (within each dtype/hidden/topk
        group).  We pick the tightest ceiling match, and if num_tokens
        exceeds the largest recorded value we clamp to that rule.

        With ``require_topk_match`` a rule only matches when its ``topk``
        equals the requested one; rules without a ``topk`` field never match in
        that pass and are picked up by the relaxed pass instead.
        """
        last_match: dict | None = None
        for rule in sorted_rules:
            if rule["dtype"] != dtype_str:
                continue
            if require_hidden_match and rule["hidden_dim"] != hidden_dim:
                continue
            if require_topk_match and rule.get("topk") != topk:
                continue
            if zero_copy is not None and rule.get("zero_copy") != zero_copy:
                continue
            if quant_type is not None and rule.get("quant_type") != quant_type:
                continue
            last_match = rule
            if num_tokens <= rule["num_tokens"]:
                return LaunchParams(
                    block_num=rule["block_num"],
                    rdma_block_num=rule["rdma_block_num"],
                    warp_per_block=rule["warp_per_block"],
                )
        if last_match is not None:
            return LaunchParams(
                block_num=last_match["block_num"],
                rdma_block_num=last_match["rdma_block_num"],
                warp_per_block=last_match["warp_per_block"],
            )
        return None

    @staticmethod
    def lookup(
        sorted_rules: list[dict],
        dtype: torch.dtype,
        num_tokens: int,
        hidden_dim: int,
        zero_copy: bool | None = None,
        quant_type: str | None = None,
        topk: int | None = None,
    ) -> LaunchParams | None:
        """Find the best matching launch params.

        Exact-matches dtype, and (for combine) zero_copy and quant_type.
        For num_tokens, picks the tightest ceiling match; if num_tokens
        exceeds the largest recorded value the largest rule is used (clamp).
        Prefers an exact hidden_dim match, then an exact topk match, falling
        back in that order to rules that don't record the dimension.  ``topk``
        is only used as a filter when the caller supplies it.
        """
        dtype_str = DTYPE_TO_CONFIG_STR.get(dtype)
        if dtype_str is None:
            return None
        # (require_hidden, require_topk) in decreasing specificity.  hidden_dim
        # is relaxed last: it sets the per-token transport volume, so a rule
        # measured at the right hidden and a different top-k is a better guess
        # than one measured at the right top-k and the wrong hidden.
        passes = (
            ((True, True), (True, False), (False, True), (False, False))
            if topk is not None
            else ((True, False), (False, False))
        )
        for require_hidden, require_topk in passes:
            result = TuningConfigManager._match_rules(
                sorted_rules,
                dtype_str,
                num_tokens,
                hidden_dim,
                zero_copy,
                quant_type,
                topk,
                require_hidden_match=require_hidden,
                require_topk_match=require_topk,
            )
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------
    # Tuning result persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_tuning_result(
        path: str | Path,
        metadata: dict,
        entry: dict,
        phase: str = "dispatch",
    ) -> None:
        """Save or merge a single tuning rule into a phase JSON file.

        entry format for dispatch:
            {"dtype": "fp4", "num_tokens": 128, "hidden_dim": 3584, "topk": 8,
             "block_num": 64, "rdma_block_num": 0, "warp_per_block": 16,
             "bandwidth_gbps": 41.5}

        entry format for combine (adds zero_copy + quant_type):
            {"dtype": "bf16", "num_tokens": 128, "hidden_dim": 7168, "topk": 8,
             "zero_copy": true, "quant_type": "none",
             "block_num": 64, "rdma_block_num": 0, "warp_per_block": 16,
             "bandwidth_gbps": 97.75}

        ``topk`` participates in the merge key, so tuning the same shape at a
        different top-k adds a rule instead of overwriting the existing one.

        A matching rule is replaced unconditionally, and what changed is
        printed (see the "Save reporting" section above for why there is no
        keep-best comparison guarding this). ``bandwidth_gbps`` is recorded for
        readers, not consulted here -- nothing in the runtime lookup path reads
        it either, only block_num/rdma_block_num/warp_per_block are.
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
            data = {"version": _SUPPORTED_VERSION, **metadata, "rules": []}

        rules: list[dict] = data.setdefault("rules", [])

        def _dispatch_key(r):
            return (
                r.get("dtype"),
                r.get("num_tokens"),
                r.get("hidden_dim"),
                r.get("topk"),
            )

        def _combine_key(r):
            return (
                r.get("dtype"),
                r.get("num_tokens"),
                r.get("hidden_dim"),
                r.get("topk"),
                r.get("zero_copy"),
                r.get("quant_type"),
            )

        def _dispatch_sort(r):
            return (r["dtype"], r["hidden_dim"], r.get("topk") or 0, r["num_tokens"])

        def _combine_sort(r):
            return (
                r["dtype"],
                r["hidden_dim"],
                r.get("topk") or 0,
                r.get("zero_copy", False),
                r.get("quant_type", ""),
                r["num_tokens"],
            )

        if phase == "dispatch":
            merge_key = _dispatch_key(entry)
            match_fn = _dispatch_key
            sort_key = _dispatch_sort
        else:
            merge_key = _combine_key(entry)
            match_fn = _combine_key
            sort_key = _combine_sort

        matched_idx = None
        for i, rule in enumerate(rules):
            if match_fn(rule) == merge_key:
                matched_idx = i
                break

        if matched_idx is not None:
            _report_rule_change(phase, merge_key, rules[matched_idx], entry)
            rules[matched_idx] = entry
        else:
            _report_rule_added(phase, merge_key, entry)
            rules.append(entry)

        rules.sort(key=sort_key)
        data["rules"] = rules

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
