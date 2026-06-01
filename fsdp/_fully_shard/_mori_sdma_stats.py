import atexit
import os
from collections import Counter
from typing import Optional


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


_ENABLED = _env_enabled("MORI_FSDP_STATS")
try:
    _INTERVAL = int(os.environ.get("MORI_FSDP_STATS_INTERVAL", "0") or "0")
except ValueError:
    _INTERVAL = 0
_RANKS = os.environ.get("MORI_FSDP_STATS_RANKS", "0").strip().lower()

_counters: Counter[str] = Counter()
_size_counts: Counter[int] = Counter()
_copy_out_miss_reasons: Counter[str] = Counter()


def enabled() -> bool:
    return _ENABLED


def record_all_gather(numel: int) -> None:
    if not _ENABLED:
        return
    _counters["all_gather_calls"] += 1
    _counters["all_gather_numel"] += int(numel)
    _size_counts[int(numel)] += 1


def record_copy_in(numel: int, skipped: bool) -> None:
    if not _ENABLED:
        return
    key = "copy_in_skipped" if skipped else "copy_in_performed"
    _counters[key] += 1
    _counters[f"{key}_numel"] += int(numel)


def record_copy_out(
    numel: int,
    skipped: bool,
    reason: Optional[str] = None,
    hit_kind: Optional[str] = None,
) -> None:
    if not _ENABLED:
        return
    key = "copy_out_skipped" if skipped else "copy_out_performed"
    _counters[key] += 1
    _counters[f"{key}_numel"] += int(numel)
    _counters["no_copy_hit" if skipped else "no_copy_miss"] += 1
    if skipped and hit_kind is not None:
        _counters[f"{hit_kind}_hit"] += 1
        _counters[f"{hit_kind}_hit_numel"] += int(numel)
    if not skipped and reason is not None:
        _copy_out_miss_reasons[reason] += 1
    _maybe_dump()


def record_register(event: str) -> None:
    if not _ENABLED:
        return
    _counters[f"register_{event}"] += 1


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0") or "0")


def _should_print() -> bool:
    rank = _rank()
    return _RANKS == "all" or str(rank) in {r.strip() for r in _RANKS.split(",")}


def _maybe_dump() -> None:
    if _INTERVAL > 0 and _counters["all_gather_calls"] % _INTERVAL == 0:
        dump(prefix="[mori-fsdp-stats:interval]")


def dump(prefix: str = "[mori-fsdp-stats]") -> None:
    if not _ENABLED or not _should_print():
        return
    rank = _rank()
    calls = _counters["all_gather_calls"]
    if calls == 0:
        return
    top_sizes = ", ".join(
        f"{numel}:{count}" for numel, count in _size_counts.most_common(8)
    )
    miss_reasons = ", ".join(
        f"{reason}:{count}" for reason, count in _copy_out_miss_reasons.most_common()
    )
    print(
        f"{prefix} rank={rank} "
        f"all_gather_calls={calls} "
        f"all_gather_numel={_counters['all_gather_numel']} "
        f"copy_in_skipped={_counters['copy_in_skipped']} "
        f"copy_in_performed={_counters['copy_in_performed']} "
        f"copy_in_skipped_numel={_counters['copy_in_skipped_numel']} "
        f"copy_in_performed_numel={_counters['copy_in_performed_numel']} "
        f"copy_out_skipped={_counters['copy_out_skipped']} "
        f"copy_out_performed={_counters['copy_out_performed']} "
        f"no_copy_hit={_counters['no_copy_hit']} "
        f"no_copy_miss={_counters['no_copy_miss']} "
        f"param_contiguous_hit={_counters['param_contiguous_hit']} "
        f"single_param_no_copy_hit={_counters['single_param_no_copy_hit']} "
        f"copy_out_skipped_numel={_counters['copy_out_skipped_numel']} "
        f"copy_out_performed_numel={_counters['copy_out_performed_numel']} "
        f"param_contiguous_hit_numel={_counters['param_contiguous_hit_numel']} "
        f"single_param_no_copy_hit_numel={_counters['single_param_no_copy_hit_numel']} "
        f"register_cache_hit={_counters['register_cache_hit']} "
        f"register_existing_hit={_counters['register_existing_hit']} "
        f"register_call={_counters['register_call']} "
        f"top_all_gather_numel={{{top_sizes}}} "
        f"copy_out_miss_reasons={{{miss_reasons}}}",
        flush=True,
    )


if _ENABLED:
    atexit.register(dump)
