# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Restore the missing metric helper from dispatch.md's 2026-09-30 method.

The imported helper was absent from commit 33ae5bf9. This implementation follows
the documented formulas and retains the original samples for independent replay.
It uses only the Python standard library; it does not sample or launch kernels.
"""

import math
from numbers import Real


def _integer(value, name, minimum):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value != int(value)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def summarize_metric_samples(
    gathered,
    *,
    num_passes,
    num_rounds,
    num_warmup,
    drop_rounds,
    max_tokens,
    topk,
    ll,
):
    """Return (rounded tuning metrics, evidence) from CPU all-gather rows.

    Each rank contributes four metadata values followed by interleaved
    dispatch/combine microseconds in pass-major, round-major order. Metadata is
    received tokens, algorithm RDMA tokens, dispatch bytes/token and combine
    bytes/token. Bandwidth is averaged after dividing each rank's own bytes by
    each retained latency. The LL scale is rank 0's, applied after that average.
    """
    num_passes = _integer(num_passes, "num_passes", 1)
    num_rounds = _integer(num_rounds, "num_rounds", 1)
    num_warmup = _integer(num_warmup, "num_warmup", 0)
    drop_rounds = _integer(drop_rounds, "drop_rounds", 0)
    max_tokens = _integer(max_tokens, "max_tokens", 0)
    topk = _integer(topk, "topk", 1)
    if drop_rounds >= num_rounds:
        raise ValueError("drop_rounds must leave at least one retained round per pass")
    if not isinstance(ll, bool):
        raise ValueError("ll must be a boolean")
    try:
        rows = [list(row) for row in gathered]
    except TypeError as exc:
        raise ValueError("gathered must contain rank rows") from exc
    if not rows:
        raise ValueError("gathered must contain at least one rank")

    row_length = 4 + 2 * num_passes * num_rounds
    for rank, row in enumerate(rows):
        if len(row) != row_length:
            raise ValueError(
                f"rank {rank}: expected {row_length} values, received {len(row)}"
            )
        for column, (name, minimum) in enumerate(
            (
                ("recv_tokens", 0),
                ("rdma_algo_tokens", 0),
                ("dispatch_bytes_per_token", 1),
                ("combine_bytes_per_token", 1),
            )
        ):
            row[column] = _integer(row[column], f"rank {rank} {name}", minimum)
        for column in range(4, row_length):
            value = row[column]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"rank {rank}, sample value {column - 4}: "
                    "latency must be finite and positive"
                )
            row[column] = float(value)

    samples_per_rank = num_passes * (num_rounds - drop_rounds)
    sample_count = len(rows) * samples_per_rank
    ll_scale_rank0 = max_tokens * topk / (rows[0][0] + 1)
    metrics = {}
    unrounded = {}
    for phase_index, phase in enumerate(("dispatch", "combine")):
        latencies, rdma_bandwidths, xgmi_bandwidths = [], [], []
        for row in rows:
            recv_tokens, rdma_tokens = row[:2]
            width = row[2 + phase_index]
            for pass_index in range(num_passes):
                for round_index in range(drop_rounds, num_rounds):
                    offset = 4 + 2 * (pass_index * num_rounds + round_index)
                    latency = row[offset + phase_index]
                    latencies.append(latency)
                    rdma_bandwidths.append(rdma_tokens * width / (1000 * latency))
                    xgmi_bandwidths.append(recv_tokens * width / (1000 * latency))
        latency = math.fsum(latencies) / sample_count
        rdma = math.fsum(rdma_bandwidths) / sample_count
        xgmi = math.fsum(xgmi_bandwidths) / sample_count
        ll_bandwidth = xgmi * ll_scale_rank0
        values = {
            "bandwidth_gbps": ll_bandwidth if ll else rdma,
            "avg_rdma_bandwidth_gbps": rdma,
            "avg_xgmi_bandwidth_gbps": xgmi,
            "avg_ll_bandwidth_gbps": ll_bandwidth,
            "avg_latency_us": latency,
        }
        unrounded[phase] = values
        metrics[phase] = {
            **{name: round(value, 2) for name, value in values.items()},
            "bandwidth_metric": "grand_mean",
        }

    evidence = {
        "schema_version": 1,
        "method": "dispatch.md:2026-09-30/grand_mean",
        "helper_provenance": "restored from documented method; absent in 33ae5bf9",
        "parameters": {
            "num_passes": num_passes,
            "num_rounds": num_rounds,
            "num_warmup": num_warmup,
            "drop_rounds": drop_rounds,
            "max_tokens": max_tokens,
            "topk": topk,
            "ll": ll,
            "world_size": len(rows),
        },
        "row_layout": {
            "metadata": [
                "recv_tokens",
                "rdma_algo_tokens",
                "dispatch_bytes_per_token",
                "combine_bytes_per_token",
            ],
            "samples": "pass-major, round-major, [dispatch_us, combine_us]",
            "includes_dropped_rounds": True,
        },
        "rank_rows": rows,
        "retained_samples_per_rank_per_phase": samples_per_rank,
        "retained_samples_per_phase": sample_count,
        "ll_scale_rank0": ll_scale_rank0,
        "metrics_unrounded": unrounded,
    }
    return metrics, evidence
