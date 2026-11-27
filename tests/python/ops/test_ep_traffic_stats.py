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
"""EP traffic statistics: the per-peer token counts the kernels keep on the device.

Metrics are decided once per process, so this file starts its own workers with
``MORI_METRICS_ENABLE=1``. ``MORI_TEST_EP_STATS_WORLD`` sets the number of ranks
(default 8); ``MORI_TEST_METRICS_DUMP=<directory>`` leaves each case's counts as JSON.
"""
import json
import os

import pytest
import torch

import mori
import mori.metrics
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    start_torch_dist_process_manager,
)
from tests.python.ops.test_dispatch_combine_intranode import _make_intranode_config
from tests.python.ops.test_dispatch_combine_internode_v1 import (
    _make_internode_v1_config,
)

ROUNDS = 3
HIDDEN = 512
EXPERTS_PER_TOKEN = 4
# bfloat16 tokens: hidden + expert indices + weights + source token id.
DISPATCH_BYTES_PER_TOKEN = (
    HIDDEN * 2 + EXPERTS_PER_TOKEN * 4 + EXPERTS_PER_TOKEN * 4 + 4
)
# hidden + weights.
COMBINE_BYTES_PER_TOKEN = HIDDEN * 2 + EXPERTS_PER_TOKEN * 4


def _world_size():
    return int(os.environ.get("MORI_TEST_EP_STATS_WORLD", "8"))


@pytest.fixture(scope="module")
def stats_workers():
    world = _world_size()
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    prev = os.environ.get("MORI_METRICS_ENABLE")
    os.environ["MORI_METRICS_ENABLE"] = "1"  # inherited by the spawned workers
    try:
        manager = start_torch_dist_process_manager(world_size=world)
    finally:
        if prev is None:
            os.environ.pop("MORI_METRICS_ENABLE", None)
        else:
            os.environ["MORI_METRICS_ENABLE"] = prev
    yield manager, world
    manager.shutdown()


def _run_intranode(rank, world_size):
    config = _make_intranode_config(
        rank, world_size, torch.bfloat16, HIDDEN, 128, 4, EXPERTS_PER_TOKEN
    )
    config.gpu_per_node = world_size
    op = mori.ops.EpDispatchCombineOp(config)
    # A kernel that times itself has its time series from the start, before any launch.
    timed = {
        s["labels"]["op"]
        for s in mori.metrics.series()
        if s["name"] == "mori_ep_kernel_seconds_total"
    }
    assert timed == {"dispatch", "combine"}, timed
    case = EpDispatchCombineTestCase(config)
    for _ in range(ROUNDS):
        # With result checks: the statistics must not disturb what EP computes.
        case.run_test_once(op, case.gen_test_data(), check_results=True)
    stats = op.get_traffic_stats()
    assert stats is not None, "metrics are on, the op must have statistics"
    series = [s for s in mori.metrics.series() if s["name"].startswith("mori_ep_")]
    return json.dumps(
        {
            "stats": {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()
            },
            "series": series,
        }
    )


# Sampling changed while the op runs: (one dispatch in this many counts, rounds).
SAMPLING_PHASES = ((1, 3), (0, 6), (4, 320), (1, 3))


def _run_intranode_sampling(rank, world_size):
    """What the counters moved by under each sampling period in turn."""
    config = _make_intranode_config(
        rank, world_size, torch.bfloat16, HIDDEN, 128, 4, EXPERTS_PER_TOKEN
    )
    config.gpu_per_node = world_size
    op = mori.ops.EpDispatchCombineOp(config)
    case = EpDispatchCombineTestCase(config)
    data = case.gen_test_data()

    def read():
        stats = op.get_traffic_stats()
        return {
            k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()
        }

    def moved(before, after):
        return {
            k: (
                [a - b for a, b in zip(after[k], before[k])]
                if isinstance(after[k], list)
                else after[k] - before[k]
            )
            for k in after
        }

    phases = []
    try:
        for period, rounds in SAMPLING_PHASES:
            now = mori.metrics.set_sampling(ep=period)
            assert now["ep"] == period, now
            published = [
                s["value"]
                for s in mori.metrics.series()
                if s["name"] == "mori_metrics_sample_period"
                and s["labels"]["module"] == "ep"
            ]
            assert published == [period], published
            before = read()
            for i in range(rounds):
                case.run_test_once(op, data, check_results=i < 3)
            phases.append(moved(before, read()))
    finally:
        mori.metrics.set_sampling(ep=1)
    return json.dumps({"phases": phases})


def _collect(manager, world):
    out = {}
    for _ in range(world):
        rank, result = manager.result_queue.get(timeout=900)
        try:
            out[rank] = json.loads(result)
        except (TypeError, ValueError):
            pytest.fail(f"rank {rank} failed:\n{result}", pytrace=False)
    dump_dir = os.environ.get("MORI_TEST_METRICS_DUMP")
    if dump_dir:
        case = (
            os.environ.get("PYTEST_CURRENT_TEST", "case").split("::")[-1].split(" ")[0]
        )
        with open(os.path.join(dump_dir, f"metrics_series_{case}.json"), "w") as f:
            json.dump({"test": case, "ranks": out}, f)
    return out


def test_intranode_counts_are_consistent_across_ranks(stats_workers):
    manager, world = stats_workers
    for _ in range(world):
        manager.task_queue.put((_run_intranode, (world,)))
    ranks = _collect(manager, world)
    stats = {r: ranks[r]["stats"] for r in range(world)}

    total = 0
    for a in range(world):
        assert stats[a]["dispatch_calls"] == ROUNDS
        assert stats[a]["combine_calls"] == ROUNDS
        # Some time, and nowhere near a second a launch.
        for op in ("dispatch", "combine"):
            assert 0 < stats[a][f"{op}_kernel_seconds"] < ROUNDS, (a, op)
        for b in range(world):
            sent = stats[a]["dispatch_sent"][b]
            assert sent == stats[b]["dispatch_recv"][a], (a, b)
            assert stats[a]["combine_sent"][b] == stats[a]["dispatch_recv"][b], (a, b)
            assert stats[a]["combine_recv"][b] == stats[a]["dispatch_sent"][b], (a, b)
            assert stats[a]["combine_sent"][b] == stats[b]["combine_recv"][a], (a, b)
            total += sent
    assert total > 0, "nothing was counted"

    gpus = set()
    for r in range(world):
        series = ranks[r]["series"]
        own = {s["labels"]["gpu"] for s in series}
        assert len(own) == 1, own
        gpu = own.pop()
        gpus.add(gpu)
        assert {s["labels"]["rank"] for s in series} == {str(r)}

        def value(name, **want):
            hits = [
                s["value"]
                for s in series
                if s["name"] == name
                and all(s["labels"].get(k) == v for k, v in want.items())
            ]
            assert len(hits) == 1, (name, want, hits)
            return hits[0]

        # Tokens that stay on the GPU are `local`, and are not XGMI traffic.
        assert (
            value(
                "mori_ep_tokens_total",
                op="dispatch",
                direction="tx",
                path="local",
                peer_gpu=gpu,
            )
            == stats[r]["dispatch_sent"][r]
        )
        assert not [
            s
            for s in series
            if s["name"] == "mori_ep_xgmi_bytes_total"
            and s["labels"]["peer_gpu"] == gpu
        ]
        xgmi_tokens = [
            s
            for s in series
            if s["name"] == "mori_ep_tokens_total" and s["labels"]["path"] == "xgmi"
        ]
        assert (
            len(xgmi_tokens) == (world - 1) * 4
        )  # peers x (dispatch, combine) x (tx, rx)
        assert not [s for s in series if s["labels"].get("path") == "rdma"]
        for s in xgmi_tokens:
            labels = s["labels"]
            per_token = (
                DISPATCH_BYTES_PER_TOKEN
                if labels["op"] == "dispatch"
                else COMBINE_BYTES_PER_TOKEN
            )
            assert (
                value(
                    "mori_ep_xgmi_bytes_total",
                    op=labels["op"],
                    direction=labels["direction"],
                    peer_gpu=labels["peer_gpu"],
                )
                == s["value"] * per_token
            )
        assert value("mori_ep_calls_total", op="dispatch", impl="v1") == ROUNDS
        for op in ("dispatch", "combine"):
            assert value(
                "mori_ep_kernel_seconds_total", op=op, impl="v1"
            ) == pytest.approx(stats[r][f"{op}_kernel_seconds"])
        assert value("mori_ep_calls_total", op="combine", kernel="intranode") == ROUNDS
    assert len(gpus) == world, gpus


# Inter-node kernel family, as the test config names it -> its `kernel` label.
INTERNODE_KERNELS = {
    "internode_v1": "internode_v1",
    "internode_v1_ll": "internode_v1ll",
}


def test_intranode_sampling_is_changed_while_the_op_runs(stats_workers):
    """One dispatch in N counts N-fold, its combine with it; 0 counts nothing;
    back at 1 the count is exact again."""
    manager, world = stats_workers
    for _ in range(world):
        manager.task_queue.put((_run_intranode_sampling, (world,)))
    ranks = _collect(manager, world)
    for r in range(world):
        first, off, sampled, last = ranks[r]["phases"]
        (_, n1), (_, _), (period, n), (_, n3) = SAMPLING_PHASES

        # Everything counts: this is what one round moves, the same every round.
        assert first["dispatch_calls"] == n1 and first["combine_calls"] == n1, (
            r,
            first,
        )
        per_round_sent = [x // n1 for x in first["dispatch_sent"]]
        per_round_recv = [x // n1 for x in first["dispatch_recv"]]
        assert [x * n1 for x in per_round_sent] == first["dispatch_sent"], r
        assert sum(per_round_sent) > 0, r

        for key, value in off.items():
            assert value == 0 or value == [0] * world, (r, key, value)

        # One in `period`: whole rounds of weight `period`, an estimate of n.
        calls = sampled["dispatch_calls"]
        assert calls % period == 0, (r, calls)
        assert 0.7 * n < calls < 1.3 * n, (r, calls)
        assert sampled["combine_calls"] == calls, (r, sampled)
        assert sampled["dispatch_sent"] == [x * calls for x in per_round_sent], r
        assert sampled["dispatch_recv"] == [x * calls for x in per_round_recv], r
        assert sampled["combine_sent"] == sampled["dispatch_recv"], r
        assert sampled["combine_recv"] == sampled["dispatch_sent"], r
        # Per counted call, of the order of the exact phase's (three cold launches).
        for op in ("dispatch", "combine"):
            per_call = sampled[f"{op}_kernel_seconds"] / calls
            exact = first[f"{op}_kernel_seconds"] / n1
            assert 0 < per_call < 20 * exact, (r, op, per_call, exact)

        assert last["dispatch_calls"] == n3 and last["combine_calls"] == n3, (r, last)
        assert last["dispatch_sent"] == [x * n3 for x in per_round_sent], r

    # The ranks counted the same rounds: the sampled counts agree across ranks.
    sampled = [ranks[r]["phases"][2] for r in range(world)]
    for a in range(world):
        assert sampled[a]["dispatch_calls"] == sampled[0]["dispatch_calls"], a
        for b in range(world):
            assert sampled[a]["dispatch_sent"][b] == sampled[b]["dispatch_recv"][a], (
                a,
                b,
            )
            assert sampled[a]["combine_sent"][b] == sampled[b]["combine_recv"][a], (
                a,
                b,
            )


def _run_internode_v1(rank, world_size, gpu_per_node, kernel="internode_v1"):
    config = _make_internode_v1_config(
        rank,
        world_size,
        kernel,
        torch.bfloat16,
        HIDDEN,
        128,
        4,
        EXPERTS_PER_TOKEN,
        gpu_per_node,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    timed = {
        s["labels"]["op"]
        for s in mori.metrics.series()
        if s["name"] == "mori_ep_kernel_seconds_total"
    }
    assert timed == {"dispatch", "combine"}, timed
    case = EpDispatchCombineTestCase(config)
    for _ in range(ROUNDS):
        case.run_test_once(op, case.gen_test_data(), check_results=True)
    stats = op.get_traffic_stats()
    assert stats is not None, "metrics are on, the op must have statistics"
    series = [
        s
        for s in mori.metrics.series()
        if s["name"].startswith(("mori_ep_", "mori_gpu_rdma_"))
    ]
    return json.dumps(
        {
            "stats": {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()
            },
            "series": series,
        }
    )


def _check_hops(stats, world):
    """A rank's count towards a peer is that peer's count from it, and combine
    returns what dispatch delivered."""
    total = 0
    for a in range(world):
        assert stats[a]["dispatch_calls"] == ROUNDS
        assert stats[a]["combine_calls"] == ROUNDS
        for op in ("dispatch", "combine"):
            assert 0 < stats[a][f"{op}_kernel_seconds"] < ROUNDS, (a, op)
        for b in range(world):
            sent = stats[a]["dispatch_sent"][b]
            assert sent == stats[b]["dispatch_recv"][a], (a, b)
            assert stats[a]["combine_sent"][b] == stats[a]["dispatch_recv"][b], (a, b)
            assert stats[a]["combine_recv"][b] == stats[a]["dispatch_sent"][b], (a, b)
            total += sent
    assert total > 0, "nothing was counted"


@pytest.mark.parametrize("kernel", list(INTERNODE_KERNELS))
def test_internode_v1_kernel_on_one_node(stats_workers, kernel):
    """The inter-node kernels with every rank on one node: no RDMA hop exists."""
    manager, world = stats_workers
    for _ in range(world):
        manager.task_queue.put((_run_internode_v1, (world, world, kernel)))
    ranks = _collect(manager, world)
    _check_hops({r: ranks[r]["stats"] for r in range(world)}, world)
    for r in range(world):
        series = ranks[r]["series"]
        assert not [s for s in series if s["labels"].get("path") == "rdma"]
        calls = [
            s
            for s in series
            if s["name"] == "mori_ep_calls_total"
            and s["labels"]["kernel"] == INTERNODE_KERNELS[kernel]
        ]
        assert sorted(s["value"] for s in calls) == [ROUNDS, ROUNDS]
        for op in ("dispatch", "combine"):
            seconds = [
                s["value"]
                for s in series
                if s["name"] == "mori_ep_kernel_seconds_total"
                and s["labels"]["op"] == op
            ]
            assert seconds == [
                pytest.approx(ranks[r]["stats"][f"{op}_kernel_seconds"])
            ], (r, op)


def _rdma_transport():
    """Whether same-host ranks talk RDMA (MORI_DISABLE_P2P=1) rather than P2P."""
    return os.environ.get("MORI_DISABLE_P2P", "").lower() in ("1", "true", "on")


@pytest.mark.parametrize("kernel", list(INTERNODE_KERNELS))
def test_internode_v1_two_nodes_on_one_box(stats_workers, kernel):
    """Two emulated nodes (gpu_per_node = world / 2): the inter-node hop is only
    ever to the rank with the same place in the other node.

    Same-host peers are reached over P2P unless MORI_DISABLE_P2P=1: only then is
    there RDMA for shmem to count, at least the dispatch tokens times their size."""
    manager, world = stats_workers
    if world < 4:
        pytest.skip("needs at least 4 ranks")
    per_node = world // 2
    for _ in range(world):
        manager.task_queue.put((_run_internode_v1, (world, per_node, kernel)))
    ranks = _collect(manager, world)
    stats = {r: ranks[r]["stats"] for r in range(world)}
    _check_hops(stats, world)

    rdma_tokens = 0
    for a in range(world):
        proxy = (a + per_node) % world  # same place, other node
        for b in range(world):
            same_node = a // per_node == b // per_node
            if not same_node and b != proxy:
                assert stats[a]["dispatch_sent"][b] == 0, (a, b)
        rdma_tokens += stats[a]["dispatch_sent"][proxy]
        series = ranks[a]["series"]
        rdma = [
            s
            for s in series
            if s["name"] == "mori_ep_tokens_total"
            and s["labels"]["path"] == "rdma"
            and s["labels"]["op"] == "dispatch"
            and s["labels"]["direction"] == "tx"
        ]
        assert len(rdma) == 1 and rdma[0]["value"] == stats[a]["dispatch_sent"][proxy]
        puts = sum(
            s["value"]
            for s in series
            if s["name"] == "mori_gpu_rdma_bytes_total"
            and s["labels"]["api"] == "shmem"
            and s["labels"]["op"] == "put"
        )
        if _rdma_transport():
            assert (
                puts >= stats[a]["dispatch_sent"][proxy] * DISPATCH_BYTES_PER_TOKEN
            ), a
        else:
            assert puts == 0, "P2P transport: shmem moved nothing over RDMA"
        for s in series:
            if s["name"] != "mori_gpu_rdma_bytes_total":
                continue
            for key in ("src_ibdev", "src_netdev", "dst_ibdev", "dst_netdev"):
                assert s["labels"][key], (key, s["labels"])
    assert rdma_tokens > 0, "no token crossed the two nodes"
