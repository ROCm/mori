#!/usr/bin/env python3
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
import argparse
import os
import random
import time
import json
import math
import signal
from enum import Enum
from typing import List, Tuple

import torch
import torch.distributed as dist

from tests.python.utils import get_free_port, TorchDistContext
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    RdmaBackendConfig,
    set_log_level,
)

# ---------------------------
# Utilities and Types
# ---------------------------


class EngineRole(Enum):
    INITIATOR = 0
    TARGET = 1


def send_bytes(b: bytes, dst: int):
    t = torch.ByteTensor(list(b))
    length_tensor = torch.IntTensor([t.numel()])
    dist.send(length_tensor, dst=dst)
    dist.send(t, dst=dst)


def recv_bytes(src: int) -> bytes:
    length_tensor = torch.IntTensor([0])
    dist.recv(length_tensor, src=src)
    length = length_tensor.item()
    t = torch.ByteTensor(length)
    dist.recv(t, src=src)
    return bytes(t.tolist())


def occasional(prob: float) -> bool:
    return random.random() < prob


# ---------------------------
# Stress Tester
# ---------------------------


class MoriIoStress:
    def __init__(
        self,
        host: str,
        port: int,
        node_rank: int,
        rank_in_node: int,
        max_buffer_size: int,
        max_transfer_batch_size: int,
        num_initiator_dev: int,
        num_target_dev: int,
        num_qp_per_transfer: int,
        num_worker_threads: int,
        iters: int,
        duration_sec: float,
        warmup_iters: int,
        op_bias_read: float,
        enable_session_prob: float,
        enable_batch_prob: float,
        sleep_us_between_iters: int,
        log_level: str,
        max_latency_samples: int,
        random_seed: int,
        stats_interval_sec: float,
        output_dir: str | None,
        log_every_n_iters: int,
    ):
        # Roles and ranks
        self.host = host
        self.port = port
        self.node_rank = node_rank
        self.role_rank = rank_in_node
        self.num_initiator_dev = num_initiator_dev
        self.num_target_dev = num_target_dev
        assert (
            self.num_initiator_dev == self.num_target_dev
        ), "num_initiator_dev must equal num_target_dev"

        if self.node_rank == 0:
            self.global_rank = self.role_rank
            self.role = EngineRole.INITIATOR
        else:
            self.global_rank = self.role_rank + self.num_initiator_dev
            self.role = EngineRole.TARGET

        # Buffers
        self.max_buffer_size = max_buffer_size  # in bytes == in elements (float8)
        self.max_transfer_batch_size = max_transfer_batch_size

        # RDMA config
        self.num_qp_per_transfer = num_qp_per_transfer
        self.num_worker_threads = num_worker_threads

        # Run control
        self.iters = iters
        self.duration_sec = duration_sec
        self.warmup_iters = max(0, warmup_iters)
        self.op_bias_read = max(0.0, min(1.0, op_bias_read))
        self.enable_session_prob = max(0.0, min(1.0, enable_session_prob))
        self.enable_batch_prob = max(0.0, min(1.0, enable_batch_prob))
        self.sleep_us_between_iters = max(0, sleep_us_between_iters)
        self.stats_interval_sec = max(0.0, stats_interval_sec)
        self.log_every_n_iters = max(0, log_every_n_iters)
        self.last_stats_flush = None
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Logging
        set_log_level(log_level)

        # Seeding
        if random_seed is not None and random_seed >= 0:
            seed = (random_seed + self.global_rank) % 2**31
            random.seed(seed)
            torch.manual_seed(seed)
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

        # Torch and memory
        self.world_size = self.num_initiator_dev + self.num_target_dev
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for this stress test but not available"
            )
        dev_count = torch.cuda.device_count()
        if self.role_rank >= dev_count:
            raise RuntimeError(
                f"Requested device index {self.role_rank} but only {dev_count} CUDA devices present"
            )
        self.device = torch.device("cuda", self.role_rank)

        # Allocate large tensor: float8 => 1 byte/elem
        self.tensor = torch.randn(
            self.max_buffer_size * self.max_transfer_batch_size
        ).to(self.device, dtype=torch.float8_e4m3fnuz)

        # Stats
        self.stats = {
            "single_ok": 0,
            "batch_ok": 0,
            "single_fail": 0,
            "batch_fail": 0,
            "bytes_total": 0,
            "lat_count": 0,
            "lat_mean": 0.0,
            "lat_M2": 0.0,
            "lat_min": float("inf"),
            "lat_max": 0.0,
            # reservoir sample list (will be capped later if enabled)
            "lat_samples": [],
            "lat_reservoir_max": (
                max_latency_samples
                if max_latency_samples and max_latency_samples > 0
                else None
            ),
        }
        self._terminate = False

        # Register signal handlers (best-effort; safe in main process only)
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except Exception:
            pass

    def _handle_signal(self, signum, frame):  # noqa
        print(f"Received signal {signum}, requesting graceful shutdown...")
        self._terminate = True

    def _record_latency_us(self, value_us: float):
        s = self.stats
        s["lat_count"] += 1
        count = s["lat_count"]
        delta = value_us - s["lat_mean"]
        s["lat_mean"] += delta / count
        delta2 = value_us - s["lat_mean"]
        s["lat_M2"] += delta * delta2
        if value_us < s["lat_min"]:
            s["lat_min"] = value_us
        if value_us > s["lat_max"]:
            s["lat_max"] = value_us
        # Reservoir sampling if enabled
        reservoir_cap = s.get("lat_reservoir_max")
        if reservoir_cap:
            samples = s["lat_samples"]
            if len(samples) < reservoir_cap:
                samples.append(value_us)
            else:
                # Replace with decreasing probability
                j = random.randint(0, count - 1)
                if j < reservoir_cap:
                    samples[j] = value_us

    def _latency_stats_snapshot(self):
        s = self.stats
        if s["lat_count"] < 2:
            std = 0.0
        else:
            std = math.sqrt(s["lat_M2"] / (s["lat_count"] - 1))
        return {
            "count": s["lat_count"],
            "mean_us": s["lat_mean"],
            "std_us": std,
            "min_us": 0.0 if s["lat_min"] == float("inf") else s["lat_min"],
            "max_us": s["lat_max"],
            "sample_us": s["lat_samples"],
        }

    def _snapshot_all(self, runtime_s: float):
        lat = self._latency_stats_snapshot()
        return {
            "role": self.role.name,
            "role_rank": self.role_rank,
            "global_rank": self.global_rank,
            "bytes_total": self.stats["bytes_total"],
            "single_ok": self.stats["single_ok"],
            "single_fail": self.stats["single_fail"],
            "batch_ok": self.stats["batch_ok"],
            "batch_fail": self.stats["batch_fail"],
            "latency": lat,
            "runtime_s": runtime_s,
            "timestamp": time.time(),
        }

    def _maybe_flush_stats(self, start_time: float, it: int, force: bool = False):
        if self.role is not EngineRole.INITIATOR:
            return
        now = time.time()
        if self.last_stats_flush is None:
            self.last_stats_flush = now
        due_time = force or (
            self.stats_interval_sec > 0
            and (now - self.last_stats_flush) >= self.stats_interval_sec
        )
        due_iter = force or (
            self.log_every_n_iters > 0 and it % self.log_every_n_iters == 0
        )
        if not (due_time or due_iter):
            return
        snap = self._snapshot_all(runtime_s=now - start_time)
        if due_iter and not due_time:
            # only lightweight log
            print(
                f"Iter {it} bytes {snap['bytes_total']:,} lat_mean_us {snap['latency']['mean_us']:.1f} failures S:{snap['single_fail']} B:{snap['batch_fail']}"
            )
        else:
            print(
                f"[STATS] t={snap['runtime_s']:.0f}s iter={it} bytes={snap['bytes_total']:,} lat_mean_us={snap['latency']['mean_us']:.1f}"
            )
        if self.output_dir:
            path = os.path.join(self.output_dir, f"stats_rank{self.global_rank}.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(snap, f, indent=2)
            except Exception as e:  # noqa
                print(f"Warning: failed to write stats file {path}: {e}")
        self.last_stats_flush = now

    def _print_config(self):
        print("MORI-IO Stress Test Config:")
        print(f"  role: {self.role.name}")
        print(f"  role_rank: {self.role_rank}")
        print(f"  global_rank: {self.global_rank} / world_size {self.world_size}")
        print(f"  host: {self.host}, port: {self.port}")
        print(
            f"  num_initiator_dev: {self.num_initiator_dev}, num_target_dev: {self.num_target_dev}"
        )
        print(f"  max_buffer_size (B): {self.max_buffer_size}")
        print(f"  max_transfer_batch_size: {self.max_transfer_batch_size}")
        print(f"  num_qp_per_transfer: {self.num_qp_per_transfer}")
        print(f"  num_worker_threads: {self.num_worker_threads}")
        print(
            f"  iters: {self.iters}, duration_sec: {self.duration_sec}, warmup_iters: {self.warmup_iters}"
        )
        print(f"  op_bias_read: {self.op_bias_read} (probability of read)")
        print(
            f"  enable_session_prob: {self.enable_session_prob}, enable_batch_prob: {self.enable_batch_prob}"
        )
        print(f"  sleep_us_between_iters: {self.sleep_us_between_iters}")
        print()

    def initialize(self):
        # Create engine and RDMA backend
        io_cfg = IOEngineConfig(host=self.host, port=self.port)
        self.engine = IOEngine(key=f"{self.role.name}-{self.role_rank}", config=io_cfg)

        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer=self.num_qp_per_transfer,
            post_batch_size=-1,
            num_worker_threads=self.num_worker_threads,
        )
        self.engine.create_backend(BackendType.RDMA, rdma_cfg)

        # Exchange engine descriptors
        engine_desc_bytes = self.engine.get_engine_desc().pack()
        if self.role is EngineRole.INITIATOR:
            for i in range(self.num_target_dev):
                send_bytes(engine_desc_bytes, self.num_initiator_dev + i)
            for i in range(self.num_target_dev):
                peer_engine_desc_bytes = recv_bytes(self.num_initiator_dev + i)
                peer_engine_desc = EngineDesc.unpack(peer_engine_desc_bytes)
                self.engine.register_remote_engine(peer_engine_desc)
        else:
            for i in range(self.num_initiator_dev):
                peer_engine_desc_bytes = recv_bytes(i)
                peer_engine_desc = EngineDesc.unpack(peer_engine_desc_bytes)
                self.engine.register_remote_engine(peer_engine_desc)
            for i in range(self.num_initiator_dev):
                send_bytes(engine_desc_bytes, i)

        # Register memory
        self.mem = self.engine.register_torch_tensor(self.tensor)

        # Exchange mem desc per role_rank peer
        if self.role is EngineRole.TARGET:
            mem_desc = self.mem.pack()
            send_bytes(mem_desc, self.role_rank)
        else:
            target_mem_desc_bytes = recv_bytes(self.num_initiator_dev + self.role_rank)
            self.target_mem = MemoryDesc.unpack(target_mem_desc_bytes)
            # Create a reusable session for this pair
            self.sess = self.engine.create_session(self.mem, self.target_mem)

    def _random_msg_size(self) -> int:
        # Random power-of-two size between 2^3 and max_buffer_size (inclusive)
        min_pow = 3
        max_pow = max(min_pow, int(self.max_buffer_size).bit_length() - 1)
        sz = 2 ** random.randint(min_pow, max_pow)
        return min(sz, self.max_buffer_size)

    def _random_batch(self, msg_size: int) -> Tuple[List[int], List[int], List[int]]:
        # Generate offsets and sizes for a batch that fits in buffer
        # Limit batch size so offset+size <= buffer_size
        max_possible = min(
            self.max_transfer_batch_size, max(1, self.max_buffer_size // msg_size)
        )
        if max_possible <= 0:
            max_possible = 1
        batch_n = random.randint(1, max_possible)
        offsets = [i * msg_size for i in range(batch_n)]
        sizes = [msg_size for _ in range(batch_n)]
        return offsets, offsets, sizes

    def _do_single_ops(
        self, is_read: bool, use_session: bool, msg_size: int
    ) -> Tuple[bool, int]:
        # Issue N single ops equal to a random batch length
        ok = True
        bytes_moved = 0

        max_possible = min(
            self.max_transfer_batch_size, max(1, self.max_buffer_size // msg_size)
        )
        n = random.randint(1, max_possible)
        status_list = []
        uids = [self.engine.allocate_transfer_uid() for _ in range(n)]

        for i in range(n):
            offset = i * msg_size
            if use_session:
                func = self.sess.read if is_read else self.sess.write
                st = func(offset, offset, msg_size, uids[i])
            else:
                func = self.engine.read if is_read else self.engine.write
                st = func(self.mem, offset, self.target_mem, offset, msg_size, uids[i])
            status_list.append(st)
            bytes_moved += msg_size

        for st in status_list:
            while st.InProgress():
                time.sleep(0.00005)  # 50 microseconds
            ok = ok and st.Succeeded()

        return ok, bytes_moved

    def _do_batch_ops(
        self, is_read: bool, use_session: bool, msg_size: int
    ) -> Tuple[bool, int]:
        offsets_src, offsets_dst, sizes = self._random_batch(msg_size)
        uid = self.engine.allocate_transfer_uid()
        if use_session:
            func = self.sess.batch_read if is_read else self.sess.batch_write
            st = func(offsets_src, offsets_dst, sizes, uid)
        else:
            func = self.engine.batch_read if is_read else self.engine.batch_write
            st = func(
                [self.mem],
                [offsets_src],
                [self.target_mem],
                [offsets_dst],
                [sizes],
                [uid],
            )[0]

        while st.InProgress():
            time.sleep(0.00005)
        ok = st.Succeeded()
        bytes_moved = sum(sizes)
        return ok, bytes_moved

    def _iteration(self) -> None:
        # One stress iteration on initiator side
        if self.role is EngineRole.TARGET:
            return

        is_read = random.random() < self.op_bias_read
        use_session = random.random() < self.enable_session_prob
        use_batch = random.random() < self.enable_batch_prob
        msg_size = self._random_msg_size()

        start = time.time()
        bytes_moved = 0
        ok = True

        if use_batch:
            res_ok, b = self._do_batch_ops(is_read, use_session, msg_size)
            self.stats["batch_ok" if res_ok else "batch_fail"] += 1
        else:
            res_ok, b = self._do_single_ops(is_read, use_session, msg_size)
            self.stats["single_ok" if res_ok else "single_fail"] += 1
        ok = ok and res_ok
        bytes_moved += b

        end = time.time()
        if self.warmup_done:
            self.stats["bytes_total"] += bytes_moved
            self._record_latency_us((end - start) * 1e6)

    def run(self):
        with TorchDistContext(
            rank=self.global_rank,
            world_size=self.world_size,
            master_addr=None,
            master_port=None,
            device_id=self.role_rank,
            backend="gloo",
        ):
            self._print_config()
            self.initialize()

            # Warmup: initiator runs some stable ops, target just sits
            self.warmup_done = False
            dist.barrier()
            warmup_rounds = self.warmup_iters
            if self.role is EngineRole.INITIATOR:
                for _ in range(warmup_rounds):
                    # deterministic small warmup
                    _ = self._do_single_ops(
                        is_read=True,
                        use_session=True,
                        msg_size=min(4096, self.max_buffer_size),
                    )
            dist.barrier()
            self.warmup_done = True

            # Stress loop
            start_time = time.time()
            it = 0
            try:
                while True:
                    if self.role is EngineRole.INITIATOR:
                        self._iteration()
                        self._maybe_flush_stats(start_time, it)

                    if self._terminate:
                        print("Termination requested.")
                        break

                    # Heartbeat (any role)
                    if getattr(self, "heartbeat_interval_sec", 0) > 0:
                        now = time.time()
                        if getattr(self, "_last_heartbeat", None) is None:
                            self._last_heartbeat = now
                        elif (
                            now - self._last_heartbeat
                        ) >= self.heartbeat_interval_sec:
                            if self.role is EngineRole.TARGET:
                                print(
                                    f"[HEARTBEAT] role=TARGET rank={self.global_rank} t={int(now - start_time)}s"
                                )
                            else:
                                # initiator prints brief stats
                                snap = self._snapshot_all(runtime_s=now - start_time)
                                print(
                                    f"[HEARTBEAT] iter={it} bytes={snap['bytes_total']:,} mean_lat_us={snap['latency']['mean_us']:.1f} fails={snap['single_fail']+snap['batch_fail']}"
                                )
                            self._last_heartbeat = now

                    it += 1
                    if self.sleep_us_between_iters > 0:
                        time.sleep(self.sleep_us_between_iters / 1e6)

                    # Stop conditions
                    if (
                        self.duration_sec > 0
                        and (time.time() - start_time) >= self.duration_sec
                    ) or self._terminate:
                        break
                    if self.iters > 0 and it >= self.iters:
                        break
                dist.barrier()
                # final flush
                self._maybe_flush_stats(start_time, it, force=True)

            except KeyboardInterrupt:
                pass

            # Final stats (initiator only)
            if self.role is EngineRole.INITIATOR:
                lat_snapshot = self._latency_stats_snapshot()
                avg_us = lat_snapshot["mean_us"] if lat_snapshot["count"] else 0.0
                min_us = lat_snapshot["min_us"]
                max_us = lat_snapshot["max_us"]

                total_mb = self.stats["bytes_total"] / 1e6
                total_s = max(1e-9, (time.time() - start_time))
                bw_gbps = (self.stats["bytes_total"] * 8.0) / 1e9 / total_s
                bw_gBps = (self.stats["bytes_total"]) / 1e9 / total_s

                print("\n==== Stress Summary (Initiator) ====")
                print(
                    f"  Total bytes: {self.stats['bytes_total']:,} B ({total_mb:.2f} MB)"
                )
                print(f"  Time: {total_s:.3f} s")
                print(f"  Throughput: {bw_gbps:.3f} Gb/s, {bw_gBps:.3f} GB/s")
                print(
                    f"  Latency (us): avg {avg_us:.2f}, min {min_us:.2f}, max {max_us:.2f}"
                )
                print(
                    f"  Single ok/fail: {self.stats['single_ok']} / {self.stats['single_fail']}"
                )
                print(
                    f"  Batch  ok/fail: {self.stats['batch_ok']} / {self.stats['batch_fail']}"
                )
                print("===================================\n")

                # Final JSON summary
                final_snap = self._snapshot_all(runtime_s=time.time() - start_time)
                final_snap["throughput_gbps"] = bw_gbps
                final_snap["throughput_GBps"] = bw_gBps
                final_snap["end_time"] = time.time()
                if self.output_dir:
                    final_path = os.path.join(
                        self.output_dir, f"final_rank{self.global_rank}.json"
                    )
                    try:
                        with open(final_path, "w", encoding="utf-8") as f:
                            json.dump(final_snap, f, indent=2)
                        print(f"Final summary written to {final_path}")
                    except Exception as e:  # noqa
                        print(f"Warning: could not write final summary: {e}")

                # Exit code strategy: success=0 if no failures
                self.exit_code = 0
                if self.stats["single_fail"] > 0 or self.stats["batch_fail"] > 0:
                    self.exit_code = 2
                if self._terminate:
                    # Distinguish external termination
                    self.exit_code = max(self.exit_code, 130)
            else:
                self.exit_code = 0


# ---------------------------
# CLI
# ---------------------------


def parse_args():
    p = argparse.ArgumentParser(description="MORI-IO Stress Test")
    p.add_argument(
        "--host",
        type=str,
        required=True,
        help="Host IP for MORI IO engine OOB communication",
    )
    p.add_argument(
        "--max-buffer-size",
        type=int,
        default=2**20,
        help="Max message size in bytes (and elements), default 1 MiB",
    )
    p.add_argument(
        "--max-transfer-batch-size",
        type=int,
        default=256,
        help="Max number of transfers per batch",
    )
    p.add_argument("--num-initiator-dev", type=int, default=1)
    p.add_argument("--num-target-dev", type=int, default=1)
    p.add_argument("--num-qp-per-transfer", type=int, default=1)
    p.add_argument("--num-worker-threads", type=int, default=1)
    p.add_argument(
        "--iters", type=int, default=0, help="Number of iterations to run; 0 to ignore"
    )
    p.add_argument(
        "--duration-sec",
        type=float,
        default=30.0,
        help="Run duration in seconds; 0 to ignore",
    )
    p.add_argument("--warmup-iters", type=int, default=8)
    p.add_argument(
        "--op-bias-read",
        type=float,
        default=0.5,
        help="Probability to choose read vs write",
    )
    p.add_argument(
        "--enable-session-prob",
        type=float,
        default=0.6,
        help="Probability to use session",
    )
    p.add_argument(
        "--enable-batch-prob",
        type=float,
        default=0.6,
        help="Probability to use batch API",
    )
    p.add_argument(
        "--sleep-us-between-iters", type=int, default=0, help="Sleep between iterations"
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Log level: trace, debug, info, warning, error, critical",
    )
    p.add_argument(
        "--max-latency-samples",
        type=int,
        default=10000,
        help="Max latency samples to retain in reservoir (0 disables)",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Base random seed (per-rank offset added); -1 to disable",
    )
    p.add_argument(
        "--stats-interval-sec",
        type=float,
        default=60.0,
        help="Interval in seconds for periodic stats flush (0 disables)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write periodic and final JSON stats",
    )
    p.add_argument(
        "--log-every-n-iters",
        type=int,
        default=0,
        help="If >0, log lightweight progress every N iterations (initiator)",
    )
    p.add_argument(
        "--heartbeat-interval-sec",
        type=float,
        default=300.0,
        help="Heartbeat print interval (all roles) to show liveness; 0 disables",
    )
    return p.parse_args()


def launch_stress(local_rank: int, node_rank: int, args):
    port = get_free_port()
    stress = MoriIoStress(
        host=args.host,
        port=port,
        node_rank=node_rank,
        rank_in_node=local_rank,
        max_buffer_size=args.max_buffer_size,
        max_transfer_batch_size=args.max_transfer_batch_size,
        num_initiator_dev=args.num_initiator_dev,
        num_target_dev=args.num_target_dev,
        num_qp_per_transfer=args.num_qp_per_transfer,
        num_worker_threads=args.num_worker_threads,
        iters=args.iters,
        duration_sec=args.duration_sec,
        warmup_iters=args.warmup_iters,
        op_bias_read=args.op_bias_read,
        enable_session_prob=args.enable_session_prob,
        enable_batch_prob=args.enable_batch_prob,
        sleep_us_between_iters=args.sleep_us_between_iters,
        log_level=args.log_level,
        max_latency_samples=args.max_latency_samples,
        random_seed=args.random_seed,
        stats_interval_sec=args.stats_interval_sec,
        output_dir=args.output_dir,
        log_every_n_iters=args.log_every_n_iters,
    )
    stress.heartbeat_interval_sec = max(0.0, args.heartbeat_interval_sec)
    stress._last_heartbeat = None
    stress.run()

    code = getattr(stress, "exit_code", 0)
    print(f"EXIT_CODE:{code}")


def main():
    args = parse_args()
    num_node = int(os.environ["WORLD_SIZE"])
    assert num_node == 2, "Stress test requires WORLD_SIZE=2 (initiator + target nodes)"
    node_rank = int(os.environ["RANK"])
    nprocs = args.num_initiator_dev if node_rank == 0 else args.num_target_dev

    torch.multiprocessing.spawn(
        launch_stress,
        args=(node_rank, args),
        nprocs=nprocs,
        join=True,
    )


if __name__ == "__main__":
    main()
