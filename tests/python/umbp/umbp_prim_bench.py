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
"""Atomic UMBP primitive benchmark, swept over concurrency.

One primitive per timed window and nothing else in it: every argument list is
built during an untimed prepare, so a measurement is the client call and not
the Python that assembles its arguments.  Five primitives:

    put         batch_put_from_ptr              whole object write
    get         batch_get_into_ptr              whole object read
    exists      batch_exists                    presence probe, moves no bytes
    put_ranges  batch_put_ranges_from_ptr       write R ranges of one object
    get_ranges  batch_get_ranges_into_ptr       read R ranges of one object

Each runs at every rank count in --ranks, which is the axis that matters most
here: a shared-lock control plane and an exclusive-lock one look identical at
one rank and diverge at eight.  Ranks are separate processes (the GIL makes
threads useless for this), started with multiprocessing's spawn so each builds
its own client, and held in lockstep by a Barrier around every timed window.

Example
-------
    python3 umbp_prim_bench.py umbp-server unix:///run/umbp/standalone/a.sock \\
        --ops put get exists put_ranges get_ranges --ranks 1 2 4 8
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

OPS = ("put", "get", "exists", "put_ranges", "get_ranges")


@dataclass(frozen=True)
class Config:
    backend: str
    address: str
    ops: tuple[str, ...]
    rank_counts: tuple[int, ...]
    keys_per_rank: int
    batch: int
    passes: int
    ranges: int
    range_bytes: int
    loc: str
    register: bool
    verify: bool
    key_prefix: str
    dram_capacity: int
    #: Fraction of each `exists` call's keys that name something never
    #: written.  0 probes only resident keys and never reaches the master.
    exists_absent_frac: float = 0.0
    #: Where to write the machine-readable result, for a caller that gates on
    #: these numbers rather than reading the table.  Empty means don't.
    json_path: str = ""

    @property
    def object_bytes(self) -> int:
        """One object is exactly its ranges, so whole and ranged ops move the
        same bytes for the same key and stay comparable."""
        return self.ranges * self.range_bytes

    def keys(
        self, op: str, ranks: int, rank: int, generation: int, absent: bool = False
    ) -> list[str]:
        # Disjoint per rank, as the linker's tp{r} suffix makes them: a shared
        # keyspace would let one rank's put satisfy another's get and turn a
        # concurrency sweep into a dedup measurement.
        #
        # The RANK COUNT is in the namespace too. Without it the ranks=2 step
        # rewrites the keys the ranks=1 step already wrote, so every put after
        # the first sweep step is an overwrite of a resident key -- which the
        # store short-circuits, making the write primitive get faster the more
        # concurrency it is under.
        # The OP is in the namespace as well, so the two write primitives do
        # not write each other's keys -- put and put_ranges both want a key
        # that does not exist yet, and whichever ran second would otherwise be
        # measuring an overwrite the store short-circuits.
        #
        # `absent` shifts the index past everything any op ever writes, so
        # those keys are guaranteed missing by construction.  Deriving them
        # from the same namespace keeps the key SHAPE identical -- same
        # length, same prefix -- so a probe that mixes the two is not
        # secretly measuring two different string costs.
        base = self.keys_per_rank if absent else 0
        return [
            f"{self.key_prefix}_{op}_n{ranks}_g{generation}_r{rank}_{base + i:07d}"
            for i in range(self.keys_per_rank)
        ]


# ---------------------------------------------------------------------------
#  Buffers
# ---------------------------------------------------------------------------


class HostBuffer:
    """Host DMA buffer from mori's allocator.

    Standalone-process mode resolves (region_base, offset) against the SERVER's
    mapping of this memory, so it must be shm-backed or nothing can map it --
    and getting that wrong is quiet, surfacing as a per-key false that looks
    exactly like the store refusing the operation.
    """

    def __init__(self, size: int, shm: bool):
        from mori.umbp import UMBPHostBufferBacking, UMBPHostMemAllocator

        self._allocator = UMBPHostMemAllocator()
        order = (
            [
                UMBPHostBufferBacking.AnonymousShmHugetlb,
                UMBPHostBufferBacking.AnonymousShm,
            ]
            if shm
            else [
                UMBPHostBufferBacking.AnonymousHugetlb,
                UMBPHostBufferBacking.Anonymous,
            ]
        )
        self._handle = None
        for backing in order:
            self._handle = self._allocator.alloc(size, backing, 2 << 20, -1, True)
            if self._handle:
                break
        if not self._handle:
            raise RuntimeError(f"host alloc({size}) failed")
        self.ptr = int(self._handle.ptr)
        self.size = size

    def write(self, offset: int, payload: bytes) -> None:
        ctypes.memmove(self.ptr + offset, payload, len(payload))

    def read(self, offset: int, size: int) -> bytes:
        return ctypes.string_at(self.ptr + offset, size)

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._allocator.free(self._handle)
            self._handle = None


class DeviceBuffer:
    """GPU buffer via raw HIP, so a read lands where the tree connector's reads
    land.  ctypes rather than torch keeps the bench usable in a bare container."""

    _D2H, _H2D = 2, 1

    def __init__(self, size: int, device: int):
        self._hip = ctypes.CDLL("libamdhip64.so")
        self._check(self._hip.hipSetDevice(ctypes.c_int(device)), "hipSetDevice")
        raw = ctypes.c_void_p()
        self._check(
            self._hip.hipMalloc(ctypes.byref(raw), ctypes.c_size_t(size)), "hipMalloc"
        )
        self._check(self._hip.hipMemset(raw, 0, ctypes.c_size_t(size)), "hipMemset")
        self.ptr = raw.value
        self.size = size
        self.device = device

    @staticmethod
    def device_count() -> int:
        hip = ctypes.CDLL("libamdhip64.so")
        n = ctypes.c_int(0)
        if hip.hipGetDeviceCount(ctypes.byref(n)) != 0:
            return 0
        return n.value

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} failed: hip rc={rc}")

    def write(self, offset: int, payload: bytes) -> None:
        self._check(
            self._hip.hipMemcpy(
                ctypes.c_void_p(self.ptr + offset),
                payload,
                ctypes.c_size_t(len(payload)),
                ctypes.c_int(self._H2D),
            ),
            "hipMemcpy H2D",
        )

    def read(self, offset: int, size: int) -> bytes:
        out = ctypes.create_string_buffer(size)
        self._check(
            self._hip.hipMemcpy(
                out,
                ctypes.c_void_p(self.ptr + offset),
                ctypes.c_size_t(size),
                ctypes.c_int(self._D2H),
            ),
            "hipMemcpy D2H",
        )
        return out.raw

    def close(self) -> None:
        ptr = getattr(self, "ptr", None)
        if ptr:
            self._hip.hipFree(ctypes.c_void_p(ptr))
            self.ptr = None


def make_buffer(cfg: Config, size: int, rank: int):
    if cfg.loc == "gpu":
        count = DeviceBuffer.device_count()
        if count == 0:
            raise RuntimeError("--loc gpu but no HIP device is visible")
        return DeviceBuffer(size, rank % count)
    return HostBuffer(size, shm=cfg.backend == "umbp-server")


# ---------------------------------------------------------------------------
#  Client
# ---------------------------------------------------------------------------


def make_client(cfg: Config):
    from mori.umbp import UMBPClient, UMBPConfig

    conf = UMBPConfig()
    if cfg.backend == "umbp-server":
        # Imported here, not at the top: a build without the standalone
        # bindings can still run the embedded backend, and an unconditional
        # import would turn that into an ImportError naming the wrong thing.
        from mori.umbp import UMBPStandaloneProcessConfig

        sp = UMBPStandaloneProcessConfig()
        sp.address = cfg.address
        sp.startup_timeout_ms = int(os.environ.get("UMBP_STANDALONE_TIMEOUT_MS", 15000))
        # standalone_process ONLY: the factory tests `distributed` first, so a
        # config carrying both quietly builds a DistributedClient instead.
        conf.standalone_process = sp
    else:  # umbp-local: in-process client, no server and no master
        conf.dram.capacity_bytes = cfg.dram_capacity
        conf.dram.use_hugepages = False
    return UMBPClient(conf)


def register(client, cfg: Config, buf) -> None:
    """Register the whole buffer once, the way the tree connector does.

    The connector calls the two-argument form, which leaves `loc` at its CPU
    default even for a device pointer -- mori detects the pointer itself.  This
    passes CPU for a GPU buffer too, deliberately: registering the way
    production registers keeps the bench on the path that is actually shipped.
    """
    if not cfg.register:
        return
    from mori.io import MemoryLocationType

    client.register_memory(buf.ptr, buf.size, MemoryLocationType.CPU, -1)


# ---------------------------------------------------------------------------
#  Operations
#
#  An Op is a prepare/invoke pair.  Everything a call needs is built in
#  prepare -- key lists, pointer lists, per-range size and offset tables -- so
#  the timed window holds one client call and nothing else.  Building those
#  lists inside the loop is what made the old bench's per-call numbers include
#  its own list comprehensions.
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    """One prepared primitive call."""

    invoke: Callable[[], object]
    keys: int
    nbytes: int
    #: The keys this call names, in slot order.  Verification needs them to
    #: know which payload belongs in which buffer slot.
    key_names: tuple[str, ...] = ()
    #: Per-key expected truth, when it is not "all of them".  A probe that
    #: deliberately names absent keys expects False for those, and a False
    #: there is the CORRECT answer -- scoring it as a failure would make the
    #: absent-key sweep unusable.  None means "expect every key to succeed".
    expect: tuple[bool, ...] | None = None


class Op:
    name: str
    #: A put must land on keys that do not exist yet, or the store short-circuits
    #: and the measurement is an overwrite of a resident key.
    fresh_keys_per_pass = False
    #: Reads need their objects present before the timed window opens.
    needs_seed = True

    def __init__(self, cfg: Config, client, buf, ranks: int, rank: int):
        self.cfg = cfg
        self.client = client
        self.buf = buf
        self.ranks = ranks
        self.rank = rank

    def slots(self, count: int) -> list[int]:
        return [self.buf.ptr + i * self.cfg.object_bytes for i in range(count)]

    def prepare(self, generation: int) -> list[Batch]:
        raise NotImplementedError


class PutOp(Op):
    name = "put"
    fresh_keys_per_pass = True
    needs_seed = False

    def prepare(self, generation: int) -> list[Batch]:
        cfg, out = self.cfg, []
        keys = cfg.keys(self.name, self.ranks, self.rank, generation)
        for lo in range(0, len(keys), cfg.batch):
            kb = keys[lo : lo + cfg.batch]
            ptrs = self.slots(len(kb))
            sizes = [cfg.object_bytes] * len(kb)
            out.append(
                Batch(
                    lambda k=kb, p=ptrs, s=sizes: self.client.batch_put_from_ptr(
                        k, p, s
                    ),
                    len(kb),
                    len(kb) * cfg.object_bytes,
                    key_names=tuple(kb),
                )
            )
        return out


class GetOp(Op):
    name = "get"

    def prepare(self, generation: int) -> list[Batch]:
        cfg, out = self.cfg, []
        keys = cfg.keys(self.name, self.ranks, self.rank, generation)
        for lo in range(0, len(keys), cfg.batch):
            kb = keys[lo : lo + cfg.batch]
            ptrs = self.slots(len(kb))
            sizes = [cfg.object_bytes] * len(kb)
            out.append(
                Batch(
                    lambda k=kb, p=ptrs, s=sizes: self.client.batch_get_into_ptr(
                        k, p, s
                    ),
                    len(kb),
                    len(kb) * cfg.object_bytes,
                    key_names=tuple(kb),
                )
            )
        return out


class ExistsOp(Op):
    """Presence probe, optionally mixed with keys that were never written.

    The mix matters more than it looks.  Under `local_first` a probe for a
    key this node holds is answered locally and the master is never asked, so
    an all-present probe measures only half the path.  A local miss proves
    nothing -- the key could live on a peer -- so every ABSENT key costs a
    master round trip.  That is the half a prefix cache actually pays on a
    cold prefix, and the half where a control-plane regression shows up.
    """

    name = "exists"

    def absent_slots(self, count: int) -> set[int]:
        """Which positions in a `count`-key call name an absent key.

        Spread evenly rather than blocked at one end, so no call is entirely
        one population and every call pays a mix.
        """
        n = int(round(count * self.cfg.exists_absent_frac))
        n = max(0, min(count, n))
        return {(i * count) // n for i in range(n)} if n else set()

    def prepare(self, generation: int) -> list[Batch]:
        cfg, out = self.cfg, []
        present = cfg.keys(self.name, self.ranks, self.rank, generation)
        # Indices at or past keys_per_rank are never written by any op, so
        # these are absent by construction rather than by hoping for a miss.
        absent = cfg.keys(self.name, self.ranks, self.rank, generation, absent=True)
        for lo in range(0, len(present), cfg.batch):
            kb = list(present[lo : lo + cfg.batch])
            holes = self.absent_slots(len(kb))
            for slot in holes:
                kb[slot] = absent[lo + slot]
            expect = tuple(slot not in holes for slot in range(len(kb)))
            # No bytes at all: the whole call is the control plane.
            out.append(
                Batch(
                    lambda k=kb: self.client.batch_exists(k),
                    len(kb),
                    0,
                    key_names=tuple(kb),
                    expect=expect,
                )
            )
        return out


class RangedOp(Op):
    """Shared geometry for the two ranged primitives.

    The ranges tile the object exactly, which the write side requires -- a put
    that left a hole would be read back as garbage rather than as a miss.
    """

    def tables(self, count: int):
        cfg = self.cfg
        sizes = [[cfg.range_bytes] * cfg.ranges] * count
        offsets = [[r * cfg.range_bytes for r in range(cfg.ranges)]] * count
        ptrs = [
            [base + r * cfg.range_bytes for r in range(cfg.ranges)]
            for base in self.slots(count)
        ]
        return ptrs, sizes, offsets


class PutRangesOp(RangedOp):
    name = "put_ranges"
    fresh_keys_per_pass = True
    needs_seed = False

    def prepare(self, generation: int) -> list[Batch]:
        cfg, out = self.cfg, []
        keys = cfg.keys(self.name, self.ranks, self.rank, generation)
        for lo in range(0, len(keys), cfg.batch):
            kb = keys[lo : lo + cfg.batch]
            ptrs, sizes, offsets = self.tables(len(kb))
            objsz = [cfg.object_bytes] * len(kb)
            out.append(
                Batch(
                    lambda k=kb, o=objsz, p=ptrs, s=sizes, f=offsets: (
                        self.client.batch_put_ranges_from_ptr(k, o, p, s, f)
                    ),
                    len(kb),
                    len(kb) * cfg.object_bytes,
                    key_names=tuple(kb),
                )
            )
        return out


class GetRangesOp(RangedOp):
    name = "get_ranges"

    def prepare(self, generation: int) -> list[Batch]:
        cfg, out = self.cfg, []
        keys = cfg.keys(self.name, self.ranks, self.rank, generation)
        for lo in range(0, len(keys), cfg.batch):
            kb = keys[lo : lo + cfg.batch]
            ptrs, sizes, offsets = self.tables(len(kb))
            out.append(
                Batch(
                    lambda k=kb, p=ptrs, s=sizes, f=offsets: (
                        self.client.batch_get_ranges_into_ptr(k, p, s, f)
                    ),
                    len(kb),
                    len(kb) * cfg.object_bytes,
                    key_names=tuple(kb),
                )
            )
        return out


OP_TYPES: dict[str, type[Op]] = {
    cls.name: cls for cls in (PutOp, GetOp, ExistsOp, PutRangesOp, GetRangesOp)
}


# ---------------------------------------------------------------------------
#  Rank worker
# ---------------------------------------------------------------------------


@dataclass
class OpResult:
    op: str
    rank: int
    calls: list[float] = field(default_factory=list)
    keys: int = 0
    nbytes: int = 0
    wall: float = 0.0
    failures: int = 0
    mismatches: int = 0


def _payload(key: str, size: int) -> bytes:
    """The exact bytes `key` must hold, all `size` of them.

    A keyed pseudo-random stream rather than a repeated tag, because the
    cheap patterns miss the bugs worth catching: with a 32-byte tag repeated
    to fill the object, a range written or read at the wrong offset lands on
    identical bytes and verifies clean.  `random.Random(str)` seeds from a
    SHA-512 of the string, so this is stable across processes and runs and
    owes nothing to PYTHONHASHSEED, and `randbytes` is C-implemented so
    filling a 64 KiB object costs microseconds.
    """
    return random.Random(key).randbytes(size)


def _seed(cfg: Config, client, buf, op: str, ranks: int, rank: int) -> None:
    """Untimed write of this op's read set (generation 0), plus the visibility wait.

    A put is not readable the instant it returns, so a read phase started too
    early reports a total miss that is indistinguishable from a broken path.
    """
    keys = cfg.keys(op, ranks, rank, 0)
    for lo in range(0, len(keys), cfg.batch):
        kb = keys[lo : lo + cfg.batch]
        # Refill the slots for THIS chunk.  The buffer only holds `batch`
        # objects, so it is reused chunk after chunk; writing it once up front
        # would leave every key past the first chunk holding the first
        # chunk's bytes, and a full-payload check would then be verifying the
        # wrong thing rather than nothing.
        for i, key in enumerate(kb):
            buf.write(i * cfg.object_bytes, _payload(key, cfg.object_bytes))
        ptrs = [buf.ptr + i * cfg.object_bytes for i in range(len(kb))]
        res = client.batch_put_from_ptr(kb, ptrs, [cfg.object_bytes] * len(kb))
        if not all(res):
            raise RuntimeError(
                f"seed put failed for {sum(1 for r in res if not r)} keys"
            )
    client.flush()
    deadline = time.time() + float(os.environ.get("UMBP_VISIBLE_TIMEOUT_S", 120))
    while time.time() < deadline:
        if all(client.batch_exists(keys[: cfg.batch])):
            return
        time.sleep(0.05)
    raise RuntimeError("seeded keys never became visible")


def _run_op(
    cfg: Config, client, buf, ranks: int, rank: int, name: str, barrier
) -> OpResult:
    op = OP_TYPES[name](cfg, client, buf, ranks, rank)
    result = OpResult(op=name, rank=rank)

    # Generation 0 is the read set. A write primitive gets its own generations
    # so every pass is a genuine insert.
    if op.needs_seed:
        _seed(cfg, client, buf, name, ranks, rank)

    for p in range(cfg.passes):
        generation = 1 + p if op.fresh_keys_per_pass else 0
        batches = op.prepare(generation)
        barrier.wait()
        start = time.perf_counter()
        for batch in batches:
            t0 = time.perf_counter()
            res = batch.invoke()
            result.calls.append(time.perf_counter() - t0)
            result.keys += batch.keys
            result.nbytes += batch.nbytes
            if res is not None:
                if batch.expect is None:
                    result.failures += sum(1 for r in res if not r)
                else:
                    # A probe that names absent keys expects False for them;
                    # counting those as failures would make the sweep useless,
                    # and NOT checking them would miss a false positive --
                    # which for a prefix-cache probe is the worse bug.
                    result.failures += sum(
                        1 for r, e in zip(res, batch.expect) if bool(r) != e
                    )
        result.wall += time.perf_counter() - start
        barrier.wait()

    if cfg.verify and name in ("get", "get_ranges"):
        result.mismatches += _verify(cfg, op, buf)
    return result


def _verify(cfg: Config, op: "Op", buf) -> int:
    """Re-read every key through the op's OWN call and compare every byte.

    Untimed, and deliberately after the timed windows rather than inside
    them.  Three things make this stronger than sampling a tag:

    * it is every key, not the first two, so a fault that only affects later
      chunks is reachable;
    * it is the whole object, so truncation and partial writes show up;
    * it goes through the op's own call, so `get_ranges` is verified with its
      real offset and size tables -- which is the only way a wrong range
      offset is ever caught.

    The buffer is zeroed first: without that a call which quietly moved
    nothing leaves the previous chunk's correct bytes in place and verifies
    clean.
    """
    zeros = b"\x00" * (cfg.batch * cfg.object_bytes)
    mismatches = 0
    for batch in op.prepare(0):
        buf.write(0, zeros[: batch.keys * cfg.object_bytes])
        batch.invoke()
        for slot, key in enumerate(batch.key_names):
            got = buf.read(slot * cfg.object_bytes, cfg.object_bytes)
            if got != _payload(key, cfg.object_bytes):
                mismatches += 1
    return mismatches


def _worker(ranks: int, rank: int, cfg: Config, barrier, queue) -> None:
    buf = client = None
    try:
        client = make_client(cfg)
        buf = make_buffer(cfg, cfg.batch * cfg.object_bytes, rank)
        register(client, cfg, buf)
        for name in cfg.ops:
            queue.put(_run_op(cfg, client, buf, ranks, rank, name, barrier))
    except BaseException as err:  # noqa: BLE001 - the parent needs the reason
        queue.put(("ERROR", rank, f"{type(err).__name__}: {err}"))
        # Release every barrier the siblings are still waiting on, or they hang
        # until the parent's join timeout instead of reporting this failure.
        try:
            barrier.abort()
        except Exception:
            pass
    finally:
        if buf is not None:
            buf.close()


# ---------------------------------------------------------------------------
#  Coordinator
# ---------------------------------------------------------------------------


def run_rank_count(cfg: Config, ranks: int) -> dict[str, list[OpResult]]:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(ranks)
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(ranks, r, cfg, barrier, queue), daemon=True)
        for r in range(ranks)
    ]
    for p in procs:
        p.start()

    expected = ranks * len(cfg.ops)
    by_op: dict[str, list[OpResult]] = {name: [] for name in cfg.ops}
    errors: list[str] = []
    for _ in range(expected):
        try:
            item = queue.get(timeout=1800)
        except Exception:
            errors.append("timed out waiting for a rank result")
            break
        if isinstance(item, tuple):
            errors.append(f"rank {item[1]}: {item[2]}")
            break
        by_op[item.op].append(item)

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    if errors:
        raise RuntimeError("; ".join(errors))
    return by_op


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


@dataclass
class Row:
    op: str
    ranks: int
    calls: int
    p50_ms: float
    p99_ms: float
    keys_per_s: float
    mib_per_s: float
    failures: int
    mismatches: int


def summarise(op: str, ranks: int, results: list[OpResult]) -> Row:
    calls = [c for r in results for c in r.calls]
    # Wall is the straggler's, not the sum: the ranks run inside one barrier, so
    # the window is as long as the slowest of them.
    wall = max((r.wall for r in results), default=0.0)
    keys = sum(r.keys for r in results)
    nbytes = sum(r.nbytes for r in results)
    return Row(
        op=op,
        ranks=ranks,
        calls=len(calls),
        p50_ms=_pct(calls, 0.5) * 1e3,
        p99_ms=_pct(calls, 0.99) * 1e3,
        keys_per_s=keys / wall if wall else 0.0,
        mib_per_s=(nbytes / 2**20) / wall if wall else 0.0,
        failures=sum(r.failures for r in results),
        mismatches=sum(r.mismatches for r in results),
    )


def report(cfg: Config, rows: list[Row]) -> None:
    print(
        f"\n{'op':<12}{'ranks':>6}{'calls':>7}{'p50_ms':>10}{'p99_ms':>10}"
        f"{'keys/s':>12}{'MiB/s':>11}{'scale':>7}{'fail':>6}{'bad':>5}"
    )
    base: dict[str, float] = {}
    for row in rows:
        base.setdefault(row.op, row.keys_per_s)
        scale = row.keys_per_s / base[row.op] if base[row.op] else 0.0
        print(
            f"{row.op:<12}{row.ranks:>6}{row.calls:>7}{row.p50_ms:>10.3f}"
            f"{row.p99_ms:>10.3f}{row.keys_per_s:>12,.0f}{row.mib_per_s:>11,.1f}"
            f"{scale:>6.2f}x{row.failures:>6}{row.mismatches:>5}"
        )
    print(
        f"\ngeometry: object={cfg.object_bytes}B ({cfg.ranges} x {cfg.range_bytes}B) "
        f"keys/rank={cfg.keys_per_rank} batch={cfg.batch} passes={cfg.passes} "
        f"loc={cfg.loc} register={cfg.register} "
        f"exists_absent_frac={cfg.exists_absent_frac:g}",
        flush=True,
    )


def write_json(cfg: Config, rows: list[Row], path: str) -> None:
    """Dump the run so a gate can read numbers instead of parsing the table.

    The geometry travels with the rows on purpose: a scaling floor is only
    meaningful against the shape that produced it, and a result file that
    does not say what was measured cannot be compared to anything later.
    """
    import json

    payload = {
        "backend": cfg.backend,
        "address": cfg.address,
        "geometry": {
            "object_bytes": cfg.object_bytes,
            "ranges": cfg.ranges,
            "range_bytes": cfg.range_bytes,
            "keys_per_rank": cfg.keys_per_rank,
            "batch": cfg.batch,
            "passes": cfg.passes,
            "loc": cfg.loc,
            "register": cfg.register,
            "exists_absent_frac": cfg.exists_absent_frac,
        },
        "rows": [asdict(row) for row in rows],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("backend", choices=["umbp-server", "umbp-local"])
    p.add_argument(
        "address",
        nargs="?",
        default="",
        help="standalone server address, e.g. unix:///run/umbp/standalone/a.sock "
        "(umbp-server only)",
    )
    p.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    p.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[1, 8],
        help="rank counts to sweep (default: 1 8). Each is a separate set of "
        "processes; a primitive is run at every one of them.",
    )
    p.add_argument("--keys-per-rank", type=int, default=1024)
    p.add_argument("--batch", type=int, default=256, help="keys per call")
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--ranges", type=int, default=8, help="ranges per object")
    p.add_argument("--range-bytes", type=int, default=131072)
    p.add_argument("--loc", choices=["host", "gpu"], default="gpu")
    p.add_argument("--no-register", dest="register", action="store_false")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--key-prefix", default=None)
    p.add_argument(
        "--exists-absent-frac",
        type=float,
        default=0.0,
        help="fraction of each exists call naming keys that were never written "
        "(0.0-1.0). Absent keys cannot be answered locally, so they are the "
        "half of the probe that consults the master.",
    )
    p.add_argument(
        "--dram-capacity-gib",
        type=int,
        default=64,
        help="umbp-local only; the server owns its own pool",
    )
    p.add_argument(
        "--json",
        dest="json_path",
        default="",
        help="also write the rows to this path as JSON, for a gate to read",
    )
    a = p.parse_args(argv)
    if a.backend == "umbp-server" and not a.address:
        p.error("umbp-server needs a server address")
    if any(r < 1 for r in a.ranks):
        p.error("--ranks must all be >= 1")
    if not 0.0 <= a.exists_absent_frac <= 1.0:
        p.error("--exists-absent-frac must be between 0.0 and 1.0")
    return Config(
        backend=a.backend,
        address=a.address,
        ops=tuple(a.ops),
        rank_counts=tuple(a.ranks),
        keys_per_rank=a.keys_per_rank,
        batch=min(a.batch, a.keys_per_rank),
        passes=a.passes,
        ranges=a.ranges,
        range_bytes=a.range_bytes,
        loc=a.loc,
        register=a.register,
        verify=a.verify,
        # Every rank count is a fresh set of processes writing the same keys, so
        # the prefix has to differ per invocation or a later sweep step reads
        # what an earlier one wrote and calls it a hit.
        exists_absent_frac=a.exists_absent_frac,
        key_prefix=a.key_prefix or f"p{os.getpid() % 100000:05d}",
        dram_capacity=a.dram_capacity_gib << 30,
        json_path=a.json_path,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    print(
        f"umbp_prim_bench backend={cfg.backend} address={cfg.address or '-'} "
        f"ops={','.join(cfg.ops)} ranks={','.join(map(str, cfg.rank_counts))} "
        f"key_prefix={cfg.key_prefix}",
        flush=True,
    )
    rows: list[Row] = []
    for ranks in cfg.rank_counts:
        by_op = run_rank_count(cfg, ranks)
        for name in cfg.ops:
            rows.append(summarise(name, ranks, by_op[name]))
            row = rows[-1]
            print(
                f"  ranks={ranks:<3} {row.op:<11} p50={row.p50_ms:8.3f} ms  "
                f"keys/s={row.keys_per_s:>11,.0f}  fail={row.failures}",
                flush=True,
            )
    # Report grouped by primitive so the rank sweep of one op reads down a
    # column rather than being interleaved with the others.
    rows.sort(key=lambda r: (cfg.ops.index(r.op), r.ranks))
    report(cfg, rows)
    if cfg.json_path:
        write_json(cfg, rows, cfg.json_path)
    bad = sum(r.failures + r.mismatches for r in rows)
    if bad:
        print(f"FAILED: {bad} failed or mismatched operations", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
