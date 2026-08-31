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
import atexit
import ctypes
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod

_DEFAULT_NR_OBJECTS = 2048
_DEFAULT_VALUE_SIZE = 2097152
_DEFAULT_SYNC_PORT = 18515


# --- optional in-process server startup (--start-server) ---


def _port_in_use(port) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def _wait_for_port(port, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_in_use(port):
            return True
        time.sleep(0.2)
    return False


_PR_SET_PDEATHSIG = 1


def _die_with_parent():
    # Linux only: ask the kernel to SIGKILL this child when the bench process
    # dies -- including when the bench is itself killed with SIGKILL, which
    # can't be caught so atexit never runs. Also start a new session so the
    # child becomes a process-group leader; this lets us SIGKILL the whole
    # group (including grandchildren that PDEATHSIG doesn't reach) on cleanup.
    os.setsid()
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)


def _kill_group(proc):
    # Kill the spawned process and any descendants it left behind (e.g.
    # mooncake_master's metrics/admin server). PDEATHSIG only covers direct
    # children and doesn't propagate to grandchildren, so reap the whole group.
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        proc.kill()


def _spawn(argv, log_path, env=None) -> subprocess.Popen:
    log = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=_die_with_parent,
        )
    finally:
        log.close()
    atexit.register(lambda: _kill_group(proc))
    return proc


# Backends that ranged_perf cannot drive.  Mooncake has no ranged/sub-object
# get in its client API at all, so offering the combination would only produce
# a runtime failure one flag later.
_RANGED_ONLY_EXCLUDED = {"mooncake"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umbp_bench.py")
    cmd_sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def _add_backend_sub(
        cmd_parser, *, batch_perf: bool = False, ranged_perf: bool = False
    ):
        be_sub = cmd_parser.add_subparsers(
            dest="backend", required=True, metavar="BACKEND"
        )
        for be_name in ("umbp", "umbp-local", "umbp-server", "mooncake"):
            if be_name in _RANGED_ONLY_EXCLUDED and ranged_perf:
                continue
            be_p = be_sub.add_parser(be_name)
            if be_name == "umbp":
                be_p.add_argument("master_address", help="e.g. ip:port")
            elif be_name == "umbp-server":
                be_p.add_argument(
                    "server_address",
                    help="standalone UMBP server, e.g. unix:///tmp/umbp.sock or ip:port",
                )
            elif be_name == "umbp-local":
                pass  # in-process: no address to connect to
            else:
                be_p.add_argument(
                    "metadata_server", help="e.g. http://ip:8080/metadata"
                )
                be_p.add_argument("master_server", help="e.g. ip:50051")
            be_p.add_argument("--nr-objects", type=int, default=_DEFAULT_NR_OBJECTS)
            be_p.add_argument("--value-size", type=int, default=_DEFAULT_VALUE_SIZE)
            be_p.add_argument(
                "--role",
                choices=["both", "writer", "reader"],
                default="both",
                help="Process role for split-node runs. 'both' (default) writes "
                "then reads in a single process (original behavior). 'writer' "
                "writes the dataset and stays alive holding it in DRAM until "
                "the reader is done. 'reader' connects to the writer's "
                "already-running master and reads. Run writer and reader on "
                "different nodes, passing the reader the writer's master "
                "address, and the same --key-prefix and --sync-port to both.",
            )
            be_p.add_argument(
                "--key-prefix",
                default=None,
                help="Shared key namespace (first 4 chars used). REQUIRED to match "
                "between writer and reader in split runs so they compute the "
                "same keys; defaults to a per-pid value when unset (fine for "
                "role=both, broken for split).",
            )
            be_p.add_argument(
                "--sync-port",
                type=int,
                default=_DEFAULT_SYNC_PORT,
                help="TCP port on the writer node used as a write-done/read-done "
                "rendezvous between a split writer and reader (default: "
                f"{_DEFAULT_SYNC_PORT}). Ignored for role=both.",
            )
            be_p.add_argument(
                "--numa-node",
                type=int,
                default=-1,
                help="Pin this process to a NUMA node (default: -1 = no binding): "
                "binds CPU affinity to the node's cpulist, allocates the bench "
                "DMA buffers on it, and (UMBP) places the DRAM tier there. The "
                "IO engine then auto-selects NUMA-local NICs. In a same-node "
                "split, give the writer and reader different --numa-node values "
                "to exercise a cross-NUMA transfer. Overridable per-allocation "
                "via UMBP_DRAM_NUMA_NODE.",
            )
            be_p.add_argument(
                "--start-server",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Launch the backend metadata/master server(s) for this run "
                "and tear them down on exit (default: on; use --no-start-server "
                "to reuse an already-running server).",
            )
            if be_name.startswith("umbp"):
                be_p.add_argument(
                    "--tier",
                    choices=["dram", "hbm", "ssd"],
                    default="dram",
                    help="Storage medium to benchmark (default: dram). Sets "
                    "UMBPDistributedConfig.medium, which is the single selector for "
                    "the ONE backend PoolClient::Init registers -- picking hbm or ssd "
                    "means the node has no DRAM pool at all, so the numbers are the "
                    "medium's own and not a mirror across two pools. Configure SSD via "
                    "UMBP_SSD_STORAGE_DIR / UMBP_SSD_CAPACITY_BYTES / UMBP_SSD_BACKEND.",
                )
                be_p.add_argument(
                    "--dst-loc",
                    choices=["host", "gpu"],
                    default="host",
                    help="Where the GET/read destination buffer lives (default: host). "
                    "'gpu' allocates it via hipMalloc and registers it as GPU-resident, "
                    "so a read from an HBM-tier pool (--tier hbm) exercises "
                    "HbmCopyEngine's device-to-device hipMemcpy path instead of "
                    "device-to-host. Combine with --tier hbm for the two HBM bench "
                    "cases: --dst-loc host = D2H, --dst-loc gpu = D2D. The write/seed "
                    "path is always host-resident (an H2D put into the pool) "
                    "regardless of this flag -- only the measured read side varies.",
                )
                be_p.add_argument(
                    "--hbm-device",
                    type=int,
                    default=0,
                    help="GPU ordinal for the HBM tier pool (--tier hbm) and, with "
                    "--dst-loc gpu, the client's own read-destination buffer "
                    "(default: 0). Overridable via UMBP_HBM_DEVICE.",
                )
            # --- SSD tier knobs, shared by every umbp deployment -------------
            #
            # UMBPBackend builds UMBPConfig by hand and never calls
            # FromEnvironment(), so these are applied by exporting the same
            # UMBP_SSD_* names the config block already reads (see
            # _apply_tier_env).  That keeps one code path -- and all of its
            # sizing warnings -- rather than a second, divergent one.
            if be_name.startswith("umbp"):
                be_p.add_argument(
                    "--ssd-dir",
                    default=None,
                    help="SSD tier directory. COMMA-SEPARATE for a multi-drive "
                    "sharded tier (one entry per physical drive) -- this is the "
                    "pure-ssd-mode production topology. Sets "
                    "UMBP_SSD_STORAGE_DIR.",
                )
                be_p.add_argument("--ssd-capacity-bytes", type=int, default=None)
                be_p.add_argument("--segment-bytes", type=int, default=None)
                be_p.add_argument(
                    "--dram-bytes",
                    type=int,
                    default=None,
                    help="DRAM tier capacity. For an SSD run keep it "
                    "far below the dataset so objects really demote.",
                )
                be_p.add_argument(
                    "--direct-io",
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help="O_DIRECT on the SSD tier (default: on). Off measures "
                    "the page cache, not the device.",
                )
                be_p.add_argument(
                    "--verify-crc",
                    choices=["on", "off", "both"],
                    default=None,
                    help="Whole-object checksum verification. Ranged reads never "
                    "verify, so 'both' (ranged_perf only) separates the byte "
                    "saving from the checksum saving.",
                )
                be_p.add_argument(
                    "--io-backend",
                    choices=["io_uring", "posix"],
                    default=None,
                    help="SSD I/O driver (default: io_uring). posix reports "
                    "batch_read=false, so every read is a separate blocking "
                    "pread -- a contrast, not the headline number.",
                )
                be_p.add_argument("--queue-depth", type=int, default=None)
                be_p.add_argument("--tier-io-threads", type=int, default=None)
                be_p.add_argument(
                    "--ranged-scratch-bytes",
                    type=int,
                    default=None,
                    help="Registered host arena for ranged multi-buffer I/O. It "
                    "is the ENTIRE opt-in: at 0 (the default) the client reports "
                    "supports_ranged_io()=False on every medium. Only the remote "
                    "path uses it. ranged_perf auto-sizes it when unset.",
                )
                be_p.add_argument(
                    "--buffer-backing",
                    choices=["auto", "shm", "hugetlb", "anon"],
                    default="auto",
                    help="Backing for the bench's own DMA buffers (default: auto). "
                    "auto = shm for umbp-server (the standalone server must map "
                    "the caller's memory, and an ordinary page returns a per-key "
                    "false that looks like the store refusing the op), hugetlb "
                    "otherwise.",
                )
            if ranged_perf:
                be_p.add_argument(
                    "--layers",
                    type=int,
                    default=32,
                    help="Layers per stored object (default: 32). object_size = "
                    "layers * layer_bytes and overrides --value-size.",
                )
                be_p.add_argument(
                    "--layer-bytes",
                    type=int,
                    default=36864,
                    help="Bytes of one layer of one page (default: 36864 = 9x4KiB, "
                    "MLA page_size=64 at 576B/token).",
                )
                be_p.add_argument(
                    "--groups",
                    type=int,
                    nargs="+",
                    default=[1, 2, 4, 8, 16, 32],
                    help="Layer-group widths to sweep. The full layer count is "
                    "always added as the whole-object-via-ranges case.",
                )
                be_p.add_argument(
                    "--first-layer",
                    type=int,
                    default=0,
                    help="First layer of the group read (default: 0).",
                )
                be_p.add_argument(
                    "--batch",
                    type=int,
                    default=32,
                    help="Objects per ranged/whole call (default: 32).",
                )
            if batch_perf or ranged_perf:
                be_p.add_argument("--passes", type=int, default=10 if batch_perf else 7)
            if batch_perf:
                be_p.add_argument("--batch-sizes", type=int, nargs="+", metavar="N")
                be_p.add_argument(
                    "--dest-layout",
                    choices=["contiguous", "shuffled"],
                    default="shuffled",
                    help="Destination buffer layout for reads (default: shuffled). "
                    "'shuffled' maps keys to a random permutation of slots in the "
                    "same buffer so consecutive keys are non-adjacent in memory "
                    "(non-consecutive transfer). 'contiguous' packs objects "
                    "back-to-back (coalesceable), but a full-dataset batch then "
                    "coalesces into a single multi-GB SGE that the ionic provider "
                    "rejects (ibv_post_send EINVAL), so large batches all-miss.",
                )
                be_p.add_argument(
                    "--shuffle-seed",
                    type=int,
                    default=0,
                    help="Seed for --dest-layout shuffled (default: 0).",
                )

    _add_backend_sub(cmd_sub.add_parser("correctness"))
    _add_backend_sub(cmd_sub.add_parser("batch_perf"), batch_perf=True)
    _add_backend_sub(cmd_sub.add_parser("exists_perf"), batch_perf=True)
    _add_backend_sub(cmd_sub.add_parser("ranged_perf"), ranged_perf=True)
    return parser


args = _build_parser().parse_args()
command = args.command

# One UMBP client class, three ways to reach it.  `backend_name` stays the
# family so every existing `backend_name == "umbp"` test keeps its meaning, and
# `deployment` carries which one:
#   distributed        -- DistributedClient in this process, talks to a master
#   local              -- StandaloneClient in this process, no network at all
#   standalone-server  -- forwards over gRPC/shm to a standalone UMBP server,
#                         which itself may be local- or distributed-backed
_DEPLOYMENT_OF = {
    "umbp": "distributed",
    "umbp-local": "local",
    "umbp-server": "standalone-server",
    "mooncake": "distributed",
}
deployment = _DEPLOYMENT_OF[args.backend]
backend_name = "umbp" if args.backend.startswith("umbp") else args.backend

command = args.command
nr_objects = args.nr_objects
role = args.role
numa_node = args.numa_node

# ranged_perf sizes an object as layers x layer_bytes; batch_perf/correctness
# size it as --value-size.  One `value_size` downstream either way, so the
# staging/pool planners below need no ranged special case.
layers = getattr(args, "layers", 0)
layer_bytes = getattr(args, "layer_bytes", 0)
if command == "ranged_perf":
    value_size = layers * layer_bytes
    if value_size <= 0:
        print(
            "ERROR: --layers and --layer-bytes must both be positive", file=sys.stderr
        )
        sys.exit(2)
else:
    value_size = args.value_size

if role != "both" and not args.key_prefix:
    print(
        "ERROR: split runs (--role writer/reader) require a shared --key-prefix on "
        "both writer and reader so they compute the same keys",
        file=sys.stderr,
    )
    sys.exit(2)

tier = getattr(args, "tier", "dram")
dst_loc = getattr(args, "dst_loc", "host")
hbm_device = getattr(args, "hbm_device", 0)


def _apply_tier_env(a) -> None:
    """Lower the SSD/DRAM CLI flags onto the UMBP_* names the config block reads.

    UMBPBackend builds UMBPConfig by hand rather than through
    FromEnvironment(), and every tier knob it honours is already spelled as an
    os.environ lookup with its own sizing warnings attached.  Routing the new
    flags through the same names keeps ONE configuration path: a flag and its
    env var cannot drift, and nothing bypasses the "staging arena >= dataset"
    or "slot < value_size" warnings that make an SSD number trustworthy.

    An explicitly exported env var still wins, so existing runbooks that set
    UMBP_SSD_* directly behave exactly as before.
    """
    mapping = {
        "ssd_dir": "UMBP_SSD_STORAGE_DIR",
        "ssd_capacity_bytes": "UMBP_SSD_CAPACITY_BYTES",
        "tier_io_threads": "UMBP_SSD_TIER_IO_THREADS",
        "dram_bytes": "UMBP_DRAM_CAPACITY_BYTES",
    }
    for attr, env in mapping.items():
        val = getattr(a, attr, None)
        if val is not None and env not in os.environ:
            os.environ[env] = str(val)
    direct = getattr(a, "direct_io", None)
    if direct is not None and "UMBP_SSD_DIRECT_IO" not in os.environ:
        os.environ["UMBP_SSD_DIRECT_IO"] = "1" if direct else "0"
    crc = getattr(a, "verify_crc", None)
    # "both" is a ranged_perf sweep, not a single setting; the sweep sets the
    # var per iteration, so leave it alone here.
    if crc in ("on", "off") and "UMBP_SSD_VERIFY_CRC" not in os.environ:
        os.environ["UMBP_SSD_VERIFY_CRC"] = "1" if crc == "on" else "0"


_apply_tier_env(args)

if getattr(args, "verify_crc", None) == "both" and command != "ranged_perf":
    print(
        "ERROR: --verify-crc both is a ranged_perf sweep; pass on/off elsewhere",
        file=sys.stderr,
    )
    sys.exit(2)
if command == "ranged_perf" and backend_name != "umbp":
    print("ERROR: ranged_perf requires a umbp backend", file=sys.stderr)
    sys.exit(2)
if dst_loc == "gpu" and backend_name != "umbp":
    print("ERROR: --dst-loc gpu is only supported for backend=umbp", file=sys.stderr)
    sys.exit(2)


# --- SSD read-staging arena sizing ------------------------------------------
# A remote SSD read is not served in place: the owner stages the value into a
# registered host arena of `slots` pages and the reader RDMAs it out. A key that
# finds no free slot does NOT retry and does NOT block -- it is reported as a
# plain cache MISS (ssd_backend.cpp bumps slot_full_rejects_ and gives up;
# UMBP_SSD_GET_MAX_ATTEMPTS is described in doc/pure-ssd-mode.md but is not
# implemented on this branch, so there is no retry lever).
#
# That makes a fixed default arena actively dangerous for a sweep: with 512
# slots a batch of 2048 keys returns exactly 512 hits and 1536 misses, every
# pass, and the run still prints a perfectly healthy-looking MiB/s -- computed
# only over the keys that got a slot. The drives are never the bottleneck, so
# drive-count scaling comes out flat and the whole comparison is silently
# meaningless. Size the arena from the sweep instead of from a constant.
def _staging_plan(nr_objects_: int, value_size_: int, batch_sizes_):
    """Slots/bytes the SSD tier needs so the largest batch can be fully staged.

    Returns (slots, total_bytes, max_batch). Slot size is total_bytes // slots
    and must be >= the largest single value, so bytes are derived from slots.
    """
    max_batch = max(batch_sizes_) if batch_sizes_ else nr_objects_
    max_batch = max(1, min(max_batch, nr_objects_))
    # Occupancy is NOT just the batch width. A staged page is freed when the
    # reader's best-effort release lands, and otherwise only when the read lease
    # expires -- so in steady state the arena also holds roughly
    # (completed reads/sec x lease). Measured on n06-21 with a 2304-slot arena:
    #
    #   1 drive,  batch 128:  3042 req/s -> 1521 + 128  = 1649 slots -> 0 misses
    #   2 drives, batch 128:  4699 req/s -> 2350 + 128  = 2478 slots -> 23% misses
    #   1 drive,  batch 2048: 2939 req/s -> 1470 + 2048 = 3518 slots -> 44% misses
    #
    # The throughput term is what makes this bite harder as drives are added:
    # faster media completes more reads per second, so it needs a BIGGER arena,
    # which is the opposite of the intuition that fewer drives are the tight case.
    # Two levers: keep the lease short (see _SSD_LEASE_MS -- it only has to cover
    # one sub-ms RDMA round trip, so the 500ms default is ~1000x more than needed)
    # and budget two batches of burst plus a fixed throughput margin.
    # The hold time is NOT just the lease: measured on 4x KIOXIA CD8 at ~15k
    # reads/s, clearing misses needed 4096 slots for a 512-wide batch, implying
    # ~200ms of hold per page against a 50ms lease -- the RDMA round trip and the
    # asynchronous release dominate. Calibrated empirically at the fastest media
    # available (30 GB/s), since under-sizing is silent and over-sizing only
    # costs host memory:
    #
    #   4 drives, batch 512, 2048 slots -> 20% misses (13.4 GB/s reported)
    #   4 drives, batch 512, 4096 slots ->  0% misses (30.0 GB/s)
    #   4 drives, batch 512, 8192 slots ->  0% misses (29.0 GB/s -- no gain,
    #                                       so 4096 is genuinely sufficient and
    #                                       the number is not cache-inflated)
    slots = max(4096, 4 * max_batch + 2048)
    # Pad the per-slot page to the 4 KiB the on-disk records are aligned to.
    slot_bytes = ((value_size_ + 4095) // 4096) * 4096
    return slots, slots * slot_bytes, max_batch


_auto_slots, _auto_staging_bytes, _auto_max_batch = _staging_plan(
    nr_objects, value_size, getattr(args, "batch_sizes", None)
)

# The SSD backend holds a staged page for this lease
# (pool_client.cpp SsdReadLeaseTtl, UMBP_SSD_READ_LEASE_MS, 3000ms default).
# It must outlive one SSD read plus RDMA, but it also bounds how fast slots
# recycle -- at the production default a fast multi-drive tier can keep
# thousands of pages pinned purely waiting for it. Default it down for the bench
# so arena sizing stays a function of the sweep instead of the media speed. Must
# be in the environment before the client is built: the C++ side caches it in a
# function-local static on first read.
_SSD_LEASE_MS = "50"
if tier == "ssd":
    os.environ.setdefault("UMBP_SSD_READ_LEASE_MS", _SSD_LEASE_MS)


# --- pool capacity sizing ----------------------------------------------------
# The master runs an eviction manager over every (node, tier): once a tier's
# used/total crosses EvictionConfig::high_watermark it evicts that tier's oldest
# keys down to low_watermark (eviction_manager.cpp; defaults 0.9 / 0.7 in
# distributed/config.h, checked every 5s). Those two watermarks are deliberately
# NOT env-overridable -- "watermarks and batch size have dedicated tuning paths
# and are intentionally excluded" -- so a bench cannot turn the reaper off. The
# only lever is to keep the dataset under the high watermark.
#
# Getting this wrong is silent and looks exactly like a correctness bug. Measured
# on n06-21 with a 32 GiB dataset in a 34 GiB pool: the write side reports a clean
# `WRITE_DONE ok=16384 fail=0`, then every later pass misses 25.6% of its keys with
# no eviction line anywhere in the client log. 32768/34816 = 94% >= 0.9 trips the
# watermark, and 0.7 * 34816 MiB / 2 MiB = 12185 keys survive -- exactly the 12185
# hits observed, reproducibly. The same dataset in a 48 GiB pool (67%) never
# evicts, which is why an oversized pool "fixes" it and why the bug only shows up
# once someone tightens the pool.
#
# Note this is a race, not a hard cap: the sweep is only wrong if the 5s eviction
# check lands before the reads. A short single-pass run over a 100%-full pool can
# pass, which makes small-scale probing actively misleading -- size for the
# watermark, do not trust a lucky short run.
_EVICT_HIGH_WM = 0.9  # EvictionConfig::high_watermark
_EVICT_LOW_WM = 0.7  # EvictionConfig::low_watermark (what it evicts DOWN to)
# Target occupancy for an auto-sized pool. Below the high watermark with room to
# spare, since over-sizing only costs host memory while under-sizing silently
# destroys the measurement.
_POOL_TARGET_OCCUPANCY = 0.75


def _pool_plan(nr_objects_: int, value_size_: int, per_entry_bytes: int = 0) -> int:
    """Tier capacity that keeps the whole dataset below the eviction watermark.

    `per_entry_bytes` overrides the on-medium cost of one value (the SSD tier
    charges capacity in 4 KiB-padded record bytes, not raw value bytes).
    """
    entry = per_entry_bytes or value_size_
    return int(nr_objects_ * entry / _POOL_TARGET_OCCUPANCY)


# The reader only needs capacity of its own when it re-caches what it fetched;
# with UMBP_CACHE_REMOTE_FETCHES=0 its pool never holds anything, and sizing it
# like the writer's just wastes memory -- which is not free, because a DRAM pool
# is hugepage-backed and NUMA-bound, so an oversized reader pool starves its own
# read buffer of hugepages on the reader's node and demotes it to 4 KiB pages.
_caches_remotely = os.environ.get("UMBP_CACHE_REMOTE_FETCHES", "1") not in (
    "0",
    "false",
    "False",
    "",
)
_holds_dataset = role != "reader" or _caches_remotely
_auto_pool_bytes = _pool_plan(nr_objects, value_size)


def _pool_capacity(env_var: str, auto_bytes: int, floor_bytes: int, label: str) -> int:
    """Resolve a tier capacity, warning when an explicit value invites eviction."""
    want = auto_bytes if _holds_dataset else floor_bytes
    configured = int(os.environ.get(env_var, want))
    dataset = nr_objects * value_size
    if _holds_dataset and configured and dataset / configured >= _EVICT_HIGH_WM:
        survivors = int(configured * _EVICT_LOW_WM / value_size)
        print(
            f"WARNING: {label} capacity ({configured / 2**30:.2f}GiB) leaves the "
            f"{dataset / 2**30:.2f}GiB dataset at "
            f"{100 * dataset / configured:.1f}% >= the {_EVICT_HIGH_WM:.0%} eviction "
            f"high watermark. The master will evict down to {_EVICT_LOW_WM:.0%}, so "
            f"about {survivors} of {nr_objects} keys survive and the rest come back "
            f"as MISSES after a clean WRITE_DONE. Unset {env_var} to auto-size "
            f"(would use {auto_bytes / 2**30:.2f}GiB).",
            flush=True,
        )
    return configured


# Resolve every tier capacity ONCE, here, rather than per-client: role=both builds
# two backends and would otherwise print each warning twice.
_DRAM_FLOOR = 8 * 1024 * 1024 * 1024
_dram_capacity_bytes = (
    _pool_capacity("UMBP_DRAM_CAPACITY_BYTES", _auto_pool_bytes, _DRAM_FLOOR, "DRAM")
    if tier == "dram"
    # Floor for hbm/ssd runs: PoolClient::Init always registers DRAM, and
    # most-available put routing would otherwise send every put to DRAM and
    # report it as the tier under test.
    else int(os.environ.get("UMBP_DRAM_CAPACITY_BYTES", 256 * 1024 * 1024))
)
_ssd_entry_bytes = ((value_size + 4095) // 4096) * 4096

_buffer_backing = getattr(args, "buffer_backing", "auto")
_io_backend = getattr(args, "io_backend", None)
_queue_depth = getattr(args, "queue_depth", None)
_segment_bytes = getattr(args, "segment_bytes", None)

# Ranged scratch: only the remote half of a ranged operation uses it, and it
# must hold a whole sub-batch of objects at once (an object larger than the
# arena fails that key).  Auto-size to the batch with headroom; zero elsewhere,
# which is what keeps ranged I/O off for deployments that never ask for it.
_ranged_scratch_bytes = getattr(args, "ranged_scratch_bytes", None)
if _ranged_scratch_bytes is None:
    _ranged_scratch_bytes = (
        max(4, getattr(args, "batch", 32)) * value_size * 2
        if command == "ranged_perf"
        else 0
    )


# --- hugepage preflight ------------------------------------------------------
# The DRAM pool, the SSD staging arena and the bench's own read buffer are all
# hugepage-backed AND bound to this process's --numa-node. Hugepages are a
# per-NUMA-node reservation, so the only number that matters is that node's
# free_hugepages -- the /proc/meminfo total can look plentiful while the pinned
# node is empty.
#
# Running short does not fail the allocation: mmap(MAP_HUGETLB) succeeds against
# the reservation, then MADV_POPULATE_WRITE returns EFAULT, the allocator falls
# back to touching pages by hand, and the process dies on the first page it
# cannot get -- a bare "Bus error (core dumped)" with no indication that page
# size or NUMA had anything to do with it. Check up front and say so instead.
def _hugepage_preflight():
    if os.environ.get("UMBP_DRAM_USE_HUGEPAGES", "1") in ("0", ""):
        return
    node = int(os.environ.get("UMBP_DRAM_NUMA_NODE", numa_node))
    if node < 0:
        return  # unpinned: the kernel may draw from any node, so no check is sound
    hp_size = int(os.environ.get("UMBP_DRAM_HUGEPAGE_SIZE", 2 * 1024 * 1024))
    try:
        with open(
            f"/sys/devices/system/node/node{node}/hugepages/"
            f"hugepages-{hp_size // 1024}kB/free_hugepages"
        ) as f:
            free_bytes = int(f.read().strip()) * hp_size
    except OSError:
        return  # no hugetlb sysfs for this size; allocation will fall back loudly

    need = {"DRAM pool": _dram_capacity_bytes}
    if tier == "ssd":
        need["SSD staging arena"] = int(
            os.environ.get("UMBP_SSD_STAGING_BYTES", _auto_staging_bytes)
        )
    if dst_loc == "host":
        # batch_perf reads the whole dataset into one buffer; correctness reuses
        # a single value-sized one. The writer allocates only a scratch value.
        if command == "batch_perf" and role != "writer":
            need["read buffer"] = nr_objects * value_size
        elif command == "exists_perf":
            # Presence queries move no value bytes; a value-sized scratch for
            # the writer is all this command ever touches.
            need["value scratch"] = value_size
        else:
            need["value scratch"] = value_size
    total = sum(need.values())
    if total <= free_bytes:
        return

    detail = ", ".join(f"{k} {v / 2**30:.2f}GiB" for k, v in need.items())
    pages = -(-total // hp_size)
    print(
        f"ERROR: this run needs about {total / 2**30:.2f}GiB of {hp_size // 1024}kB "
        f"hugepages on NUMA node {node} ({detail}), but that node has only "
        f"{free_bytes / 2**30:.2f}GiB free. Hugepages are reserved PER NUMA NODE, so "
        f"a healthy-looking /proc/meminfo total does not help a pinned process. "
        f"Continuing would not fail cleanly -- it would die with a bare "
        f"'Bus error (core dumped)' while faulting the pages in.\n"
        f"  Fix (as root):  echo {pages} > /sys/devices/system/node/node{node}"
        f"/hugepages/hugepages-{hp_size // 1024}kB/nr_hugepages\n"
        f"  If the node cannot satisfy that, its free memory is probably page "
        f"cache: 'sync; echo 3 > /proc/sys/vm/drop_caches' first.\n"
        f"  Or run without hugepages:  UMBP_DRAM_USE_HUGEPAGES=0  (4 KiB pages; "
        f"lowers DRAM-tier numbers and makes large RDMA registrations more likely "
        f"to fail, so do not mix such runs into a comparison).",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


if backend_name == "umbp":
    _hugepage_preflight()

    from mori.io import MemoryLocationType
    from mori.umbp import (
        UMBPClient,
        UMBPConfig,
        UMBPDistributedConfig,
        UMBPIoBackend,
        UMBPMedium,
        UMBPStandaloneProcessConfig,
    )

    if deployment == "distributed":
        master_address = args.master_address
        _resolve_host = master_address.split(":")[0]
    elif deployment == "standalone-server":
        master_address = ""
        server_address = args.server_address
        # A unix socket has no host to rendezvous against; a split run over one
        # would be two processes on the same node anyway, so localhost is right.
        _resolve_host = (
            "127.0.0.1"
            if server_address.startswith("unix:")
            else server_address.split("//")[-1].split(":")[0]
        )
    else:  # local
        master_address = ""
        _resolve_host = "127.0.0.1"
else:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    metadata_server = args.metadata_server
    master_server = args.master_server
    _resolve_host = master_server.split(":")[0]

key_prefix = args.key_prefix if args.key_prefix else f"{os.getpid() % 10000:04d}"
key_size = 20
pattern = b"\xab"

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
    _s.connect((_resolve_host, 1))
    node_address = _s.getsockname()[0]


def make_key(object_id: int) -> str:
    suffix = f"{object_id:016d}"
    prefix_part = key_prefix[: key_size - len(suffix)].ljust(
        key_size - len(suffix), "_"
    )
    return f"{prefix_part}{suffix}"


def expected_payload(size: int) -> bytes:
    reps = (size // len(pattern)) + 1
    return (pattern * reps)[:size]


# --- backend abstraction ---


class Backend(ABC):
    """Uniform KV-store interface; subclass per backend."""

    @classmethod
    @abstractmethod
    def start_server(cls, args):
        """Launch this backend's metadata/master server(s) for the run; tear down on exit."""

    @abstractmethod
    def put(self, key: str, payload: bytes, ptr: int, size: int) -> bool:
        """Write value. ptr is a pre-registered DMA buffer (may be ignored by some backends)."""

    @abstractmethod
    def flush(self): ...

    @abstractmethod
    def register_memory(self, ptr: int, size: int, loc: str = "cpu", device: int = -1):
        """Register a memory region for RDMA (no-op for backends that don't need it).
        loc/device describe the CALLER's allocation ("cpu" or "gpu" + ordinal),
        not any storage medium -- needed so a GPU-resident buffer routes through
        HbmCopyEngine instead of being treated as host memory."""

    @abstractmethod
    def get_into_ptr(self, key: str, ptr: int, size: int) -> bool:
        """Read value into ptr. Returns True on hit."""

    @abstractmethod
    def batch_get_into_ptr(self, keys: list, ptrs: list, sizes: list) -> list:
        """Batch read into ptrs. Returns bytes_read per key (0 = miss)."""

    def batch_exists(self, keys: list) -> list:
        """Presence of each key. Not abstract: only the UMBP backends implement
        it, and a backend that cannot answer should say so rather than silently
        report every key missing."""
        raise NotImplementedError(f"{type(self).__name__} has no batch_exists")

    def batch_exists_consecutive(self, keys: list) -> int:
        """How many keys exist consecutively from index 0 (early-stop).  This is
        the shape a prefix-cache probe actually uses."""
        raise NotImplementedError(
            f"{type(self).__name__} has no batch_exists_consecutive"
        )

    @abstractmethod
    def wait_visible(self, keys: list, timeout: float = 30.0) -> int:
        """Block until the store reports `keys` present, returning how many are.

        A put is not necessarily readable the moment it returns. In distributed
        mode a peer publishes its ADDs to the master on the heartbeat, and the
        size-based auto-flush that would ship them sooner only trips at
        UMBP_AUTO_FLUSH_EVENT_THRESHOLD (default 128) completed puts -- and on
        the SSD medium not even then, since those events always wait for the
        interval. So a small dataset written and immediately read comes back as
        a total miss that looks exactly like a broken read path.

        Default is a no-op for backends whose writes are synchronously visible.
        """
        return len(keys)

    def supports_ranged_io(self) -> bool:
        """Whether this backend can serve sub-object reads.

        Advisory on the UMBP side: the ranged entry points forward
        unconditionally, so this is what a caller ASKS, not a guard the data
        path enforces (see doc/design-ssd-ranged-io.md 3.1).  ranged_perf
        asserts it before timing anything, so a run cannot silently measure a
        path the deployment claims not to have.
        """
        return False

    def batch_get_ranges_into_ptr(self, keys, ptrs, sizes, offsets) -> list:
        raise NotImplementedError(f"{type(self).__name__} has no ranged I/O")

    def make_reader(self) -> "Backend":
        """Return a backend instance for reading (may be self)."""


class UMBPBackend(Backend):
    @classmethod
    def start_server(cls, args):
        # In a split run the reader connects using the exact master port it was
        # given on the CLI, so the writer must bind that port instead of an
        # auto-picked one. role=both keeps auto-picking to avoid collisions on
        # repeated single-node runs.
        parts = args.master_address.split(":")
        desired_port = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        port = (
            desired_port if (args.role == "writer" and desired_port) else _free_port()
        )
        mori_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        fqdn = subprocess.run(
            ["hostname", "-f"], capture_output=True, text=True
        ).stdout.strip()
        binary = os.path.join(mori_root, f"build_{fqdn}", "src", "umbp", "umbp_master")
        if not os.path.exists(binary):
            print(
                f"umbp_master binary not found: {binary} (run ./build.sh first)",
                file=sys.stderr,
            )
            sys.exit(1)
        env = dict(os.environ)
        env.setdefault("MORI_UMBP_LOG_LEVEL", "DEBUG")
        # umbp_master links libmori_io / libmori_application straight out of the
        # build tree and nothing installs them, so in a container where mori was
        # only pip-installed (not system-installed) the spawn dies with
        # "libmori_io.so: cannot open shared object file". The failure surfaces
        # only as "umbp_master failed to start", so put the tree on the loader
        # path rather than leaving that to the caller's environment.
        _libdirs = [
            os.path.join(mori_root, f"build_{fqdn}", sub)
            for sub in ("src/io", "src/application", "src/metrics", "src/shmem")
        ]
        env["LD_LIBRARY_PATH"] = ":".join(
            [d for d in _libdirs if os.path.isdir(d)]
            + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
        proc = _spawn([binary, f"0.0.0.0:{port}", "0"], "/tmp/umbp_master.log", env)
        if not _wait_for_port(port):
            print(
                "umbp_master failed to start; see /tmp/umbp_master.log", file=sys.stderr
            )
            sys.exit(1)
        print(f"started umbp_master pid={proc.pid} on 0.0.0.0:{port}", flush=True)
        # Return the address the client should connect to (port may have been
        # auto-picked, so it can differ from what was passed on the CLI).
        return f"{args.master_address.split(':')[0]}:{port}"

    def __init__(
        self,
        node_id: str,
        port: int,
        reader_node_id: str = None,
        reader_port: int = None,
    ):
        cfg = UMBPConfig()
        # PoolClient::Init ALWAYS registers DRAM, and the routing plane has no
        # tier order -- ConfigurableRoutePutStrategy/kMostAvailable simply picks
        # whichever medium advertises the most free bytes (route_put_strategy.h).
        # So with the 8 GiB DRAM default a --tier hbm run (4 GiB) would route
        # every put to DRAM and report it as HBM. Shrink DRAM to a floor well
        # below the target pool so most-available can only pick the target.
        # The bench's whole working set must also fit in that headroom.
        # For a --tier dram run this pool holds the dataset, so it is sized from
        # the eviction watermark (see _pool_plan) rather than from a constant --
        # an 8 GiB default silently evicted most of any dataset bigger than
        # ~7 GiB. For the other tiers it stays a floor.
        cfg.dram.capacity_bytes = _dram_capacity_bytes
        cfg.dram.use_hugepages = os.environ.get("UMBP_DRAM_USE_HUGEPAGES", "1") not in (
            "0",
            "",
        )
        cfg.dram.hugepage_size = int(
            os.environ.get("UMBP_DRAM_HUGEPAGE_SIZE", 2 * 1024 * 1024)
        )
        cfg.dram.numa_node = int(os.environ.get("UMBP_DRAM_NUMA_NODE", numa_node))
        dist = UMBPDistributedConfig()
        dist.master_config.master_address = master_address
        dist.master_config.node_id = node_id
        dist.master_config.node_address = node_address
        dist.peer_service_port = port
        # Re-caching remotely-fetched blocks into the LOCAL tier defaults on,
        # which silently changes what a tier benchmark measures: the reader
        # gets its own pool of the tier under test, so after pass 1 most reads
        # are served from the reader's own memory (a local copy) instead of
        # RDMA out of the writer's tier. Set UMBP_CACHE_REMOTE_FETCHES=0 to
        # pin every pass to a genuine remote read. Default keeps the
        # production behavior.
        dist.cache_remote_fetches = os.environ.get(
            "UMBP_CACHE_REMOTE_FETCHES", "1"
        ) not in ("0", "false", "False", "")
        # Local-first (PoolClientConfig::local_first): resolve on this node's
        # own media before asking the master, and skip the master entirely when
        # the whole batch is local.  Exposed here because it is the knob a
        # single-node get sweep exists to measure -- with it off every batch
        # pays a BatchRouteGet round trip it cannot use.
        dist.local_first = os.environ.get("UMBP_LOCAL_FIRST", "1") not in (
            "0",
            "false",
            "False",
            "",
        )
        dist.io_engine.host = node_address
        # The ranged scratch arena is the whole opt-in for ranged I/O, on every
        # medium including SSD, and it defaults to zero.  ranged_perf sizes it
        # from the batch when the flag is unset; other commands leave it off so
        # a deployment that never issues ranged operations allocates and
        # registers nothing.
        dist.ranged_scratch_size = _ranged_scratch_bytes
        if tier == "hbm":
            dist.medium = UMBPMedium.HBM
            dist.hbm.device = int(os.environ.get("UMBP_HBM_DEVICE", hbm_device))
            # Same eviction watermark applies to HBM: the master's eviction
            # manager walks every (node, tier), not just DRAM. The old flat 4 GiB
            # default evicted all but ~1400 keys of a 32 GiB dataset.
            dist.hbm.capacity_bytes = _pool_capacity(
                "UMBP_HBM_CAPACITY_BYTES",
                _auto_pool_bytes,
                4 * 1024 * 1024 * 1024,
                "HBM",
            )
        if tier == "ssd":
            # medium is the distributed-side selector; cfg.ssd carries the actual
            # knobs (its own `enabled` describes the LOCAL-mode tier and is not
            # what turns the distributed medium on -- see UMBPDistributedConfig).
            dist.medium = UMBPMedium.SSD
            # SSD is not directly addressable: a remote read stages the value
            # into a registered host arena of ssd_staging_buffer_slots pages.
            # Too few slots does not slow the sweep down, it turns the keys that
            # miss out on a slot into cache MISSES -- see _staging_plan above for
            # why that is silent and why the default is computed, not constant.
            # The env vars stay as escape hatches, but an explicit value that is
            # too small for the sweep is now called out instead of obeyed quietly.
            dist.ssd_staging_buffer_slots = int(
                os.environ.get("UMBP_SSD_STAGING_SLOTS", _auto_slots)
            )
            # Slot size = size / slots, and must be >= the largest single value.
            dist.ssd_staging_buffer_size = int(
                os.environ.get("UMBP_SSD_STAGING_BYTES", _auto_staging_bytes)
            )
            _slot_bytes = (
                dist.ssd_staging_buffer_size // dist.ssd_staging_buffer_slots
                if dist.ssd_staging_buffer_slots
                else 0
            )
            print(
                f"[staging] slots={dist.ssd_staging_buffer_slots} "
                f"arena={dist.ssd_staging_buffer_size / 2**30:.2f}GiB "
                f"slot={_slot_bytes / 2**20:.2f}MiB "
                f"(auto: slots={_auto_slots} arena={_auto_staging_bytes / 2**30:.2f}GiB "
                f"for max_batch={_auto_max_batch}) "
                f"read_lease_ms={os.environ.get('UMBP_DRAM_READ_LEASE_MS')}",
                flush=True,
            )
            if dist.ssd_staging_buffer_slots < _auto_max_batch:
                print(
                    f"WARNING: staging slots ({dist.ssd_staging_buffer_slots}) < largest "
                    f"batch ({_auto_max_batch}). Keys past the slot count return as "
                    f"MISSES, not slow hits, and the reported MiB/s is computed only "
                    f"over the keys that got a slot -- the sweep will look fast and "
                    f"scale flat. Unset UMBP_SSD_STAGING_SLOTS to auto-size.",
                    flush=True,
                )
            if _slot_bytes < value_size:
                print(
                    f"WARNING: staging slot ({_slot_bytes}B) < value_size "
                    f"({value_size}B); values cannot be staged and every read will "
                    f"miss. Unset UMBP_SSD_STAGING_BYTES to auto-size.",
                    flush=True,
                )
            # The other half of the trap: an arena at least as big as the whole
            # dataset means a repeated pass finds every key still staged from the
            # previous one (SsdBackend::Resolve serves a live lease straight from
            # the arena and renews it), so passes 2..N are timed against host DRAM
            # and never touch a drive. That reads as a several-fold speedup and
            # flattens drive-count scaling just as effectively as the miss bug.
            _dataset_bytes = nr_objects * value_size
            if dist.ssd_staging_buffer_size >= _dataset_bytes:
                print(
                    f"WARNING: staging arena "
                    f"({dist.ssd_staging_buffer_size / 2**30:.2f}GiB) >= dataset "
                    f"({_dataset_bytes / 2**30:.2f}GiB), so repeated passes are served "
                    f"from the staging arena in host DRAM rather than from the drives. "
                    f"For a real SSD number raise --nr-objects (>= ~4x the largest "
                    f"batch) or lower --batch-sizes, and cross-check with iostat.",
                    flush=True,
                )
            cfg.ssd.enabled = True
            cfg.ssd.storage_dir = os.environ.get(
                "UMBP_SSD_STORAGE_DIR", f"/tmp/umbp_ssd_{node_id}"
            )
            # SSD is watermarked twice over -- the master's eviction manager and
            # PeerSsdManager's own high/low watermarks (both 0.9/0.7) -- and it
            # charges capacity in 4 KiB-padded record bytes, so size from the
            # padded entry. This is the total across every drive in
            # UMBP_SSD_STORAGE_DIR, not a per-drive budget.
            cfg.ssd.capacity_bytes = _pool_capacity(
                "UMBP_SSD_CAPACITY_BYTES",
                _pool_plan(nr_objects, value_size, _ssd_entry_bytes),
                32 * 1024 * 1024 * 1024,
                "SSD",
            )
            cfg.ssd.ssd_backend = os.environ.get("UMBP_SSD_BACKEND", "file")
            # This bench builds UMBPConfig directly and never calls
            # UMBPConfig::FromEnvironment(), so the UMBP_SSD_* env vars that
            # configure the tier do NOT reach it on their own -- they have to be
            # applied here through the pybind fields. Leaving direct_io unset is
            # the trap this block exists to avoid: buffered reads can be served
            # entirely from page cache, moving zero bytes to the device, which
            # makes any drive-count or DRAM-vs-SSD comparison meaningless.
            cfg.ssd.direct_io = os.environ.get("UMBP_SSD_DIRECT_IO", "1") not in (
                "0",
                "",
            )
            cfg.ssd.verify_crc = os.environ.get("UMBP_SSD_VERIFY_CRC", "1") not in (
                "0",
                "",
            )
            cfg.ssd.tier_io_threads = int(os.environ.get("UMBP_SSD_TIER_IO_THREADS", 4))
            # posix reports batch_read=false, so a batch degrades to one
            # blocking pread at a time -- which penalises exactly the
            # many-small-reads shape ranged I/O produces.  io_uring is the
            # production setting and the only one where a batch is submitted
            # as a batch.
            if _io_backend is not None:
                cfg.ssd.io.backend = (
                    UMBPIoBackend.IoUring
                    if _io_backend == "io_uring"
                    else UMBPIoBackend.Posix
                )
            if _queue_depth is not None:
                cfg.ssd.io.queue_depth = _queue_depth
            if _segment_bytes is not None:
                cfg.ssd.segment_size_bytes = _segment_bytes
            cfg.ssd.shard_io_threads = int(
                os.environ.get("UMBP_SSD_SHARD_IO_THREADS", 0)
            )
            cfg.ssd.single_flight_reads = os.environ.get(
                "UMBP_SSD_SINGLE_FLIGHT", "1"
            ) not in ("0", "")
            # Hugepage-back the staging arena by default. _staging_plan sizes it
            # from the sweep, so it is routinely 8 GiB+, and the arena is pinned
            # twice: hipHostRegister for GPU access and then ibv_reg_mr for RDMA.
            # At 4 KiB pages that second registration is ~2M pages and fails with
            # `ibv_reg_mr and dmabuf registration both failed ... errno:12
            # (Cannot allocate memory)`, which is FATAL -- the writer dies and the
            # reader only sees `ReadMessageHeader(type) failed ... EOF`, giving no
            # hint that page size was the problem. 2 MiB pages cut the count 512x
            # and the same arena registers fine.
            #
            # Safe as a default: HostMemAllocator falls back to anonymous pages
            # with a warning when MAP_HUGETLB cannot be satisfied, so a host with
            # no hugepage pool still runs (see LogHugepageFallbackOnce).
            dist.ssd_staging_use_hugepages = os.environ.get(
                "UMBP_SSD_STAGING_HUGEPAGES", "1"
            ) not in ("0", "")
            # storage_dir is comma-separated for a multi-drive tier (one entry
            # per physical drive), so each entry needs its own mkdir -- a single
            # makedirs would create one literal directory named "a,b".
            for _d in cfg.ssd.storage_dir.split(","):
                _d = _d.strip()
                if _d:
                    os.makedirs(_d, exist_ok=True)
        else:
            # Keep the SSD tier dark for dram/hbm runs: cfg.ssd.enabled defaults
            # to True, and leaving it on would let the local (non-distributed)
            # SSD path allocate a 32 GB store nobody reads.
            cfg.ssd.enabled = False
        cfg.distributed = dist
        self._client = UMBPClient(cfg)
        self._reader_node_id = reader_node_id
        self._reader_port = reader_port

    def put(self, key: str, payload: bytes, ptr: int, size: int) -> bool:
        ctypes.memmove(ptr, payload, size)
        return self._client.put_from_ptr(key, ptr, size)

    def flush(self):
        self._client.flush()

    def register_memory(self, ptr: int, size: int, loc: str = "cpu", device: int = -1):
        mori_loc = MemoryLocationType.GPU if loc == "gpu" else MemoryLocationType.CPU
        self._client.register_memory(ptr, size, mori_loc, device)

    def get_into_ptr(self, key: str, ptr: int, size: int) -> bool:
        return self._client.get_into_ptr(key, ptr, size)

    def batch_get_into_ptr(self, keys: list, ptrs: list, sizes: list) -> list:
        raw = self._client.batch_get_into_ptr(keys, ptrs, sizes)
        return [sz if hit else 0 for hit, sz in zip(raw, sizes)]

    def wait_visible(self, keys: list, timeout: float = 30.0) -> int:
        return _wait_visible_via_exists(self._client, keys, timeout)

    def batch_exists(self, keys: list) -> list:
        return self._client.batch_exists(keys)

    def batch_exists_consecutive(self, keys: list) -> int:
        return self._client.batch_exists_consecutive(keys)

    def supports_ranged_io(self) -> bool:
        return self._client.supports_ranged_io()

    def batch_get_ranges_into_ptr(self, keys, ptrs, sizes, offsets) -> list:
        return self._client.batch_get_ranges_into_ptr(keys, ptrs, sizes, offsets)

    def make_reader(self) -> "UMBPBackend":
        return UMBPBackend(self._reader_node_id, self._reader_port)


class MooncakeBackend(Backend):
    @classmethod
    def start_server(cls, args):
        scheme, _, rest = args.metadata_server.partition("://")
        meta_hostport, _, meta_path = rest.partition("/")
        meta_host = meta_hostport.partition(":")[0]
        master_host = args.master_server.partition(":")[0]
        # For a split run honor the CLI-provided meta/master ports (role=writer)
        # so the reader on another node can connect using the same addresses.
        cli_meta_port = int(meta_hostport.partition(":")[2] or 0)
        cli_master_port = int(args.master_server.partition(":")[2] or 0)
        meta_port = (
            cli_meta_port if (args.role == "writer" and cli_meta_port) else _free_port()
        )
        master_port = (
            cli_master_port
            if (args.role == "writer" and cli_master_port)
            else _free_port()
        )
        # mooncake_master also binds a metrics/admin HTTP server, hardcoded to
        # 9003 by default. --enable_metric_reporting=false only disables
        # reporting, not the bind, so a leaked master from a prior run holds
        # 9003 and the next master collides. Give it a free port too.
        metrics_port = _free_port()
        meta = _spawn(
            [
                "mooncake_http_metadata_server",
                "--port",
                str(meta_port),
                "--host",
                "0.0.0.0",
            ],
            "/tmp/mc_http_metadata.log",
        )
        master = _spawn(
            [
                "mooncake_master",
                "--rpc_port",
                str(master_port),
                "--metrics_port",
                str(metrics_port),
                "--enable_metric_reporting=false",
            ],
            "/tmp/mc_master.log",
        )
        if not _wait_for_port(meta_port) or not _wait_for_port(master_port):
            print(
                "mooncake servers failed to start; see /tmp/mc_*.log", file=sys.stderr
            )
            sys.exit(1)
        print(
            f"started mooncake servers meta pid={meta.pid} port={meta_port} "
            f"master pid={master.pid} port={master_port}",
            flush=True,
        )
        # Return the addresses the client should use (ports may have been auto-picked).
        return (
            f"{scheme}://{meta_host}:{meta_port}/{meta_path}",
            f"{master_host}:{master_port}",
        )

    def __init__(self):
        # Enable hugepages (2MB) by default, matching the UMBP backend. These are
        # read by the C++ client at store construction / setup() time. Presence of
        # MC_STORE_USE_HUGEPAGE (any value) enables it; MC_STORE_HUGEPAGE_SIZE
        # accepts "2MB" or "1GB". Both stay overridable via the environment.
        os.environ.setdefault("MC_STORE_USE_HUGEPAGE", "1")
        os.environ.setdefault("MC_STORE_HUGEPAGE_SIZE", "2MB")
        segment_size = max(1 << 24, nr_objects * value_size * 2)
        # local_buffer_size is the client-side RDMA staging pool. It is
        # registered on EVERY NIC at setup, so on multi-NIC ionic an 8 GB pool
        # blows up the device/IOMMU page-table footprint (8 GB x N NICs) and
        # ibv_reg_mr returns ENOMEM at >=6 NICs. The zero-copy batch_get_into
        # path never touches it; only sequential put()/single get() stage
        # through it, one 2 MB object at a time, so a small pool is plenty.
        # Override via MC_LOCAL_BUFFER_SIZE (bytes) if a workload needs more.
        local_buffer_size = int(
            os.environ.get("MC_LOCAL_BUFFER_SIZE", 256 * 1024 * 1024)
        )
        self._store = MooncakeDistributedStore()
        # rdma_devices is a comma-separated HCA list. Empty string => Mooncake
        # auto-discovers all RDMA NICs (no longer pinned to a single device).
        rdma_devices = os.environ.get("MC_RDMA_DEVICES", "")
        # Pass the sizes by keyword: setup()'s positional order is
        # (..., global_segment_size, local_buffer_size, ...) -- the reverse of
        # how it reads -- so a positional swap silently mis-sizes the segment
        # (segment too small => puts fail "insufficient space").
        ret = self._store.setup(
            node_address,
            metadata_server,
            global_segment_size=segment_size,
            local_buffer_size=local_buffer_size,
            protocol="rdma",
            rdma_devices=rdma_devices,
            master_server_addr=master_server,
        )
        if ret != 0:
            print(f"mooncake setup failed: {ret}", file=sys.stderr)
            sys.exit(1)
        self._config = ReplicateConfig()
        self._config.replica_num = 1

    def put(self, key: str, payload: bytes, ptr: int, size: int) -> bool:
        return self._store.put(key, payload, self._config) == 0

    def flush(self):
        pass

    def register_memory(self, ptr: int, size: int, loc: str = "cpu", device: int = -1):
        if loc != "cpu":
            raise RuntimeError(
                "MooncakeBackend.register_memory: only cpu buffers are supported"
            )
        self._store.register_buffer(ptr, size)

    def get_into_ptr(self, key: str, ptr: int, size: int) -> bool:
        data = self._store.get(key)
        if data in (None, b""):
            return False
        ctypes.memmove(ptr, bytes(data), min(len(data), size))
        return True

    def batch_get_into_ptr(self, keys: list, ptrs: list, sizes: list) -> list:
        return self._store.batch_get_into(keys, ptrs, sizes)

    def make_reader(self) -> "MooncakeBackend":
        return self


# --- backend init ---


def _wait_visible_via_exists(client, keys: list, timeout: float) -> int:
    deadline = time.time() + timeout
    seen = 0
    while True:
        found = client.batch_exists(keys)
        seen = sum(1 for f in found if f)
        if seen == len(keys) or time.time() >= deadline:
            return seen
        time.sleep(0.25)


class UMBPLocalBackend(Backend):
    """Local mode: a StandaloneClient in this process.  No master, no network.

    The point of keeping this alongside the networked deployments is that it is
    the only configuration where a measurement is purely the storage tier --
    no RDMA, no gRPC, no staging arena -- so it is the baseline that says
    whether a distributed number is bounded by the medium or by the transport.
    """

    @classmethod
    def start_server(cls, args):
        return None  # nothing to start

    def __init__(self, verify_crc: bool = None):
        cfg = UMBPConfig()
        # Must be far smaller than the dataset, or Put keeps everything in the
        # fast tier and the SSD is never read.
        cfg.dram.capacity_bytes = int(
            os.environ.get("UMBP_DRAM_CAPACITY_BYTES", 64 << 20)
        )
        cfg.dram.use_hugepages = os.environ.get("UMBP_DRAM_USE_HUGEPAGES", "0") not in (
            "0",
            "",
        )
        if tier == "ssd":
            cfg.ssd.enabled = True
            cfg.ssd.storage_dir = os.environ.get(
                "UMBP_SSD_STORAGE_DIR", f"/tmp/umbp_ssd_local_{os.getpid()}"
            )
            cfg.ssd.capacity_bytes = int(
                os.environ.get("UMBP_SSD_CAPACITY_BYTES", 64 << 30)
            )
            cfg.ssd.direct_io = os.environ.get("UMBP_SSD_DIRECT_IO", "1") not in (
                "0",
                "",
            )
            cfg.ssd.tier_io_threads = int(os.environ.get("UMBP_SSD_TIER_IO_THREADS", 8))
            if _segment_bytes is not None:
                cfg.ssd.segment_size_bytes = _segment_bytes
            if _io_backend is not None:
                cfg.ssd.io.backend = (
                    UMBPIoBackend.IoUring
                    if _io_backend == "io_uring"
                    else UMBPIoBackend.Posix
                )
            if _queue_depth is not None:
                cfg.ssd.io.queue_depth = _queue_depth
            # verify_crc is swept by ranged_perf, so the caller may override the
            # env-derived value per iteration.
            cfg.ssd.verify_crc = (
                verify_crc
                if verify_crc is not None
                else os.environ.get("UMBP_SSD_VERIFY_CRC", "1") not in ("0", "")
            )
            for _d in cfg.ssd.storage_dir.split(","):
                _d = _d.strip()
                if _d:
                    os.makedirs(_d, exist_ok=True)
        else:
            cfg.ssd.enabled = False
            cfg.dram.capacity_bytes = _dram_capacity_bytes
        # Promotion would pull an object into DRAM on its first read, so every
        # later measurement would be a DRAM hit rather than a tier hit.
        cfg.eviction.auto_promote_on_read = os.environ.get(
            "UMBP_AUTO_PROMOTE_ON_READ", "0"
        ) not in ("0", "")
        self._client = UMBPClient(cfg)

    def put(self, key: str, payload: bytes, ptr: int, size: int) -> bool:
        ctypes.memmove(ptr, payload, size)
        return self._client.batch_put_from_ptr([key], [ptr], [size])[0]

    def flush(self):
        self._client.flush()

    def register_memory(self, ptr: int, size: int, loc: str = "cpu", device: int = -1):
        return True  # local mode is a CPU-local memcpy; nothing to pin

    def get_into_ptr(self, key: str, ptr: int, size: int) -> bool:
        return self._client.batch_get_into_ptr([key], [ptr], [size])[0]

    def batch_get_into_ptr(self, keys: list, ptrs: list, sizes: list) -> list:
        res = self._client.batch_get_into_ptr(keys, ptrs, sizes)
        return [sizes[i] if r else 0 for i, r in enumerate(res)]

    def wait_visible(self, keys: list, timeout: float = 30.0) -> int:
        return _wait_visible_via_exists(self._client, keys, timeout)

    def batch_exists(self, keys: list) -> list:
        return self._client.batch_exists(keys)

    def batch_exists_consecutive(self, keys: list) -> int:
        return self._client.batch_exists_consecutive(keys)

    def supports_ranged_io(self) -> bool:
        return self._client.supports_ranged_io()

    def batch_get_ranges_into_ptr(self, keys, ptrs, sizes, offsets) -> list:
        return self._client.batch_get_ranges_into_ptr(keys, ptrs, sizes, offsets)

    def make_reader(self) -> "UMBPLocalBackend":
        return self  # one process, one client: reads come from the same store


class UMBPServerBackend(Backend):
    """Standalone-process mode: forwards to a UMBP server over gRPC + shm.

    This is the deployment the tree connector actually runs, and the only one
    where the client and the pool are different processes.  The server may be
    local- or distributed-backed; get_backend_mode() reports which, and with a
    distributed-backed server a read leaves this node over RDMA.
    """

    @classmethod
    def start_server(cls, args):
        # The server is long-lived and externally managed (it owns the node's
        # pool), so the bench never starts or stops one.
        return None

    def __init__(self, address: str):
        cfg = UMBPConfig()
        sp = UMBPStandaloneProcessConfig()
        sp.address = address
        sp.startup_timeout_ms = int(os.environ.get("UMBP_STANDALONE_TIMEOUT_MS", 15000))
        # standalone_process ONLY: the factory tests `distributed` first, so a
        # config carrying both silently builds a second DistributedClient
        # instead of a client that forwards to the server.
        cfg.standalone_process = sp
        self._client = UMBPClient(cfg)
        print(
            f"connected to standalone server {address}: "
            f"mode={self._client.get_deployment_mode()} "
            f"backend={self._client.get_backend_mode()} "
            f"ranged={self._client.supports_ranged_io()}",
            flush=True,
        )

    def put(self, key: str, payload: bytes, ptr: int, size: int) -> bool:
        ctypes.memmove(ptr, payload, size)
        return self._client.batch_put_from_ptr([key], [ptr], [size])[0]

    def flush(self):
        self._client.flush()

    def register_memory(self, ptr: int, size: int, loc: str = "cpu", device: int = -1):
        mori_loc = MemoryLocationType.GPU if loc == "gpu" else MemoryLocationType.CPU
        return self._client.register_memory(ptr, size, mori_loc, device)

    def get_into_ptr(self, key: str, ptr: int, size: int) -> bool:
        return self._client.batch_get_into_ptr([key], [ptr], [size])[0]

    def batch_get_into_ptr(self, keys: list, ptrs: list, sizes: list) -> list:
        res = self._client.batch_get_into_ptr(keys, ptrs, sizes)
        return [sizes[i] if r else 0 for i, r in enumerate(res)]

    def wait_visible(self, keys: list, timeout: float = 30.0) -> int:
        return _wait_visible_via_exists(self._client, keys, timeout)

    def batch_exists(self, keys: list) -> list:
        return self._client.batch_exists(keys)

    def batch_exists_consecutive(self, keys: list) -> int:
        return self._client.batch_exists_consecutive(keys)

    def supports_ranged_io(self) -> bool:
        return self._client.supports_ranged_io()

    def batch_get_ranges_into_ptr(self, keys, ptrs, sizes, offsets) -> list:
        return self._client.batch_get_ranges_into_ptr(keys, ptrs, sizes, offsets)

    def make_reader(self) -> "UMBPServerBackend":
        # The dataset lives in the SERVER, not in this process, so a reader is
        # just another client of the same socket -- and unlike the distributed
        # case the writer may exit without taking the data with it.
        return self


_BACKENDS = {
    "umbp": UMBPBackend,
    "umbp-local": UMBPLocalBackend,
    "umbp-server": UMBPServerBackend,
    "mooncake": MooncakeBackend,
}


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        return _s.getsockname()[1]


# --- split writer/reader rendezvous (TCP, hosted on the writer node) ---


# Idle timeout, NOT a run-duration budget: the reader sends KEEPALIVE_INTERVAL
# pings (see _SyncKeepalive), so this only expires when the peer has gone truly
# silent. It used to be a flat 600s wait for b"DONE", which made the writer give
# up on any sweep slower than 10 minutes -- and the writer's exit kills the
# umbp_master it spawned (PR_SET_PDEATHSIG) and unregisters its keys, so the
# still-running reader saw every subsequent read as a MISS with no error on
# either side. A 4 GiB SSD sweep reads at ~190 MiB/s and takes ~14 minutes, so
# it hit this every time while DRAM/HBM (seconds) never did.
SYNC_IDLE_TIMEOUT = 300.0
KEEPALIVE_INTERVAL = 30.0


def _sync_recv_until(conn, token: bytes, timeout: float = SYNC_IDLE_TIMEOUT):
    conn.settimeout(timeout)
    buf = b""
    while token not in buf:
        try:
            chunk = conn.recv(64)
        except socket.timeout as exn:
            # Loud and specific: the old silent death cost a long debugging
            # session, because "writer vanished" surfaced only as 100% misses
            # in the reader's RESULT lines.
            raise TimeoutError(
                f"no data from sync peer for {timeout:.0f}s while waiting for "
                f"{token!r}. The peer is either dead or wedged; a slow-but-live "
                f"reader should be sending keepalives every "
                f"{KEEPALIVE_INTERVAL:.0f}s."
            ) from exn
        if not chunk:
            raise ConnectionError(f"sync peer closed before sending {token!r}")
        buf += chunk


class _SyncKeepalive:
    """Reader-side liveness pings on the rendezvous socket.

    The writer must hold its dataset until the reader is done, and the reader's
    runtime is unbounded (it scales with medium speed x objects x passes). Rather
    than guess a timeout large enough for the slowest medium, the reader proves
    it is alive; any byte resets the writer's idle timer, and the writer's
    accumulating buffer just ignores non-token bytes.

    Shares the socket with the final b"DONE", so sends are serialized under a
    lock and the thread is stopped before DONE goes out -- interleaved sendall()
    from two threads could split the token.
    """

    def __init__(self, conn, interval: float = KEEPALIVE_INTERVAL):
        self._conn = conn
        self._interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5.0)
        return False

    @property
    def lock(self):
        return self._lock

    def _loop(self):
        while not self._stop.wait(self._interval):
            with self._lock:
                try:
                    self._conn.sendall(b"PING\n")
                except OSError:
                    return  # writer gone; the read path will surface it


def _sync_listen(port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    return srv


def _sync_connect(host: str, port: int, timeout: float = 300.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            return socket.create_connection((host, port), timeout=5)
        except OSError as e:
            last = e
            time.sleep(0.5)
    raise TimeoutError(f"could not reach writer sync {host}:{port}: {last}")


def _bind_numa_cpus(node: int):
    # Pin CPU affinity to the given NUMA node's cpulist. Memory placement is
    # handled separately (UMBP DRAM tier + bench buffers take an explicit
    # numa_node), but binding CPUs first means any first-touch allocation and
    # the IO engine's worker threads also land on this node. Call before
    # constructing any client or allocating buffers so threads inherit it.
    if node < 0:
        return
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            spec = f.read().strip()
    except OSError as e:
        print(
            f"WARNING: NUMA node {node} has no cpulist ({e}); not binding CPUs",
            file=sys.stderr,
        )
        return
    cpus = set()
    for part in spec.split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        return
    os.sched_setaffinity(0, cpus)
    print(f"bound to NUMA node {node}: {len(cpus)} cpus ({spec})", flush=True)


# Only the writer (or a combined both) owns the master/metadata server(s); the
# reader connects to the writer's already-running master, so it never starts one.
# local mode has no server at all, and a standalone server is long-lived and
# externally managed (it owns the node's pool), so neither is started here.
if args.start_server and role != "reader" and deployment == "distributed":
    _resolved = _BACKENDS[args.backend].start_server(args)
    # The server may have been moved to an auto-picked free port; point the
    # client at the address(es) it actually bound.
    if backend_name == "umbp":
        master_address = _resolved
    else:
        metadata_server, master_server = _resolved

# Pin to the requested NUMA node before any client/thread/buffer is created so
# affinity and first-touch placement are inherited. In a same-node split the
# writer and reader pass different --numa-node values (cross-NUMA transfer).
_bind_numa_cpus(numa_node)

# Build the writer only for role in {both, writer}; build a standalone reader
# backend lazily (role=reader uses _build_reader; role=both uses
# writer.make_reader() so the two clients still share one process).
writer = None
if deployment == "local":

    def _build_reader():
        return UMBPLocalBackend()

    if role in ("both", "writer"):
        writer = UMBPLocalBackend()
        print(
            f"backend=umbp-local tier={tier} role={role} "
            f"ranged={writer.supports_ranged_io()}",
            flush=True,
        )
elif deployment == "standalone-server":

    def _build_reader():
        return UMBPServerBackend(server_address)

    if role in ("both", "writer"):
        writer = UMBPServerBackend(server_address)
        print(
            f"backend=umbp-server tier={tier} role={role} server={server_address}",
            flush=True,
        )
elif backend_name == "umbp":

    def _build_reader():
        rid = f"reader_{os.getpid()}_{int(time.time())}"
        return UMBPBackend(rid, _free_port())

    if role in ("both", "writer"):
        ts = int(time.time())
        node_id = f"{command}_{os.getpid()}_{ts}"
        reader_node_id = f"reader_{os.getpid()}_{ts}"
        peer_service_port = _free_port()
        reader_peer_port = _free_port()
        writer = UMBPBackend(
            node_id, peer_service_port, reader_node_id, reader_peer_port
        )
        print(
            f"backend=umbp tier={tier} dst_loc={dst_loc} role={role} "
            f"writer_node_id={node_id} reader_node_id={reader_node_id} "
            f"node_address={node_address} writer_port={peer_service_port} "
            f"reader_port={reader_peer_port}",
            flush=True,
        )
elif backend_name == "mooncake":

    def _build_reader():
        return MooncakeBackend()

    if role in ("both", "writer"):
        print(
            f"backend=mooncake role={role} node_address={node_address} metadata={metadata_server} master={master_server}",
            flush=True,
        )
        writer = MooncakeBackend()

# --- host buffer allocation (hugepage-backed via mori's UMBP allocator) ---


class HostBuffer:
    """Host DMA buffer allocated through mori's UMBPHostMemAllocator so the bench
    buffers are hugepage-backed (matching the DRAM tier) instead of the 4 KiB-paged
    ctypes arrays used previously. Keeps the handle alive for the object's lifetime
    and exposes a raw integer `ptr`. Honors the same UMBP_DRAM_USE_HUGEPAGES /
    UMBP_DRAM_HUGEPAGE_SIZE env knobs (default: on, 2MB)."""

    _allocator = None

    def __init__(self, size: int):
        from mori.umbp import UMBPHostBufferBacking, UMBPHostMemAllocator

        if HostBuffer._allocator is None:
            HostBuffer._allocator = UMBPHostMemAllocator()
        use_hp = os.environ.get("UMBP_DRAM_USE_HUGEPAGES", "1") not in ("0", "")
        hp_size = int(os.environ.get("UMBP_DRAM_HUGEPAGE_SIZE", 2 * 1024 * 1024))
        # Standalone-process mode resolves (region_base, offset) triples against
        # the SERVER's mapping of this memory, so the buffer has to be shm-backed
        # or nothing can map it.  Getting this wrong is quiet: the call returns a
        # per-key false, which at the call site is indistinguishable from the
        # store refusing the operation.
        want = _buffer_backing
        if want == "auto":
            want = (
                "shm"
                if deployment == "standalone-server"
                else ("hugetlb" if use_hp else "anon")
            )
        backing = {
            "shm": (
                UMBPHostBufferBacking.AnonymousShmHugetlb
                if use_hp
                else UMBPHostBufferBacking.AnonymousShm
            ),
            "hugetlb": UMBPHostBufferBacking.AnonymousHugetlb,
            "anon": UMBPHostBufferBacking.Anonymous,
        }[want]
        numa = int(os.environ.get("UMBP_DRAM_NUMA_NODE", numa_node))
        self._handle = HostBuffer._allocator.alloc(size, backing, hp_size, numa, True)
        if not self._handle and want == "shm" and use_hp:
            # Reserved hugetlb shm is the common thing to be missing; plain shm
            # still satisfies the mapping requirement above.
            backing = UMBPHostBufferBacking.AnonymousShm
            self._handle = HostBuffer._allocator.alloc(
                size, backing, hp_size, numa, True
            )
            if self._handle:
                print(
                    "NOTE: hugetlb shm unavailable, using 4 KiB shm pages",
                    file=sys.stderr,
                )
        if not self._handle:
            raise RuntimeError(f"UMBPHostMemAllocator.alloc({size}) failed")
        if use_hp and self._handle.actual_backing == UMBPHostBufferBacking.Anonymous:
            print(
                "WARNING: requested hugepages but kernel demoted to 4 KiB pages; "
                "check vm.nr_hugepages and HugePages_Free in /proc/meminfo",
                file=sys.stderr,
            )
        self.ptr = int(self._handle.ptr)
        self.size = size

    def __del__(self):
        # At interpreter shutdown module globals are torn down first, so the
        # class object this method references can already be None. Bail out
        # rather than raising an "Exception ignored in __del__" per buffer,
        # which is pure noise that hides real errors in the same output.
        handle = getattr(self, "_handle", None)
        cls = globals().get("HostBuffer")
        if handle is not None and cls is not None and cls._allocator is not None:
            try:
                cls._allocator.free(handle)
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass
            self._handle = None


# --- device (GPU) buffer allocation for HBM D2D testing, via raw HIP calls ---
#
# ctypes + libamdhip64.so rather than a torch dependency: all that's needed is
# an allocation + memset/memcpy primitive, and this keeps the bench usable in
# a container without torch installed.

_HIP_MEMCPY_DEVICE_TO_HOST = 2


def _load_hip():
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError(
        "libamdhip64.so not found; ROCm/HIP runtime required for --dst-loc gpu"
    )


def _hip_check(rc: int, what: str):
    if rc != 0:
        raise RuntimeError(f"{what} failed: hipError_t={rc}")


class DeviceBuffer:
    """GPU-resident buffer allocated via raw HIP (hipMalloc), so a read from an
    HBM-tier pool exercises HbmCopyEngine's device-to-device hipMemcpy path
    from Python instead of device-to-host."""

    def __init__(self, size: int, device: int = 0):
        self._hip = _load_hip()
        _hip_check(self._hip.hipSetDevice(ctypes.c_int(device)), "hipSetDevice")
        raw_ptr = ctypes.c_void_p()
        _hip_check(
            self._hip.hipMalloc(ctypes.byref(raw_ptr), ctypes.c_size_t(size)),
            f"hipMalloc({size})",
        )
        _hip_check(self._hip.hipMemset(raw_ptr, 0, ctypes.c_size_t(size)), "hipMemset")
        self.ptr = raw_ptr.value
        self.size = size
        self.device = device

    def zero(self):
        _hip_check(
            self._hip.hipMemset(
                ctypes.c_void_p(self.ptr), 0, ctypes.c_size_t(self.size)
            ),
            "hipMemset",
        )

    def read_back(self, offset: int, size: int) -> bytes:
        """Copy `size` bytes at `offset` back to host (D2H), for correctness
        verification only -- never on the batch_perf hot path."""
        host_buf = ctypes.create_string_buffer(size)
        _hip_check(
            self._hip.hipMemcpy(
                host_buf,
                ctypes.c_void_p(self.ptr + offset),
                ctypes.c_size_t(size),
                ctypes.c_int(_HIP_MEMCPY_DEVICE_TO_HOST),
            ),
            "hipMemcpy D2H (verify)",
        )
        return host_buf.raw

    def __del__(self):
        ptr = getattr(self, "ptr", None)
        hip = getattr(self, "_hip", None)
        if ptr and hip is not None:
            try:
                hip.hipFree(ctypes.c_void_p(ptr))
            except Exception:
                pass
            self.ptr = None


if role in ("both", "writer"):
    buf = HostBuffer(value_size)
    ptr = buf.ptr
    writer.register_memory(ptr, value_size)


# --- shared write helper ---


def write_all() -> bool:
    payload = expected_payload(value_size)
    ok = fail = 0
    for i in range(nr_objects):
        if writer.put(make_key(i), payload, ptr, value_size):
            ok += 1
        else:
            fail += 1
    print(f"WRITE_DONE ok={ok} fail={fail} total={nr_objects}", flush=True)
    writer.flush()
    # See Backend.wait_visible: flush() does not guarantee the master has the
    # ADDs yet, and reading too early reports a total miss that is
    # indistinguishable from a broken read path.
    _keys = [make_key(i) for i in range(nr_objects)]
    _seen = writer.wait_visible(
        _keys, float(os.environ.get("UMBP_VISIBLE_TIMEOUT_S", 30))
    )
    if _seen != nr_objects:
        print(
            f"WARNING: only {_seen}/{nr_objects} keys visible after the write "
            f"barrier; reads will under-count. Raise UMBP_VISIBLE_TIMEOUT_S or "
            f"check the peer's heartbeat.",
            flush=True,
        )
    else:
        print(f"WRITE_VISIBLE {_seen}/{nr_objects}", flush=True)
    return fail == 0


# --- write + reader setup (role-aware) ---
#
# role=writer : write the dataset, then host a TCP rendezvous on this node:
#   signal READY (data flushed) -> wait for the reader's DONE -> release & exit.
#   The writer must outlive the reads because UMBP serves them via RDMA out of
#   this process's DRAM.
# role=reader : connect to the writer's rendezvous, wait for READY, read, then
#   signal DONE.
# role=both   : original single-process behavior, no rendezvous.

if role == "writer":
    srv = _sync_listen(args.sync_port)
    write_all()
    print(
        f"WRITER_READY holding dataset; waiting for reader on :{args.sync_port}",
        flush=True,
    )
    conn, peer = srv.accept()
    conn.sendall(b"READY\n")
    print(f"reader connected from {peer[0]}:{peer[1]}; signalled READY", flush=True)
    _sync_recv_until(conn, b"DONE")
    print("READER_DONE; writer releasing dataset and exiting", flush=True)
    conn.close()
    srv.close()
    sys.exit(0)

sync_conn = None
if role == "reader":
    sync_conn = _sync_connect(_resolve_host, args.sync_port)
    print(
        f"connected to writer sync {_resolve_host}:{args.sync_port}; waiting for READY",
        flush=True,
    )
    _sync_recv_until(sync_conn, b"READY")
    print("writer signalled READY; starting reads", flush=True)
    # Prove liveness for the whole read phase, however long the medium makes it.
    sync_keepalive = _SyncKeepalive(sync_conn)
    sync_keepalive.__enter__()
    atexit.register(sync_keepalive.__exit__)
    reader = _build_reader()
else:  # both
    write_all()
    reader = writer.make_reader()

# --- commands (reader side) ---

exit_code = 0

if command == "correctness":
    if dst_loc == "gpu":
        read_buf = DeviceBuffer(value_size, device=hbm_device)
        read_ptr = read_buf.ptr
        reader.register_memory(read_ptr, value_size, loc="gpu", device=hbm_device)
    else:
        read_buf = HostBuffer(value_size)
        read_ptr = read_buf.ptr
        reader.register_memory(read_ptr, value_size)
    hits = misses = mismatches = 0
    expected = expected_payload(value_size)
    for i in range(nr_objects):
        if dst_loc == "gpu":
            read_buf.zero()
        else:
            ctypes.memset(read_ptr, 0, value_size)
        if not reader.get_into_ptr(make_key(i), read_ptr, value_size):
            misses += 1
            continue
        hits += 1
        actual = (
            read_buf.read_back(0, value_size)
            if dst_loc == "gpu"
            else ctypes.string_at(read_ptr, value_size)
        )
        if actual != expected:
            mismatches += 1
    print(
        f"CORRECTNESS hits={hits} misses={misses} mismatches={mismatches} total={nr_objects}",
        flush=True,
    )
    if hits == nr_objects and mismatches == 0:
        print("CORRECTNESS_OK", flush=True)
    else:
        print("CORRECTNESS_FAILED", flush=True)
        exit_code = 1

elif command == "batch_perf":
    passes = args.passes
    batch_sizes = args.batch_sizes if args.batch_sizes else [1, 32, 128, nr_objects]

    keys = [make_key(i) for i in range(nr_objects)]
    buf_total = value_size * nr_objects
    whole = (
        DeviceBuffer(buf_total, device=hbm_device)
        if dst_loc == "gpu"
        else HostBuffer(buf_total)
    )
    base = whole.ptr
    # Destination slot for each key. 'contiguous' packs them back-to-back so the
    # IO engine can coalesce adjacent SGEs; 'shuffled' assigns each key a random
    # slot in the same buffer so consecutive keys land at non-adjacent addresses
    # (simulates a non-consecutive transfer; defeats adjacency coalescing).
    if args.dest_layout == "shuffled":
        import random

        slots = list(range(nr_objects))
        random.Random(args.shuffle_seed).shuffle(slots)
        print(f"dest_layout=shuffled seed={args.shuffle_seed}", flush=True)
    else:
        slots = range(nr_objects)
    ptrs = [base + slot * value_size for slot in slots]
    sizes = [value_size] * nr_objects

    if dst_loc == "gpu":
        reader.register_memory(base, buf_total, loc="gpu", device=hbm_device)
    else:
        reader.register_memory(base, buf_total)

    for batch_size in batch_sizes:
        batches = [keys[i : i + batch_size] for i in range(0, nr_objects, batch_size)]
        ptr_batches = [
            ptrs[i : i + batch_size] for i in range(0, nr_objects, batch_size)
        ]
        size_batches = [
            sizes[i : i + batch_size] for i in range(0, nr_objects, batch_size)
        ]

        # warmup
        for kb, pb, sb in zip(batches, ptr_batches, size_batches):
            reader.batch_get_into_ptr(kb, pb, sb)

        hits = misses = 0
        total_bytes = 0
        start = time.perf_counter()
        for _ in range(passes):
            for kb, pb, sb in zip(batches, ptr_batches, size_batches):
                results = reader.batch_get_into_ptr(kb, pb, sb)
                for r in results:
                    if r > 0:
                        hits += 1
                        total_bytes += r
                    else:
                        misses += 1
        elapsed = time.perf_counter() - start
        total_reqs = nr_objects * passes
        total_batches = len(batches) * passes
        print(
            f"RESULT batch_size={batch_size} hits={hits} misses={misses} requests={total_reqs} "
            f"batches={total_batches} duration={elapsed:.3f}s "
            f"req_per_s={total_reqs / elapsed:.2f} batch_per_s={total_batches / elapsed:.2f} "
            f"MiB_per_s={(total_bytes / 1024 / 1024) / elapsed:.2f} "
            f"avg_latency_ms={(elapsed / total_reqs) * 1000:.3f}",
            flush=True,
        )

elif command == "exists_perf":
    # Presence-probe sweep: the shape a prefix cache uses before it reads.
    #
    # Separate from batch_perf on purpose.  A get has a local half that is
    # already parallel and already batched, so a routing change is only part of
    # its cost.  An Exists moves no bytes at all -- under route-first it is
    # ONE master RPC and nothing else -- so it isolates what consulting the
    # master actually costs, with no copy to hide behind.
    #
    # Two key populations, because local-first is asymmetric:
    #   present -- this node holds them, so a local resolve is conclusive and
    #              the master is never asked.
    #   absent  -- a local miss proves nothing, so every key still goes to the
    #              master.  This is the worst case: the local resolve is pure
    #              added work.
    passes = args.passes
    batch_sizes = args.batch_sizes if args.batch_sizes else [1, 32, 128, nr_objects]

    present_keys = [make_key(i) for i in range(nr_objects)]
    # Same key shape, never written.  make_key's namespace is the object index,
    # so pushing past nr_objects cannot collide with the dataset.
    absent_keys = [make_key(i) for i in range(nr_objects, 2 * nr_objects)]

    def _sweep(label: str, probe_keys: list, fn, unit: str):
        for batch_size in batch_sizes:
            batches = [
                probe_keys[i : i + batch_size]
                for i in range(0, len(probe_keys), batch_size)
            ]
            for kb in batches:  # warmup
                fn(kb)

            found = 0
            start = time.perf_counter()
            for _ in range(passes):
                for kb in batches:
                    found += fn(kb)
            elapsed = time.perf_counter() - start
            total_reqs = len(probe_keys) * passes
            total_batches = len(batches) * passes
            print(
                f"RESULT op={unit} population={label} batch_size={batch_size} "
                f"found={found} requests={total_reqs} batches={total_batches} "
                f"duration={elapsed:.3f}s key_per_s={total_reqs / elapsed:.2f} "
                f"batch_per_s={total_batches / elapsed:.2f} "
                f"avg_latency_ms={(elapsed / total_batches) * 1000:.4f}",
                flush=True,
            )

    # WHICH client probes decides what is being measured, and it is the whole
    # point of this command:
    #   holder  -- the node that wrote the keys.  Under local_first its own
    #              backends answer for `present` and the master is never asked.
    #              This is the single-node deployment.
    #   probe   -- the separate reader node, which holds nothing.  Every key,
    #              present or not, has to go to the master.  Unlike batch_perf
    #              nothing re-caches onto it here, because a presence query
    #              moves no bytes.
    clients = [("probe", reader)]
    if role == "both":
        clients.insert(0, ("holder", writer))

    try:
        for who, client in clients:
            for pop, pkeys in (("present", present_keys), ("absent", absent_keys)):
                _sweep(
                    f"{who}/{pop}",
                    pkeys,
                    lambda kb, c=client: sum(c.batch_exists(kb)),
                    "exists",
                )
                _sweep(
                    f"{who}/{pop}",
                    pkeys,
                    lambda kb, c=client: c.batch_exists_consecutive(kb),
                    "exists_consecutive",
                )
    except NotImplementedError as exc:
        print(f"EXISTS_UNSUPPORTED {exc}", flush=True)
        sys.exit(2)

elif command == "ranged_perf":
    # Ranged (sub-object) vs whole-object over the SAME keys in the SAME
    # process.  The headline is the RATIO at each fetched fraction: it cancels
    # the node's load, which absolute MiB/s on a shared machine does not.
    #
    # Useful bandwidth counts only the bytes the caller asked for.  Whole-object
    # moves object_size per key regardless, so its useful figure is deliberately
    # the smaller one -- that gap IS the thing being measured.
    import statistics

    # WHICH client issues the reads decides what is measured.  The bench's
    # `reader` is a SEPARATE UMBP node holding nothing, so its ranged reads are
    # remote until ranged_locality_prefetch happens to have pulled an object
    # over -- background, best-effort, and therefore not a controlled input.
    # Probing on the holder makes every read local by construction, which is
    # the single-node deployment local_first exists for.
    if os.environ.get("UMBP_PROBE_ON_HOLDER", "0") not in ("0", "false", "False", ""):
        if role != "both":
            print(
                "ERROR: UMBP_PROBE_ON_HOLDER needs role=both (there is no local "
                "holder in this process otherwise)",
                file=sys.stderr,
            )
            sys.exit(2)
        reader = writer
        print("probe_on=holder", flush=True)
    else:
        print("probe_on=reader", flush=True)

    if not reader.supports_ranged_io():
        print(
            "ABORT: backend reports supports_ranged_io()=False. Pass "
            "--ranged-scratch-bytes (or set UMBP_DISTRIBUTED_RANGED_SCRATCH_BYTES "
            "on the standalone server) and retry.",
            file=sys.stderr,
        )
        sys.exit(2)

    passes = args.passes
    batch = min(args.batch, nr_objects)
    first = args.first_layer
    groups = sorted(
        {g for g in args.groups if 0 < g <= layers - first} | {layers - first}
    )
    keys = [make_key(i) for i in range(nr_objects)]

    # Destinations are laid out LAYER-MAJOR (buffer[layer][object]) on purpose,
    # so one object's slices land batch*layer_bytes apart.  The tree connector
    # keeps one buffer per layer, and a destination-contiguous layout would
    # flatter the ranged path by making its scattered writes adjacent.
    ranged_dst = HostBuffer(layers * batch * layer_bytes)
    whole_dst = HostBuffer(batch * value_size)
    reader.register_memory(ranged_dst.ptr, layers * batch * layer_bytes)
    reader.register_memory(whole_dst.ptr, batch * value_size)

    def _slot(layer: int, obj: int) -> int:
        return ranged_dst.ptr + (layer * batch + obj) * layer_bytes

    def _windows(n_passes: int):
        # Walk the dataset rather than re-reading key 0..batch every pass: a
        # fixed window is served from whatever the tier staged last time.
        span = max(1, nr_objects - batch + 1)
        return [keys[(i * batch) % span :][:batch] for i in range(n_passes + 1)]

    def _time_ranged(group, n_passes):
        sizes_v = [[layer_bytes] * len(group)] * batch
        offs_v = [[ly * layer_bytes for ly in group]] * batch
        out = []
        for w in _windows(n_passes):
            dsts = [[_slot(ly, o) for ly in group] for o in range(len(w))]
            t0 = time.perf_counter()
            res = reader.batch_get_ranges_into_ptr(
                w, dsts, sizes_v[: len(w)], offs_v[: len(w)]
            )
            dt = time.perf_counter() - t0
            if not all(res):
                raise RuntimeError(
                    f"ranged get missed {list(res).count(False)}/{len(res)} keys"
                )
            out.append(dt)
        return out[1:]  # drop the untimed warmup

    def _time_whole(n_passes):
        out = []
        for w in _windows(n_passes):
            dsts = [whole_dst.ptr + o * value_size for o in range(len(w))]
            t0 = time.perf_counter()
            res = reader.batch_get_into_ptr(w, dsts, [value_size] * len(w))
            dt = time.perf_counter() - t0
            if not all(r > 0 for r in res):
                raise RuntimeError("whole-object get missed keys")
            out.append(dt)
        return out[1:]

    # Correctness before timing: a fast wrong answer is the failure mode ranged
    # I/O makes easy, and it is invisible in a bandwidth number.
    _probe = list(range(first, min(first + 4, layers)))
    _w = keys[:batch]
    _res = reader.batch_get_ranges_into_ptr(
        _w,
        [[_slot(ly, o) for ly in _probe] for o in range(len(_w))],
        [[layer_bytes] * len(_probe)] * len(_w),
        [[ly * layer_bytes for ly in _probe]] * len(_w),
    )
    _expected = expected_payload(value_size)
    _bad = 0
    for o in range(min(len(_w), 8)):
        for ly in _probe:
            got = ctypes.string_at(_slot(ly, o), min(64, layer_bytes))
            want = _expected[ly * layer_bytes : ly * layer_bytes + len(got)]
            if got != want:
                _bad += 1
    if not all(_res) or _bad:
        print(
            f"ABORT: ranged correctness failed (misses={list(_res).count(False)} "
            f"mismatched_slices={_bad})",
            file=sys.stderr,
        )
        sys.exit(2)
    print("CORRECTNESS_OK (ranged bytes match the whole-object payload)", flush=True)

    whole_med = statistics.median(_time_whole(passes))
    print(
        f"\nobject={value_size}B layers={layers} layer={layer_bytes}B "
        f"batch={batch} passes={passes} tier={tier} deployment={deployment}\n",
        flush=True,
    )
    hdr = (
        f"{'G':>4} {'fetched':>8} {'ranged_ms':>10} {'whole_ms':>10} "
        f"{'ranged_MiB/s':>13} {'whole_MiB/s':>12} {'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr), flush=True)
    for g in groups:
        group = list(range(first, first + g))
        med = statistics.median(_time_ranged(group, passes))
        useful = batch * g * layer_bytes
        frac = g / layers
        r_bw = useful / med / (1 << 20)
        w_bw = useful / whole_med / (1 << 20)
        print(
            f"{g:>4} {frac * 100:>7.1f}% {med * 1e3:>10.2f} {whole_med * 1e3:>10.2f} "
            f"{r_bw:>13.1f} {w_bw:>12.1f} {whole_med / med:>7.2f}x",
            flush=True,
        )
        print(
            f"RESULT group={g} fetched_frac={frac:.4f} ranged_ms={med * 1e3:.3f} "
            f"whole_ms={whole_med * 1e3:.3f} ranged_useful_MiB_s={r_bw:.1f} "
            f"whole_useful_MiB_s={w_bw:.1f} speedup={whole_med / med:.3f} "
            f"ranged_wire_MiB={useful / (1 << 20):.1f} "
            f"whole_wire_MiB={batch * value_size / (1 << 20):.1f}",
            flush=True,
        )

# In a split run, tell the writer we're finished so it can release the dataset
# and exit; the writer is blocked in _sync_recv_until(b"DONE") until this lands.
if sync_conn is not None:
    # Stop the keepalive first: it shares this socket, and two threads in
    # sendall() can interleave bytes and split the token the writer greps for.
    sync_keepalive.__exit__()
    try:
        with sync_keepalive.lock:
            sync_conn.sendall(b"DONE\n")
        sync_conn.close()
    except OSError:
        pass

if exit_code:
    sys.exit(exit_code)
