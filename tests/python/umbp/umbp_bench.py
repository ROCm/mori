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
    proc = subprocess.Popen(
        argv, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=_die_with_parent
    )
    atexit.register(lambda: _kill_group(proc))
    return proc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umbp_bench.py")
    cmd_sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def _add_backend_sub(cmd_parser, *, batch_perf: bool = False):
        be_sub = cmd_parser.add_subparsers(
            dest="backend", required=True, metavar="BACKEND"
        )
        for be_name in ("umbp", "mooncake"):
            be_p = be_sub.add_parser(be_name)
            if be_name == "umbp":
                be_p.add_argument("master_address", help="e.g. ip:port")
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
            if batch_perf:
                be_p.add_argument("--passes", type=int, default=10)
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
    return parser


args = _build_parser().parse_args()
command = args.command
backend_name = args.backend
nr_objects = args.nr_objects
value_size = args.value_size
role = args.role
numa_node = args.numa_node

if role != "both" and not args.key_prefix:
    print(
        "ERROR: split runs (--role writer/reader) require a shared --key-prefix on "
        "both writer and reader so they compute the same keys",
        file=sys.stderr,
    )
    sys.exit(2)

if backend_name == "umbp":
    from mori.umbp import UMBPClient, UMBPConfig, UMBPDistributedConfig

    master_address = args.master_address
    _resolve_host = master_address.split(":")[0]
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
    def register_memory(self, ptr: int, size: int):
        """Register a memory region for RDMA (no-op for backends that don't need it)."""

    @abstractmethod
    def get_into_ptr(self, key: str, ptr: int, size: int) -> bool:
        """Read value into ptr. Returns True on hit."""

    @abstractmethod
    def batch_get_into_ptr(self, keys: list, ptrs: list, sizes: list) -> list:
        """Batch read into ptrs. Returns bytes_read per key (0 = miss)."""

    @abstractmethod
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
        cfg.dram.capacity_bytes = int(
            os.environ.get("UMBP_DRAM_CAPACITY_BYTES", 8 * 1024 * 1024 * 1024)
        )
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
        dist.io_engine.host = node_address
        cfg.distributed = dist
        self._client = UMBPClient(cfg)
        self._reader_node_id = reader_node_id
        self._reader_port = reader_port

    def put(self, key: str, payload: bytes, ptr: int, size: int) -> bool:
        ctypes.memmove(ptr, payload, size)
        return self._client.put_from_ptr(key, ptr, size)

    def flush(self):
        self._client.flush()

    def register_memory(self, ptr: int, size: int):
        self._client.register_memory(ptr, size)

    def get_into_ptr(self, key: str, ptr: int, size: int) -> bool:
        return self._client.get_into_ptr(key, ptr, size)

    def batch_get_into_ptr(self, keys: list, ptrs: list, sizes: list) -> list:
        raw = self._client.batch_get_into_ptr(keys, ptrs, sizes)
        return [sz if hit else 0 for hit, sz in zip(raw, sizes)]

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

    def register_memory(self, ptr: int, size: int):
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

_BACKENDS = {"umbp": UMBPBackend, "mooncake": MooncakeBackend}


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        return _s.getsockname()[1]


# --- split writer/reader rendezvous (TCP, hosted on the writer node) ---


def _sync_recv_until(conn, token: bytes, timeout: float = 600.0):
    conn.settimeout(timeout)
    buf = b""
    while token not in buf:
        chunk = conn.recv(64)
        if not chunk:
            raise ConnectionError(f"sync peer closed before sending {token!r}")
        buf += chunk


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
if args.start_server and role != "reader":
    _resolved = _BACKENDS[backend_name].start_server(args)
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
if backend_name == "umbp":

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
            f"backend=umbp role={role} writer_node_id={node_id} reader_node_id={reader_node_id} "
            f"node_address={node_address} writer_port={peer_service_port} reader_port={reader_peer_port}",
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
        backing = (
            UMBPHostBufferBacking.AnonymousHugetlb
            if use_hp
            else UMBPHostBufferBacking.Anonymous
        )
        numa = int(os.environ.get("UMBP_DRAM_NUMA_NODE", numa_node))
        self._handle = HostBuffer._allocator.alloc(size, backing, hp_size, numa, True)
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
        handle = getattr(self, "_handle", None)
        if handle is not None and HostBuffer._allocator is not None:
            try:
                HostBuffer._allocator.free(handle)
            except Exception:
                pass
            self._handle = None


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
    reader = _build_reader()
else:  # both
    write_all()
    reader = writer.make_reader()

# --- commands (reader side) ---

exit_code = 0

if command == "correctness":
    read_buf = HostBuffer(value_size)
    read_ptr = read_buf.ptr
    reader.register_memory(read_ptr, value_size)
    hits = misses = mismatches = 0
    expected = expected_payload(value_size)
    for i in range(nr_objects):
        ctypes.memset(read_ptr, 0, value_size)
        if not reader.get_into_ptr(make_key(i), read_ptr, value_size):
            misses += 1
            continue
        hits += 1
        if ctypes.string_at(read_ptr, value_size) != expected:
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
    whole = HostBuffer(buf_total)
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

# In a split run, tell the writer we're finished so it can release the dataset
# and exit; the writer is blocked in _sync_recv_until(b"DONE") until this lands.
if sync_conn is not None:
    try:
        sync_conn.sendall(b"DONE\n")
        sync_conn.close()
    except OSError:
        pass

if exit_code:
    sys.exit(exit_code)
