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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Local SSD tier: ranged (sub-object) GET vs whole-object GET.

The shape mirrors what the sglang tree connector issues.  One stored object is
one KV page across every layer; a layer-wise load asks only for the byte slices
belonging to one group of layers, each into its own destination buffer.  The
question this answers is the one that decides whether ranged SSD is worth
having: to obtain G of L layers, is it cheaper to read the whole object and
throw most of it away, or to read just those slices?

Destinations are laid out layer-major on purpose (buffer[layer][object]), so an
object's slices land far apart in memory -- the tree connector keeps one buffer
per layer, and a destination-contiguous layout would flatter the ranged path.

Ranged reads never verify record checksums (a record CRC covers the whole
value), so --verify-crc is swept rather than fixed: it separates "ranged moved
fewer bytes" from "ranged skipped the checksum".

Example:
    python3 umbp_ranged_bench.py --ssd-dir /mnt/nvme0/umbp_ranged \\
        --layers 61 --layer-bytes 36864 --objects 256 --groups 1 4 8 16 61
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import statistics
import sys
import time

import mori.umbp as umbp


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PiB"


def build_config(args, verify_crc: bool):
    cfg = umbp.UMBPConfig()
    # DRAM must be too small to hold the dataset, otherwise Put keeps everything
    # in the fast tier and the SSD is never read.  A floor keeps room for the
    # demote machinery itself.
    cfg.dram.capacity_bytes = args.dram_bytes
    cfg.ssd.enabled = True
    cfg.ssd.storage_dir = args.ssd_dir
    cfg.ssd.capacity_bytes = args.ssd_capacity_bytes
    cfg.ssd.segment_size_bytes = args.segment_bytes
    cfg.ssd.direct_io = args.direct_io
    cfg.ssd.verify_crc = verify_crc
    cfg.ssd.tier_io_threads = args.tier_io_threads
    # The posix driver reports batch_read=false, so both paths degrade to one
    # blocking pread at a time (queue depth 1) -- which penalises the many-small-
    # reads shape ranged I/O produces.  io_uring is the production setting and
    # the only one where a batch is actually submitted as a batch.
    cfg.ssd.io.backend = (
        umbp.UMBPIoBackend.IoUring
        if args.io_backend == "io_uring"
        else umbp.UMBPIoBackend.Posix
    )
    cfg.ssd.io.queue_depth = args.queue_depth
    # Promotion would pull an object into DRAM on its first read, so every later
    # measurement would be a DRAM hit rather than an SSD one.
    cfg.eviction.auto_promote_on_read = False
    return cfg


class HostBuffer:
    """Page-aligned host allocation, so O_DIRECT can read straight into it."""

    def __init__(self, size: int):
        self.size = size
        self._raw = ctypes.create_string_buffer(size + 4096)
        base = ctypes.addressof(self._raw)
        self.ptr = (base + 4095) & ~4095

    def fill(self, byte: int) -> None:
        ctypes.memset(self.ptr, byte, self.size)


def seed(client, args, object_size: int) -> list[str]:
    src = HostBuffer(object_size)
    # A pattern that varies with offset, so a mis-addressed range is visible.
    blob = bytes((i * 31) & 0xFF for i in range(min(object_size, 1 << 20)))
    for off in range(0, object_size, len(blob)):
        n = min(len(blob), object_size - off)
        ctypes.memmove(src.ptr + off, blob[:n], n)

    keys = [f"{args.key_prefix}_obj_{i}" for i in range(args.objects)]
    t0 = time.perf_counter()
    ok = client.batch_put_from_ptr(
        keys, [src.ptr] * len(keys), [object_size] * len(keys)
    )
    elapsed = time.perf_counter() - t0
    if not all(ok):
        sys.exit(f"seed failed: {sum(1 for x in ok if x)}/{len(ok)} objects written")
    total = object_size * len(keys)
    print(
        f"  seeded {len(keys)} x {human_bytes(object_size)} = {human_bytes(total)} "
        f"in {elapsed:.2f}s ({total / elapsed / 1e9:.2f} GB/s)"
    )
    return keys


def time_calls(fn, passes: int) -> list[float]:
    samples = []
    for _ in range(passes):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def run_whole_object(client, keys, object_size, passes):
    dst = HostBuffer(object_size * len(keys))
    ptrs = [dst.ptr + i * object_size for i in range(len(keys))]
    sizes = [object_size] * len(keys)

    def once():
        results = client.batch_get_into_ptr(keys, ptrs, sizes)
        if not all(results):
            raise RuntimeError(
                f"whole-object GET missed {sum(1 for r in results if not r)}/{len(results)}"
            )

    once()  # warm the index, not the page cache (direct I/O bypasses it)
    return time_calls(once, passes)


def run_ranged(client, keys, args, group, layer_bytes, passes):
    """One call fetching `group` layers of every object."""
    n = len(keys)
    # Layer-major destinations: slice (layer, object) sits at its own address.
    dst = HostBuffer(group * n * layer_bytes)
    ptrs, sizes, offsets = [], [], []
    for obj in range(n):
        obj_ptrs, obj_sizes, obj_offsets = [], [], []
        for g in range(group):
            layer = args.first_layer + g
            obj_ptrs.append(dst.ptr + (g * n + obj) * layer_bytes)
            obj_sizes.append(layer_bytes)
            obj_offsets.append(layer * layer_bytes)
        ptrs.append(obj_ptrs)
        sizes.append(obj_sizes)
        offsets.append(obj_offsets)

    def once():
        results = client.batch_get_ranges_into_ptr(keys, ptrs, sizes, offsets)
        if not all(results):
            raise RuntimeError(
                f"ranged GET missed {sum(1 for r in results if not r)}/{len(results)}"
            )

    once()
    return time_calls(once, passes)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ssd-dir",
        required=True,
        help="Directory on the device under test. Comma-separate for a "
        "sharded multi-drive tier.",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=61,
        help="Layers per stored object (default: 61, DSv3/Kimi-shaped).",
    )
    p.add_argument(
        "--layer-bytes",
        type=int,
        default=36864,
        help="Bytes of one layer of one page (default: 36864 = 9x4KiB, "
        "MLA page_size=64 at 576B/token).",
    )
    p.add_argument(
        "--objects", type=int, default=256, help="Pages in the batch (default: 256)."
    )
    p.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Layer-group widths to sweep (default: 1 2 4 8 16 32). The "
        "full layer count is always added as the whole-object-via-"
        "ranges case.",
    )
    p.add_argument(
        "--first-layer",
        type=int,
        default=0,
        help="First layer of the group read (default: 0).",
    )
    p.add_argument("--passes", type=int, default=7)
    p.add_argument("--tier-io-threads", type=int, default=8)
    p.add_argument(
        "--io-backend",
        choices=["io_uring", "posix"],
        default="io_uring",
        help="SSD I/O driver (default: io_uring). posix has "
        "batch_read=false, so every read is a separate blocking "
        "pread -- useful as a contrast, not as the headline number.",
    )
    p.add_argument("--queue-depth", type=int, default=256)
    p.add_argument(
        "--direct-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="O_DIRECT (default: on). Off measures the page cache, not " "the device.",
    )
    p.add_argument(
        "--verify-crc",
        choices=["on", "off", "both"],
        default="both",
        help="Whole-object checksum verification (default: both). Ranged "
        "reads never verify, so 'both' separates the byte saving "
        "from the checksum saving.",
    )
    p.add_argument(
        "--dram-bytes",
        type=int,
        default=64 << 20,
        help="DRAM tier capacity (default: 64MiB). Must be far smaller "
        "than the dataset so objects demote to SSD.",
    )
    p.add_argument("--ssd-capacity-bytes", type=int, default=64 << 30)
    p.add_argument("--segment-bytes", type=int, default=1 << 30)
    p.add_argument("--key-prefix", default=f"rb{os.getpid()}")
    p.add_argument(
        "--keep", action="store_true", help="Leave the SSD directory behind."
    )
    args = p.parse_args()

    object_size = args.layers * args.layer_bytes
    dataset = object_size * args.objects
    groups = sorted(set(g for g in args.groups if 0 < g <= args.layers) | {args.layers})

    print(
        f"object={human_bytes(object_size)} ({args.layers} layers x "
        f"{human_bytes(args.layer_bytes)})  objects={args.objects}  "
        f"dataset={human_bytes(dataset)}"
    )
    print(
        f"ssd_dir={args.ssd_dir}  direct_io={args.direct_io}  "
        f"io_backend={args.io_backend}  qd={args.queue_depth}  "
        f"tier_io_threads={args.tier_io_threads}  passes={args.passes}"
    )
    if dataset < 4 * args.dram_bytes:
        print(
            f"  WARNING: dataset {human_bytes(dataset)} is close to the DRAM tier "
            f"({human_bytes(args.dram_bytes)}); some reads may be DRAM hits."
        )

    crc_modes = {"on": [True], "off": [False], "both": [True, False]}[args.verify_crc]

    for verify_crc in crc_modes:
        first_dir = args.ssd_dir.split(",")[0]
        shutil.rmtree(first_dir, ignore_errors=True)
        for d in args.ssd_dir.split(","):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        print(
            f"\n=== verify_crc={'on' if verify_crc else 'off'} "
            f"(whole-object only; ranged never verifies) ==="
        )
        client = umbp.UMBPClient(build_config(args, verify_crc))
        if not client.supports_ranged_io():
            sys.exit(
                "client reports supports_ranged_io()=False -- "
                "ranged SSD support is not in this build"
            )
        keys = seed(client, args, object_size)

        whole = run_whole_object(client, keys, object_size, args.passes)
        whole_med = statistics.median(whole)
        print(
            f"\n  whole-object GET: {whole_med * 1e3:8.2f} ms  "
            f"{dataset / whole_med / 1e9:6.2f} GB/s  (all {args.layers} layers)"
        )

        print(
            f"\n  {'group':>6} {'fetched':>9} {'ms':>9} {'GB/s':>7} "
            f"{'vs whole':>9}  {'verdict':<8}"
        )
        for group in groups:
            useful = group * args.layer_bytes * args.objects
            samples = run_ranged(
                client, keys, args, group, args.layer_bytes, args.passes
            )
            med = statistics.median(samples)
            frac = group / args.layers
            speedup = whole_med / med
            verdict = "win" if speedup > 1.0 else "lose"
            print(
                f"  {group:>6} {frac * 100:>8.1f}% {med * 1e3:>9.2f} "
                f"{useful / med / 1e9:>7.2f} {speedup:>8.2f}x  {verdict:<8}"
            )

        client.clear()
        del client

    if not args.keep:
        for d in args.ssd_dir.split(","):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
