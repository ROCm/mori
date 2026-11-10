# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Verify pointer, batch and ranged KV I/O in a small local UMBP store."""

import argparse
import ctypes
import json
from pathlib import Path
import tempfile
import uuid


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def round_trip(umbp, config, mode):
    client = umbp.UMBPClient(config)
    allocator = umbp.UMBPHostMemAllocator()
    buffer = None
    try:
        backing = (
            umbp.UMBPHostBufferBacking.AnonymousShm
            if mode == "standalone"
            else umbp.UMBPHostBufferBacking.Anonymous
        )
        buffer = allocator.alloc(4096, backing)
        require(bool(buffer), "Host buffer allocation failed")
        ptr = buffer.ptr
        if mode == "standalone":
            require(
                client.register_memory(ptr, buffer.requested_size),
                "Shared host buffer registration failed",
            )

        namespace = f"demo/{uuid.uuid4().hex}"
        single_key = f"{namespace}/single"
        payload = bytes(range(256))
        ctypes.memmove(ptr, payload, len(payload))
        require(client.put_from_ptr(single_key, ptr, len(payload)), "PUT failed")
        require(client.exists(single_key), "Published key is missing")
        require(client.get_into_ptr(single_key, ptr + 1024, len(payload)), "GET failed")
        require(ctypes.string_at(ptr + 1024, len(payload)) == payload, "Byte mismatch")

        batch_keys = [f"{namespace}/batch/0", f"{namespace}/batch/1"]
        payloads = [b"K" * 256, b"V" * 256]
        for offset, value in zip((0, 256), payloads):
            ctypes.memmove(ptr + offset, value, len(value))
        require(
            client.batch_put_from_ptr(batch_keys, [ptr, ptr + 256], [256, 256])
            == [True, True],
            "Batch PUT failed",
        )
        require(
            client.batch_get_into_ptr(batch_keys, [ptr + 1024, ptr + 1280], [256, 256])
            == [True, True],
            "Batch GET failed",
        )
        for offset, value in zip((1024, 1280), payloads):
            require(
                ctypes.string_at(ptr + offset, len(value)) == value, "Batch mismatch"
            )

        range_key = f"{namespace}/ranges"
        ctypes.memmove(ptr, b"AAAABBBBCCCC", 12)
        # Inner lists describe ranges of ONE object; offsets are object-relative.
        require(
            client.batch_put_ranges_from_ptr(
                [range_key], [12], [[ptr + 4, ptr + 8, ptr]], [[4, 4, 4]], [[4, 8, 0]]
            )
            == [True],
            "Ranged PUT failed",
        )
        require(
            client.batch_get_ranges_into_ptr(
                [range_key], [[ptr + 1024, ptr + 1028]], [[4, 4]], [[0, 8]]
            )
            == [True],
            "Ranged GET failed",
        )
        require(ctypes.string_at(ptr + 1024, 8) == b"AAAACCCC", "Range mismatch")
        keys = [single_key, *batch_keys, range_key]
        require(
            client.batch_exists_consecutive([*keys, f"{namespace}/missing"]) == 4,
            "Consecutive-prefix lookup failed",
        )
        require(client.flush(), "Flush failed")
        return {
            "ok": True,
            "namespace": namespace,
            "objects_verified": len(keys),
            "checks": ["single", "batch", "ranges", "consecutive_prefix"],
            "deployment": str(client.get_deployment_mode()),
            "backend": str(client.get_backend_mode()),
            "is_distributed": client.is_distributed(),
        }
    finally:
        client.close()
        # A failed RPC can leave completion unknown. Retain such mappings until
        # this demo process exits; never reuse memory still visible to the server.
        if buffer is not None and not client.has_quarantined_transfers:
            allocator.free(buffer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("embedded", "standalone"), default="embedded"
    )
    parser.add_argument("--medium", choices=("dram", "ssd"), default="dram")
    parser.add_argument("--address", help="Existing standalone Unix socket address")
    parser.add_argument(
        "--ssd-parent", type=Path, help="Parent for a temporary SSD cache directory"
    )
    args = parser.parse_args()
    if args.mode == "standalone" and (
        not args.address or args.medium != "dram" or args.ssd_parent
    ):
        parser.error(
            "standalone requires --address; configure its storage at the server"
        )
    if args.mode == "embedded" and args.address:
        parser.error("--address is only used with --mode standalone")
    if args.ssd_parent and args.medium != "ssd":
        parser.error("--ssd-parent requires --medium ssd")

    # Import after argument parsing so --help also works without a Mori install.
    import mori.umbp as umbp

    config = umbp.UMBPConfig()
    config.dram.capacity_bytes = 4 * 1024 * 1024
    config.dram.use_hugepages = False
    config.ssd.enabled = False
    config.sim.enabled = False
    config.backend_policy_path = ""
    config.distributed = None
    config.standalone_process = None
    if args.mode == "standalone":
        process = umbp.UMBPStandaloneProcessConfig()
        process.address = args.address
        process.auto_start = False
        config.standalone_process = process
        print(json.dumps(round_trip(umbp, config, args.mode), indent=2))
    elif args.medium == "ssd":
        with tempfile.TemporaryDirectory(
            prefix="mori-umbp-demo-", dir=args.ssd_parent
        ) as path:
            config.ssd.enabled = True
            config.ssd.storage_dir = path
            config.ssd.capacity_bytes = 64 * 1024 * 1024
            config.ssd.segment_size_bytes = 8 * 1024 * 1024
            config.ssd.io.backend = umbp.UMBPIoBackend.Posix
            storage = umbp.UMBPDistributedConfig()
            storage.medium = umbp.UMBPMedium.SSD
            storage.io_engine.host = ""
            storage.peer_service_port = 0
            storage.dram_page_size = 64 * 1024
            storage.ranged_scratch_size = 0
            storage.ssd_staging_buffer_slots = 8
            config.distributed = storage
            print(json.dumps(round_trip(umbp, config, args.mode), indent=2))
    else:
        print(json.dumps(round_trip(umbp, config, args.mode), indent=2))


if __name__ == "__main__":
    main()
