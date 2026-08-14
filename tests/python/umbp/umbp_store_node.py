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
"""Standalone UMBP storage-node launcher.

Joins a UMBP distributed pool as a pure storage/capacity node (DRAM/SSD)
WITHOUT starting an SGLang inference engine. It does this by constructing
sglang's own `UMBPStore` (the exact same HiCache L3 backend class an sglang
engine instantiates) with `mem_pool_host=None`, so this process:

  - reads the exact same env vars sglang reads (UMBP_MASTER_ADDRESS,
    UMBP_SSD_ENABLED, UMBP_SSD_DIR, UMBP_SSD_CAPACITY, UMBP_NODE_ID,
    UMBP_NODE_ADDRESS, UMBP_IO_ENGINE_HOST, UMBP_IO_ENGINE_PORT,
    UMBP_PEER_SERVICE_PORT, UMBP_SPDK_*, ...) via UMBPConfig.from_environment(),
  - accepts the exact same `--hicache-storage-backend-extra-config` JSON
    string / "@file.json|yaml|toml" syntax sglang's server_args.py takes,
  - registers with the UMBP master and starts the heartbeat exactly as a
    real sglang rank's UMBPStore would.

There is no model, no scheduler, no GPU work — it just allocates DRAM/SSD
capacity, registers it with the master, and idles.

Run this in the SAME container/image you use to launch the sglang engine
(needs sglang + mori built with BUILD_UMBP=ON importable).

Example (pure-SSD storage node, joining an existing UMBP master):

    UMBP_MASTER_ADDRESS=10.0.0.5:50051 \\
    UMBP_SSD_ENABLED=1 \\
    UMBP_SSD_DIR=/mnt/nvme0,/mnt/nvme1 \\
    UMBP_SSD_CAPACITY=1000000000000 \\
    UMBP_NODE_ID=ssd-node-1 \\
    UMBP_NODE_ADDRESS=10.0.0.9 \\
    python umbp_store_node.py

Or pass the same knobs via extra_config, exactly like an sglang launch
script's --hicache-storage-backend-extra-config:

    python umbp_store_node.py --hicache-storage-backend-extra-config \\
      '{"master_address": "10.0.0.5:50051", "node_id": "ssd-node-1", \\
        "node_address": "10.0.0.9", "ssd_enabled": true, \\
        "ssd_storage_dir": "/mnt/nvme0,/mnt/nvme1", \\
        "ssd_capacity_bytes": 1000000000000}'
"""

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
from typing import Optional

logger = logging.getLogger("umbp_store_node")

# Keys that sglang's HiRadixCache pops out of extra_config before it ever
# reaches UMBPStore (see hiradix_cache.py::_parse_storage_backend_extra_config).
# Stripped here too so a config blob copy-pasted from an sglang launch script
# doesn't trigger UMBPStore's "unknown extra_config key" warning.
_HIRADIX_ONLY_EXTRA_KEYS = (
    "prefetch_threshold",
    "prefetch_timeout_base",
    "prefetch_timeout_per_ki_token",
    "prefetch_timeout_max",
    "hicache_storage_pass_prefix_keys",
)


def _parse_extra_config(raw: Optional[str]) -> dict:
    """Same parsing as sglang's _parse_storage_backend_extra_config: a JSON
    string, or "@path" to a .json/.toml/.yaml file."""
    extra_config = {}
    if raw:
        if raw.startswith("@"):
            path = raw[1:]
            ext = os.path.splitext(path)[1].lower()
            with open(path, "rb" if ext == ".toml" else "r") as f:
                if ext == ".json":
                    extra_config = json.load(f)
                elif ext == ".toml":
                    import tomllib

                    extra_config = tomllib.load(f)
                elif ext in (".yaml", ".yml"):
                    import yaml

                    extra_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file {path} (format: {ext})")
        else:
            extra_config = json.loads(raw)
    for key in _HIRADIX_ONLY_EXTRA_KEYS:
        extra_config.pop(key, None)
    return extra_config


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Join a UMBP distributed pool as a storage-only node "
            "(no SGLang engine, no GPU)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hicache-storage-backend-extra-config",
        dest="extra_config_raw",
        default=None,
        help=(
            "Same flag name/format sglang uses: a JSON string, or '@path' to a "
            "json/toml/yaml file, with UMBPStore extra_config keys (e.g. "
            "master_address, node_id, node_address, ssd_enabled, "
            "ssd_storage_dir, ssd_capacity_bytes, ...). Anything not set here "
            "falls back to the matching UMBP_* env var, same as sglang."
        ),
    )
    # Rank/topology fields mirror sglang's HiCacheStorageConfig. A pure
    # storage node normally leaves these at their defaults (rank 0 / size 1);
    # they only matter if you intentionally run several store processes
    # per node and want them to shard SSD dirs / node_ids by rank the same
    # way sglang's per-rank UMBPStore instances do.
    p.add_argument("--tp-rank", type=int, default=0)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--pp-rank", type=int, default=0)
    p.add_argument("--pp-size", type=int, default=1)
    p.add_argument("--attn-cp-rank", type=int, default=0)
    p.add_argument("--attn-cp-size", type=int, default=1)
    p.add_argument("--is-mla-model", action="store_true")
    p.add_argument("--enable-storage-metrics", action="store_true")
    p.add_argument("--model-name", default=None)
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
        from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore
    except ImportError as exc:
        logger.error(
            "Could not import sglang's UMBPStore (%s). Run this inside the same "
            "container/image used to launch the sglang UMBP engine — it needs "
            "sglang and mori (built with BUILD_UMBP=ON) importable.",
            exc,
        )
        sys.exit(1)

    extra_config = _parse_extra_config(args.extra_config_raw)

    storage_config = HiCacheStorageConfig(
        tp_rank=args.tp_rank,
        tp_size=args.tp_size,
        pp_rank=args.pp_rank,
        pp_size=args.pp_size,
        attn_cp_rank=args.attn_cp_rank,
        attn_cp_size=args.attn_cp_size,
        is_mla_model=args.is_mla_model,
        enable_storage_metrics=args.enable_storage_metrics,
        is_page_first_layout=True,
        model_name=args.model_name,
        extra_config=extra_config,
    )

    logger.info(
        "Starting UMBP storage-only node on %s (no SGLang engine, no GPU). "
        "extra_config keys=%s",
        socket.gethostname(),
        sorted(extra_config.keys()),
    )

    store = UMBPStore(storage_config=storage_config, mem_pool_host=None)

    is_distributed = False
    try:
        is_distributed = bool(store.client.is_distributed())
    except Exception:
        pass
    if not is_distributed:
        logger.warning(
            "UMBP client did NOT start in distributed mode — no master_address "
            "was resolved from extra_config['master_address'] or the "
            "UMBP_MASTER_ADDRESS env var. This process is only a local/standalone "
            "pool, not a capacity contributor to a remote master."
        )
    else:
        logger.info(
            "UMBP storage node registered with master and heartbeating. "
            "Idling — press Ctrl-C to leave the pool and exit."
        )

    stop = False

    def _handle_signal(signum, _frame):
        nonlocal stop
        logger.info("Received signal %d, shutting down store node...", signum)
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while not stop:
        time.sleep(5)

    store.flush()
    logger.info("UMBP storage node stopped.")


if __name__ == "__main__":
    main()
