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
"""Passive pure-SSD UMBP storage peer.

Joins an existing UMBP distributed cluster (given master_address) as a
node with NO DRAM tier and an SSD tier spanning 2 local drives, then idles
forever. Does not originate any Put/Get itself -- it exists purely so the
master's routing table includes this node's SSD capacity, letting OTHER
UMBP clients (e.g. an sglang engine's per-TP-rank UMBPStore on another
node) route some keys here over RDMA.

See mori/src/umbp/doc/pure-ssd-mode.md: "every setting is also a plain
field on UMBPConfig / UMBPDistributedConfig for callers driving the client
directly, without SGLang."
"""
import os
import signal
import socket
import sys
import time

from mori.umbp import (
    UMBPClient,
    UMBPConfig,
    UMBPDistributedConfig,
    UMBPDurabilityConfig,
    UMBPDurabilityMode,
)

master_address = sys.argv[1]
storage_dirs = sys.argv[2]  # comma-separated, e.g. "/umbp_ssd/drive3,/umbp_ssd/drive4"
ssd_capacity_bytes = int(sys.argv[3]) if len(sys.argv) > 3 else 68719476736

node_id = f"ssd_worker_{socket.gethostname()}_{os.getpid()}_{int(time.time())}"

_master_host = master_address.split(":")[0]
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
    _s.connect((_master_host, 1))
    node_address = _s.getsockname()[0]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
    _s.bind(("", 0))
    peer_service_port = _s.getsockname()[1]

cfg = UMBPConfig()
cfg.dram.capacity_bytes = 0  # pure-SSD-mode switch

cfg.ssd.enabled = True
cfg.ssd.storage_dir = storage_dirs
cfg.ssd.capacity_bytes = ssd_capacity_bytes
cfg.ssd.ssd_backend = "file"
cfg.ssd.direct_io = True
cfg.ssd.verify_crc = False
cfg.ssd.tier_io_threads = 4
durability = UMBPDurabilityConfig()
durability.mode = UMBPDurabilityMode.Strict
cfg.ssd.durability = durability

dist = UMBPDistributedConfig()
dist.master_config.master_address = master_address
dist.master_config.node_id = node_id
dist.master_config.node_address = node_address
dist.peer_service_port = peer_service_port
dist.io_engine.host = node_address
dist.cache_remote_fetches = False
dist.ssd_staging_buffer_slots = int(sys.argv[4]) if len(sys.argv) > 4 else 512
dist.ssd_staging_buffer_size = int(sys.argv[5]) if len(sys.argv) > 5 else 4294967296
dist.ssd_write_staging_slots = int(sys.argv[4]) if len(sys.argv) > 4 else 512
dist.ssd_write_staging_size = int(sys.argv[5]) if len(sys.argv) > 5 else 4294967296
dist.ssd_staging_use_hugepages = True
dist.ssd_staging_hugepage_size = 2097152
cfg.distributed = dist

print(
    f"node_id={node_id} node_address={node_address} "
    f"peer_service_port={peer_service_port} storage_dir={storage_dirs} "
    f"ssd_capacity_bytes={ssd_capacity_bytes} master={master_address}",
    flush=True,
)

client = UMBPClient(cfg)
print(f"is_distributed={client.is_distributed()}", flush=True)
print("SSD_WORKER_READY", flush=True)


def _handle_clear(signum, frame):
    ok = client.clear()
    print(f"CLEAR_SIGNAL_RECEIVED ok={ok}", flush=True)


signal.signal(signal.SIGUSR1, _handle_clear)

while True:
    time.sleep(3600)
