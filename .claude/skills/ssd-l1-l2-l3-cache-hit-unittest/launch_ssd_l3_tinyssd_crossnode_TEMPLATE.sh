#!/bin/bash
# Engine launch template for 4nvme/8nvme cross-node topologies: tiny (1MB) local SSD
# capacity so this engine's own 8 per-TP-rank identities can never be capacity-selected
# as a put target -- all real L3 traffic goes to standalone umbp_ssd_worker.py processes
# instead (see SKILL.md "Standalone UMBP worker architecture").
#
# REPLACE before use:
#   <DRIVE1_DIR> <DRIVE2_DIR>  -- this engine node's own 2 local SSD mount paths
#                                 (inside the container, e.g. /umbp_ssd/drive1,/umbp_ssd/drive2)
#   ${UMBP_MASTER}             -- set by the caller as an env var before invoking this
#                                 script, e.g. export UMBP_MASTER=10.235.192.84:51051
#
# This is the EXACT env-var set validated across the whole cross-node investigation --
# every single var below was present in every launch that produced a clean result.
# Don't drop any of them without re-validating; several look unrelated to SSD/L3 at
# first glance (aiter/NCCL/CPU-affinity block) but were part of every working config.
set -x

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_LOG_MS=true
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=0
export SGLANG_HEALTH_CHECK_TIMEOUT=120
export SGLANG_TOOL_STRICT_LEVEL=1
export SGLANG_HUGEPAGE_SIZE=2MB
export SGLANG_HICACHE_BLOCK_QUOTA=16

export SGLANG_USE_AITER=1
export ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export SGLANG_AITER_MLA_PERSIST=1
export AITER_MXFP4_MOE_SF=1

export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export NCCL_SOCKET_IFNAME=enp81s0f1
export GLOO_SOCKET_IFNAME=enp81s0f1

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/sgl-workspace/aiter
export HF_HUB_CACHE=/hf_cache

export MORI_SHMEM_MODE=ISOLATION
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export SGLANG_MORI_DISPATCH_DTYPE=auto
export SGLANG_MORI_QP_PER_TRANSFER=4
export SGLANG_MORI_NUM_WORKERS=4
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=1024

export UMBP_DISABLE_ZERO_COPY_REGISTER=false
export UMBP_DRAM_USE_HUGEPAGES=1
export UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES=1
export MORI_SHMEM_HEAP_SIZE=1G
export MORI_IO_SQ_BACKOFF_TIMEOUT_US=15000000
export MORI_IO_QP_MAX_SEND_WR=32767
export MORI_IO_QP_MAX_CQE=65536
export MORI_IO_QP_MAX_SGE=4
export MORI_IO_TC_DISABLE=0
export MORI_RDMA_TC=96
export MORI_IO_TC=96
export MORI_RDMA_SL=3
export MORI_IO_SL=3
export MORI_UMBP_LOG_LEVEL=info
export HICACHE_UMBP=on
export UMBP_SSD_READ_LEASE_MS=30000
export UMBP_SSD_TIMING=1
export UMBP_SSD_DIRECT_IO=1
export UMBP_SSD_VERIFY_CRC=0
export UMBP_SSD_TIER_IO_THREADS=4
export UMBP_SSD_DURABILITY=strict
export SGLANG_MORI_FP8_COMB=false
export SGLANG_SET_CPU_AFFINITY=true
export SGLANG_CPUSET=0-255

NODE_IP="$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -1)"

python3 -m sglang.launch_server \
  --model-path /agent_ci/models/Kimi-K2.6-MXFP4 \
  --served-model-name Kimi-K2.6-MXFP4 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --context-length 262144 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.5 \
  --max-running-requests 1 \
  --max-total-tokens 148480 \
  --chunked-prefill-size 2048 \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --attention-backend aiter \
  --watchdog-timeout 3600 \
  --decode-log-interval 10 \
  --log-level info \
  --page-size 64 \
  --trust-remote-code \
  --enable-metrics \
  --enable-cache-report \
  --disable-custom-all-reduce \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --chat-template /agent_ci/models/Kimi-K2.6-MXFP4/chat_template.jinja \
  --nccl-port 42000 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.2 \
  --hicache-write-policy write_through \
  --hicache-mem-layout page_first \
  --hicache-io-backend kernel \
  --hicache-storage-backend mori \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"dram_capacity_bytes": 0, "ssd_enabled": true, "ssd_storage_dir": "<DRIVE1_DIR>,<DRIVE2_DIR>", "ssd_capacity_bytes": 1048576, "ssd_backend": "file", "ssd_io_backend": "io_uring", "master_address": "'${UMBP_MASTER}'", "node_address": "'${NODE_IP}'", "io_engine_port": "19600", "peer_service_port": "19700", "cache_remote_fetches": true, "kv_events_subscriber": true, "kv_events_endpoint": "tcp://localhost:6557", "staging_buffer_size": 2147483648, "ssd_staging_buffer_slots": 512, "ssd_staging_buffer_size": 4294967296, "ssd_write_staging_slots": 512, "ssd_write_staging_size": 4294967296, "ssd_staging_use_hugepages": true, "ssd_staging_hugepage_size": 2097152}' \
  --kv-events-config '{"publisher": "zmq", "endpoint": "tcp://*:6557"}'
