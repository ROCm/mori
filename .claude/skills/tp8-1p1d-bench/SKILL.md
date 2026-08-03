---
name: tp8-1p1d-bench
description: >-
  Run TP8 1P1D DeepSeek-V4-Pro benchmark on the Spur MI350X cluster.
  Launches prefill, decode, and router containers, runs vllm bench serve
  at multiple concurrency levels, and collects logs. Use for baseline vs
  patched A/B comparisons of MORI IO optimizations.
---

# TP8 1P1D Benchmark Skill

## Cluster
- **Login:** vpolamre@134.199.197.117 (spur-login-atl)
- **Prefill node (fabric-1):** vpolamre@129.212.183.161
- **Decode node (fabric-2):** vpolamre@165.245.129.46
- SSH directly to nodes — no srun needed when Slurm allocation is active

## Images
- **Serve:** itej89/open-source:vllm_di_ci_dsvv4_serve1aadba4_routec04b24f33
- **Router:** itej89/open-source:vllm-router_feat_enable_remote_tp_size_be5aa9c

## Network
- RDMA NICs: ionic_0..ionic_7 (AINIC, 400GbE each)
- GID index: 1
- RDMA fabric IPs: 192.168.50.x (eth2)
- Control/rendezvous: eth0 (public IPs)
- Router port: 30000 on prefill node

## Logs
- Base dir: /home/tej/Documents/ws_mori_feat/logs/
- Each run gets: `tp8_1p1d_{baseline|patched}_{YYYYMMDD_HHMMSS}/`
- Files per run: commands.txt, prefill.log, decode.log, router.log, bench_serving_conc{1,8,32,64,128}.log

## Execution Checklist

### Phase 0: Cleanup
- [ ] Stop and remove any existing containers on both nodes:
  ```
  ssh vpolamre@129.212.183.161 'docker rm -f prefill proxy mori-bench 2>/dev/null'
  ssh vpolamre@165.245.129.46 'docker rm -f decode mori-bench 2>/dev/null'
  ```
- [ ] Verify no stale GPU processes: `ssh <node> 'fuser /dev/kfd 2>/dev/null'`
- [ ] Create timestamped log dir locally

### Phase 1: Start Servers
Order matters — router MUST start first (servers ping it on startup):

1. **Router (fabric-1)** — no GPU, starts instantly
2. **Prefill (fabric-1)**
   - For baseline: use stock image, no MORI rebuild
   - For patched: add MORI rebuild step + `-e MORI_IO_NUM_NICS_PER_TRANSFER=2`
   - Container name: `prefill`
   - Port: 20005
   - Log: `prefill.log`
   - Wait for: "Application startup complete" or model loaded message

2. **Decode (fabric-2)**
   - Same image/patching as prefill
   - Container name: `decode`
   - Port: 40005
   - Log: `decode.log`
   - Wait for: "Application startup complete"

3. **Router (fabric-1)**
   - Container name: `proxy`
   - Port: 30000
   - Log: `router.log`
   - No GPU needed

### Phase 2: Health Check
- [ ] Smoke test curl through router:
  ```
  curl http://129.212.183.161:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/data/models2/DeepSeek-V4-Pro","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":50,"temperature":0.7}'
  ```
- [ ] Verify response has valid completion text

### Phase 3: Benchmark Serving
Run from inside a container on either node (needs vllm installed):

```
for CONC in 1 8 32 64 128; do
  vllm bench serve \
    --backend openai-chat \
    --base-url http://129.212.183.161:30000 \
    --model /data/models2/DeepSeek-V4-Pro \
    --dataset-name random \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 256 \
    --max-concurrency $CONC \
    --request-rate inf \
    2>&1 | tee bench_serving_conc${CONC}.log
done
```

Key metrics to capture from each run:
- **Throughput:** requests/s, tokens/s
- **Latency:** mean TTFT (time to first token), mean TPOT (time per output token)
- **P99 latency**

### Phase 4: Collect Logs
- [ ] Copy prefill.log from fabric-1: `scp vpolamre@129.212.183.161:/path/prefill.log .`
- [ ] Copy decode.log from fabric-2: `scp vpolamre@165.245.129.46:/path/decode.log .`
- [ ] Copy bench logs
- [ ] Copy router.log

### Phase 5: Cleanup
- [ ] Stop all containers on both nodes
- [ ] Do NOT release Slurm allocation if running patched next

## Docker Run Template

### Common flags (both prefill and decode)
```
--user "$(id -u):$(id -g)"
--device /dev/dri --device /dev/kfd --device /dev/infiniband
--network host --ipc host
--group-add "$(getent group video | cut -d: -f3)"
--group-add "$(getent group render | cut -d: -f3)"
--cap-add SYS_PTRACE --cap-add IPC_LOCK
--security-opt seccomp=unconfined
--shm-size 64G
--ulimit nofile=1048576:1048576 --ulimit memlock=-1:-1
-v /data:/data
-e HOME=/data/vpolamre
-e USER="$(id -un)"
-e VLLM_ROCM_USE_AITER=1
-e TRITON_CACHE_DIR=/tmp/triton_cache
-e VLLM_CACHE_ROOT=/tmp/vllm_cache
-e VLLM_ENGINE_READY_TIMEOUT_S=36000
-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000
-e MORI_RDMA_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
-e MORI_IB_GID_INDEX=1
-e MORI_SHMEM_HEAP_SIZE=16G
-e MORI_GPU_ARCHS=gfx950
-e NCCL_IB_GID_INDEX=1
```

### Patched-only additions
```
-e CCACHE_DIR=/tmp/ccache
-e MORI_IO_NUM_NICS_PER_TRANSFER=2
```

### Patched entrypoint prefix (rebuild MORI before vllm serve)
```
export CCACHE_DIR=/tmp/ccache && mkdir -p /tmp/ccache &&
git clone --recurse-submodules -b feat/io-optimizations https://github.com/itej89/MORI.git /tmp/mori_build &&
cd /tmp/mori_build &&
BUILD_UMBP=OFF MORI_GPU_ARCHS=gfx950 pip install . --no-build-isolation &&
cd / &&
```

### vllm serve flags
```
vllm serve /data/models2/DeepSeek-V4-Pro \
  -tp 8 \
  --port {20005|40005} \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --kv-transfer-config "{...}"
```

### KV transfer config
- Prefill: `kv_role=kv_producer`, `proxy_ip=129.212.183.161`, `http_port=20005`, `handshake_port=6301`, `notify_port=6105`
- Decode: `kv_role=kv_consumer`, `proxy_ip=129.212.183.161`, `http_port=40005`, `handshake_port=7301`, `notify_port=7501`

## Troubleshooting
- **Model load hangs:** Check `/data/models2/DeepSeek-V4-Pro` exists on both nodes (shared NFS)
- **RDMA connection fails:** Verify `ibv_devinfo -d ionic_0` shows PORT_ACTIVE, check GID index 1
- **ccache permission denied:** Set `-e HOME=/data/vpolamre -e CCACHE_DIR=/tmp/ccache`
- **pip install fails with tail truncation:** Don't trust `tail -N` — grep for "Successfully installed" to confirm
- **Benchmark hangs:** Router may not have discovered both servers — check router.log for registered endpoints
