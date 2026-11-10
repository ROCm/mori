#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?expected hicache or umbp}
MODEL_PATH=${MODEL_PATH:-/model/DeepSeek-V4-Pro}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Pro}
PORT=${PORT:-18000}
NCCL_PORT=${NCCL_PORT:-19500}
GPU_KV_TOKENS=${GPU_KV_TOKENS:-65536}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-512}

unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export AITER_BF16_FP8_MOE_BOUND=0
export GPU_MAX_HW_QUEUES=2
export HSA_NO_SCRATCH_RECLAIM=0
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_GROUPED_ASM=1
export SGLANG_DSV4_REASONING_EFFORT=high
export SGLANG_DSV4_UNIFIED_KV_FP8=0
export SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_FP8_WO_A_FUSED_INVROPE=1
export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1
export SGLANG_USE_ROCM700A=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export SGLANG_ENABLE_WAR_BARRIER=1

case "$MODE" in
  umbp)
    export UMBP_STANDALONE_ADDRESS=${UMBP_E2E_ADDRESS:?run through run.py to start a private standalone}
    EXTRA=(--enable-unified-cache-external-linker --unified-cache-external-linker-backend mori
           --hicache-storage-backend-extra-config '{"ssd_enabled":false,"standalone_auto_start":false}')
    ;;
  hicache) EXTRA=(--enable-hierarchical-cache) ;;
  *) printf 'Unknown mode: %s\n' "$MODE" >&2; exit 2 ;;
esac

COMMAND=(python3 -m sglang.launch_server
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code --host 127.0.0.1 --port "$PORT" --nccl-port "$NCCL_PORT"
  --tensor-parallel-size 4 --data-parallel-size 1 --expert-parallel-size 1
  --attention-backend dsv4 --dtype bfloat16 --kv-cache-dtype bfloat16
  --page-size 256 --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --prefill-decode-interval 20 --enforce-shared-experts-fusion
  --hicache-host-memory-mode cache --hicache-ratio 58.7890625
  --hicache-write-policy write_through --hicache-mem-layout page_first --hicache-io-backend kernel
  "${EXTRA[@]}" --enable-cache-report
  --swa-full-tokens-ratio 0.1 --max-running-requests 130
  --disable-cuda-graph --cuda-graph-max-bs-decode 128
  --max-total-tokens "$GPU_KV_TOKENS" --random-seed 352644285 --mem-fraction-static 0.82
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4
  --watchdog-timeout 240 --enable-metrics)

if [[ ${2:-} == --dry-run ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  exec "${COMMAND[@]}"
fi
