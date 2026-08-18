#!/usr/bin/env bash
set -euo pipefail

cd /home/dasidler/mori

timestamp="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-bench_results/dispatch_sweep_${timestamp}}"
DTYPE="${DTYPE:-bf16}"
COMBINE_DTYPE="${COMBINE_DTYPE:-${DTYPE}}"
BLOCKS="${BLOCKS:-8 16 32 64}"
TOKENS="${TOKENS:-64 128 256 512 1024 2048 4096}"
WORLD_SIZE="${WORLD_SIZE:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-7168}"

mkdir -p "${RESULT_DIR}/raw"
printf 'dtype=%s\ncombine_dtype=%s\nblocks=%s\ntokens=%s\n' \
  "${DTYPE}" "${COMBINE_DTYPE}" "${BLOCKS}" "${TOKENS}" > "${RESULT_DIR}/metadata.txt"

for sdma in 0 1; do
  for blocks in ${BLOCKS}; do
    for tokens in ${TOKENS}; do
      log="${RESULT_DIR}/raw/sdma_${sdma}_blocks_${blocks}_tokens_${tokens}.log"
      if [[ -f "${log}" ]] && grep -qE 'Round 9 e2e|End-to-end result: skipped \(cross-type\)' "${log}"; then
        echo "SKIP complete sdma=${sdma} blocks=${blocks} tokens=${tokens}"
        continue
      fi

      echo "RUN sdma=${sdma} blocks=${blocks} tokens=${tokens} dtype=${DTYPE} combine=${COMBINE_DTYPE}"
      docker exec \
        -e MORI_ENABLE_SDMA="${sdma}" \
        -e ENABLE_PROFILER=0 \
        mori_dev_bench \
        bash -lc "cd /home/dasidler/mori && PYTHONPATH=/home/dasidler/mori/python:/home/dasidler/mori MORI_OPS_LOG_LEVEL=INFO MORI_SHMEM_HEAP_SIZE=6G timeout 360 python3 tests/python/ops/bench_dispatch_combine.py --cmd bench --world-size '${WORLD_SIZE}' --max-tokens '${tokens}' --dtype '${DTYPE}' --combine-dtype '${COMBINE_DTYPE}' --hidden-dim '${HIDDEN_DIM}' --zero-copy 1 --dispatch-block-num '${blocks}' --dispatch-warp-per-block 16 --combine-block-num '${blocks}' --combine-warp-per-block 4" \
        2>&1 | tee "${log}"
    done
  done
done
