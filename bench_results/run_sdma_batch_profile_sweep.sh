#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_sdma_batch_profile_sweep.sh

Configuration is supplied through environment variables:
  RESULT_DIR, DTYPE, COMBINE_DTYPE, BLOCKS, TOKENS, WORLD_SIZE,
  HIDDEN_DIM, DISPATCH_WARPS, COMBINE_WARPS, CAPTURE_ITERS, CLEAR_JIT
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

timestamp="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-bench_results/sdma_batch_profile_${timestamp}}"
DTYPE="${DTYPE:-fp8_e4m3_fnuz}"
COMBINE_DTYPE="${COMBINE_DTYPE:-bf16}"
BLOCKS="${BLOCKS:-8 16 32 64}"
TOKENS="${TOKENS:-64 128 256 512 1024 2048 4096}"
WORLD_SIZE="${WORLD_SIZE:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-7168}"
DISPATCH_WARPS="${DISPATCH_WARPS:-16}"
COMBINE_WARPS="${COMBINE_WARPS:-4}"
CAPTURE_ITERS="${CAPTURE_ITERS:-3}"
CLEAR_JIT="${CLEAR_JIT:-1}"

mkdir -p "${RESULT_DIR}"
printf 'dtype=%s\ncombine_dtype=%s\nblocks=%s\ntokens=%s\nworld_size=%s\nhidden_dim=%s\ndispatch_warps=%s\ncapture_iters=%s\n' \
  "${DTYPE}" "${COMBINE_DTYPE}" "${BLOCKS}" "${TOKENS}" "${WORLD_SIZE}" \
  "${HIDDEN_DIM}" "${DISPATCH_WARPS}" "${CAPTURE_ITERS}" > "${RESULT_DIR}/metadata.txt"

if [[ "${CLEAR_JIT}" == "1" ]]; then
  docker exec mori_dev_bench bash -lc 'rm -rf /root/.mori/jit'
fi

for blocks in ${BLOCKS}; do
  for tokens in ${TOKENS}; do
    case_dir="${RESULT_DIR}/blocks_${blocks}_tokens_${tokens}"
    mkdir -p "${case_dir}"
    if [[ -f "${case_dir}/trace_rank$((WORLD_SIZE - 1)).json" ]]; then
      echo "SKIP complete blocks=${blocks} tokens=${tokens}"
      continue
    fi

    log="${case_dir}/profile.log"
    echo "RUN blocks=${blocks} tokens=${tokens} dtype=${DTYPE}"
    docker exec \
      -e MORI_ENABLE_SDMA=1 \
      -e ENABLE_PROFILER=1 \
      -e MORI_PROFILE_CAPTURE_ITERS="${CAPTURE_ITERS}" \
      mori_dev_bench \
      bash -lc "cd /home/dasidler/mori && PYTHONPATH=/home/dasidler/mori/python:/home/dasidler/mori MORI_OPS_LOG_LEVEL=INFO MORI_SHMEM_HEAP_SIZE=6G timeout 360 python3 tests/python/ops/bench_dispatch_combine.py --cmd profile --world-size '${WORLD_SIZE}' --max-tokens '${tokens}' --dtype '${DTYPE}' --combine-dtype '${COMBINE_DTYPE}' --hidden-dim '${HIDDEN_DIM}' --zero-copy 1 --dispatch-block-num '${blocks}' --dispatch-warp-per-block '${DISPATCH_WARPS}' --combine-block-num '${blocks}' --combine-warp-per-block '${COMBINE_WARPS}'" \
      2>&1 | tee "${log}"

    stamp="$(sed -n 's/.*trace_intranode_rank0_\([0-9_]*\)\.json.*/\1/p' "${log}" | tail -1)"
    if [[ -z "${stamp}" ]]; then
      echo "ERROR: no trace timestamp found for blocks=${blocks} tokens=${tokens}" >&2
      exit 1
    fi

    for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
      src="trace_intranode_rank${rank}_${stamp}.json"
      if [[ ! -f "${src}" ]]; then
        echo "ERROR: missing ${src}" >&2
        exit 1
      fi
      mv "${src}" "${case_dir}/trace_rank${rank}.json"
    done
  done
done

echo "Profiles written to ${RESULT_DIR}"
echo "Plot with: docker exec mori_dev_bench bash -lc 'cd /home/dasidler/mori && python3 bench_results/plot_sdma_batch_occupancy.py --input ${RESULT_DIR}'"
