#!/usr/bin/env bash
# ==========================================================================
# E2E — cross-node FSDP2 training step (Qwen-7B, seq2048, bf16).
# Runs native (RCCL) + one mori variant and writes one log per backend.
# Compare last_loss (bit-exact) and avg_tflops_per_gpu. EVERY switch is baked
# in per variant, so there are NO env vars to set — it just runs.
#
#   bash run_e2e.sh              # native + mori         (host-proxy ASYNC, CPU-posted RDMA; ~1.10x, bit-exact) [default]
#   bash run_e2e.sh mori-ibgda   # native + mori-ibgda   (device IBGDA, GPU-initiated RDMA + deferred fence; ~1.06x, bit-exact)
#   WORLD=w8 bash run_e2e.sh     # world=8 (default w16)
#
# Node pair + repo from env (defaults below). Writes
# ../results/e2e_<world>_native.log and ../results/e2e_<world>_<variant>.log .
# ==========================================================================
set -u
VARIANT="${1:-mori}"
case "$VARIANT" in
  # host-proxy ASYNC: CPU worker posts the cross-node RDMA + runs the landing
  # fence at copy-out (inter leg CU-free AND hidden behind the backward GEMM).
  mori)       MORI_ENV="MORI_FSDP_HOST_PROXY=1 MORI_FSDP_HOSTPROXY_CAP_MB=512 MORI_SHMEM_HEAP_SIZE=17179869184 MORI_HOSTPROXY_ASYNC=1" ;;
  # device IBGDA: GPU threads post/poll the RDMA WQEs; deferred (non-inline)
  # host landing fence overlaps the backward GEMM (MORI_HIER_DEBUG_SYNC=0).
  mori-ibgda) MORI_ENV="MORI_HIER_DEBUG_SYNC=0" ;;
  *) echo "usage: bash run_e2e.sh [mori|mori-ibgda]   (default: mori)"; exit 2 ;;
esac
MASTER="${MASTER:-useocpm2m-097-040}"
WORKER="${WORKER:-useocpm2m-097-083}"
MASTER_IP="${MASTER_IP:-10.158.213.159}"
CTR="${CTR:-mori-sglang-mingzhi}"
WT="${MORI_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
EX="$WT/examples/fsdp_sdma"
CFG="${QWEN_CFG:-$(cd "$(dirname "$0")" && pwd)/qwen7b_vocab32000}"
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/results/mi300x_mlx5}"
IFACE="${IFACE:-eth0}"
IB="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
GID="${NCCL_IB_GID_INDEX:-3}"
WORLD="${WORLD:-w16}"
STEPS="${STEPS:-500}"

case "$WORLD" in
  w8)  NPROC=4; DEVS=0,1,2,3 ;;
  w16) NPROC=8; DEVS=0,1,2,3,4,5,6,7 ;;
  *) echo "WORLD must be w8 or w16"; exit 2 ;;
esac

FABRIC="GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE \
NCCL_IB_HCA=$IB NCCL_IB_GID_INDEX=$GID MORI_RDMA_DEVICES=$IB"
ARGS="--model-name-or-path $CFG --seq-len 2048 --steps $STEPS --warmup 6 --micro-batch-size 1 --dtype bf16 --print-every 5"
mkdir -p "$OUT"

run() {  # <native|mori> <logtag>
  local mode="$1" logtag="$2" menv="" port=$(( 29500 + RANDOM % 300 ))
  # mori mode: common routing (SDMA HierAllGather) + the variant switches from
  # $MORI_ENV (host-proxy ASYNC for `mori`, DEBUG_SYNC=0 deferred fence for
  # `mori-ibgda`). All baked in — nothing for the caller to export.
  [ "$mode" = mori ] && menv="MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 $MORI_ENV"
  local base="export HIP_VISIBLE_DEVICES=$DEVS PYTHONPATH=$WT/python:$EX $FABRIC $menv; cd $EX"
  local tr="torchrun --nnodes=2 --nproc_per_node=$NPROC --master_addr=$MASTER_IP --master_port=$port"
  local log="$OUT/e2e_${WORLD}_${logtag}.log"
  echo "== E2E $WORLD $logtag ($STEPS steps, $(date -u +%T)) menv='$menv' -> $log =="
  for n in "$MASTER" "$WORKER"; do ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench.py; pkill -9 -f torchrun; true'" 2>/dev/null; done; sleep 2
  ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$base && $tr --node_rank=1 bench.py --mode $mode $ARGS >/tmp/e2e_${WORLD}_${logtag}_w.log 2>&1'" &
  local wp=$!; sleep 5
  ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$base && $tr --node_rank=0 bench.py --mode $mode $ARGS 2>&1'" | tee "$log" | grep -E 'tflops_per_gpu|last_loss' | tail -3
  wait "$wp" 2>/dev/null || true
}

run native native
run mori "$VARIANT"
echo "== E2E done: $OUT/e2e_${WORLD}_{native,mori}.log =="
