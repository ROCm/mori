#!/usr/bin/env bash
# ==========================================================================
# E2E — cross-node FSDP2 training step (Qwen-7B, seq2048, bf16).
# Runs native (RCCL) and mori (SDMA HierAllGather, deferred landing fence,
# no inline host-drain) and writes one log per backend. Compare last_loss
# (bit-exact) and avg_tflops_per_gpu.
#
#   bash run_e2e.sh            # world=16 (default), both native + mori
#   WORLD=w8 bash run_e2e.sh   # world=8
#
# Node pair + repo from env (defaults below). Writes
# ../results/e2e_<world>_native.log and ../results/e2e_<world>_mori.log .
# ==========================================================================
set -u
MASTER="${MASTER:-useocpm2m-097-040}"
WORKER="${WORKER:-useocpm2m-097-083}"
MASTER_IP="${MASTER_IP:-10.158.213.159}"
CTR="${CTR:-mori-sglang-mingzhi}"
WT="${MORI_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
EX="$WT/examples/fsdp_sdma"
CFG="${QWEN_CFG:-$(cd "$(dirname "$0")" && pwd)/qwen7b_vocab32000}"
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/results}"
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

run() {  # <native|mori>
  local mode="$1" menv="" port=$(( 29500 + RANDOM % 300 ))
  # mori: CPU-posted host-proxy transport with deferred-completion overlap
  # (ASYNC posts the cross-node write + step-1 gather now and runs the landing
  # fence at copy-out, so the inter leg is CU-free AND hidden behind the backward
  # GEMM). ASYNC auto-enables the double-buffered recv staging + landing drain
  # needed to stay bit-exact. w16: 269 TFLOPS/gpu = 1.07x native, per-window loss
  # bit-identical to native (4 reps). Bulk bytes stay on RDMA/SDMA (no CU copy).
  [ "$mode" = mori ] && menv="MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 MORI_FSDP_HOST_PROXY=1 MORI_FSDP_HOSTPROXY_CAP_MB=512 MORI_SHMEM_HEAP_SIZE=17179869184 MORI_HOSTPROXY_ASYNC=1"
  local base="export HIP_VISIBLE_DEVICES=$DEVS PYTHONPATH=$WT/python:$EX $FABRIC $menv; cd $EX"
  local tr="torchrun --nnodes=2 --nproc_per_node=$NPROC --master_addr=$MASTER_IP --master_port=$port"
  local log="$OUT/e2e_${WORLD}_${mode}.log"
  echo "== E2E $WORLD $mode ($STEPS steps, $(date -u +%T)) -> $log =="
  for n in "$MASTER" "$WORKER"; do ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench.py; pkill -9 -f torchrun; true'" 2>/dev/null; done; sleep 2
  ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$base && $tr --node_rank=1 bench.py --mode $mode $ARGS >/tmp/e2e_${WORLD}_${mode}_w.log 2>&1'" &
  local wp=$!; sleep 5
  ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$base && $tr --node_rank=0 bench.py --mode $mode $ARGS 2>&1'" | tee "$log" | grep -E 'tflops_per_gpu|last_loss' | tail -3
  wait "$wp" 2>/dev/null || true
}

run native
run mori
echo "== E2E done: $OUT/e2e_${WORLD}_{native,mori}.log =="
