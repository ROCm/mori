#!/usr/bin/env bash
# ==========================================================================
# One-key cross-node FSDP2 E2E benchmark (Qwen-7B, 2 nodes).
#
#   bash run_e2e.sh            # native (RCCL) baseline, world=8 AND 16
#   bash run_e2e.sh --mori     # mori SDMA HierAllGather, world=8 AND 16
#
# Node pair + repo are taken from env (defaults below); everything else
# (env vars, NIC list, model config, torchrun args) is fixed so every run
# is identical. Prints avg_tflops_per_gpu + last_loss per world.
# ==========================================================================
set -u

# ---- config (override via env) ----
MASTER="${MASTER:-useocpm2m-097-094}"
WORKER="${WORKER:-useocpm2m-097-115}"
MASTER_IP="${MASTER_IP:-10.158.213.63}"
CTR="${CTR:-mori-sglang-mingzhi}"
# repo root = three levels up from this script (examples/fsdp_sdma/scripts/..)
WT="${MORI_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
EX="$WT/examples/fsdp_sdma"
CFG="${QWEN_CFG:-$EX/scripts/qwen7b_vocab32000}"   # Qwen2 config dir, vocab_size=32000
IFACE="${IFACE:-eth0}"
IB="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
GID="${NCCL_IB_GID_INDEX:-3}"

MODE=native; [ "${1:-}" = "--mori" ] && MODE=mori
FABRIC="GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE \
NCCL_IB_HCA=$IB NCCL_IB_GID_INDEX=$GID MORI_RDMA_DEVICES=$IB"
ARGS="--model-name-or-path $CFG --seq-len 2048 --steps 20 --warmup 6 --micro-batch-size 1 --dtype bf16 --print-every 5"

run_world() {  # <w8|w16>
  local world="$1" nproc devs
  case "$world" in
    w8)  nproc=4; devs=0,1,2,3 ;;
    w16) nproc=8; devs=0,1,2,3,4,5,6,7 ;;
  esac
  # mori backend routing: MORI_FSDP_ENABLE_HIER=1 selects the HierAllGather adapter
  # (the plain MORI_ENABLE_SDMA default routes to the flat oneshot backend that
  # faults cross-node). w16 uses the adapter's rpn>=8 host-drain bit-exact base;
  # w8 uses the host-proxy transport (the device rpn=4 kernel needs a .so fix).
  local menv=""
  if [ "$MODE" = mori ]; then
    menv="MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1"
    [ "$world" = w8 ] && menv="$menv MORI_FSDP_HOST_PROXY=1 MORI_FSDP_HOSTPROXY_CAP_MB=512 MORI_SHMEM_HEAP_SIZE=17179869184"
  fi
  local port=$(( 29500 + RANDOM % 300 ))
  local base="export HIP_VISIBLE_DEVICES=$devs PYTHONPATH=$WT/python:$EX $FABRIC $menv; cd $EX"
  local tr="torchrun --nnodes=2 --nproc_per_node=$nproc --master_addr=$MASTER_IP --master_port=$port"
  echo "== E2E $world $MODE ($(date -u +%T)) =="
  for n in "$MASTER" "$WORKER"; do ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench.py; pkill -9 -f torchrun; true'" 2>/dev/null; done; sleep 2
  ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$base && $tr --node_rank=1 bench.py --mode $MODE $ARGS >/tmp/e2e_${world}_${MODE}_w.log 2>&1'" &
  local wp=$!; sleep 5
  ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$base && $tr --node_rank=0 bench.py --mode $MODE $ARGS 2>&1'" | grep -E 'avg_tflops_per_gpu|last_loss|Slow wait' | tail -4
  wait "$wp" 2>/dev/null || true
}

echo "### FSDP2 E2E — mode=$MODE — nodes $MASTER/$WORKER ###"
run_world w8
run_world w16
echo "### E2E done ###"
