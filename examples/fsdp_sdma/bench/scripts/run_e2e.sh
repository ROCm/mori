#!/usr/bin/env bash
# ==========================================================================
# E2E — cross-node FSDP2 training step (Qwen-7B, seq2048, bf16, world=16).
# Always runs the RCCL baseline + ONE mori variant, writes one raw log each,
# and prints last_loss (bit-exact vs RCCL) + avg_tflops_per_gpu. EVERY switch
# is baked in per variant — NO env vars to set, it just runs.
#
# The four configs (mori variant = intra-node leg x inter-node leg):
#   RCCL         baseline: native torch.distributed.all_gather_into_tensor.
#   hp_sdma      host-proxy (CPU-posted RDMA inter) + intra SDMA (XGMI copy
#                engine, CU-free). This PR's optimized path. ~1.20x, bit-exact. [default]
#   hp_cu        host-proxy inter + intra NCCL (CU). ~1.10x, bit-exact.
#   ibgda_sdma   device IBGDA (GPU-posted RDMA inter, deferred fence) + intra
#                SDMA. ~1.07x, bit-exact.
#
#   bash run_e2e.sh              # RCCL + hp_sdma   [default]
#   bash run_e2e.sh hp_cu        # RCCL + hp_cu
#   bash run_e2e.sh ibgda_sdma   # RCCL + ibgda_sdma
#   WORLD=w8 bash run_e2e.sh     # world=8 (default w16)
#
# Platform (node pair + NIC fabric) is a choice; default mi300x_mlx5:
#   PLATFORM=mi355x_ainic bash run_e2e.sh   # MI355X + AINIC (ionic) node pair & NICs
#
# Node pair + repo from env (platform defaults below; individual env still
# overrides). Writes raw logs to
# ../results/<platform>/raw/e2e_<world>_{RCCL,<variant>}.log .
# ==========================================================================
set -u
VARIANT="${1:-hp_sdma}"
# Common mori routing (SDMA HierAllGather + hierarchical) for every variant.
COMMON="MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 MORI_SHMEM_HEAP_SIZE=17179869184"
case "$VARIANT" in
  # host-proxy async (CPU-posted cross-node RDMA, deferred landing fence) + intra
  # SDMA copy engine (CU-free). The TWIN/EVENT_SYNC/NOSYNC perf levers are ON by
  # default in code, so only the SDMA-intra switch is needed here.
  hp_sdma)     MORI_ENV="MORI_FSDP_HOST_PROXY=1 MORI_FSDP_HOSTPROXY_CAP_MB=512 MORI_HOSTPROXY_ASYNC=1 MORI_HOSTPROXY_SDMA_INTRA=1" ;;
  # host-proxy async + intra NCCL (CU).
  hp_cu)       MORI_ENV="MORI_FSDP_HOST_PROXY=1 MORI_FSDP_HOSTPROXY_CAP_MB=512 MORI_HOSTPROXY_ASYNC=1" ;;
  # device IBGDA (GPU threads post/poll RDMA WQEs) + intra SDMA, deferred fence.
  ibgda_sdma)  MORI_ENV="MORI_HIER_DEBUG_SYNC=0" ;;
  *) echo "usage: bash run_e2e.sh [hp_sdma|hp_cu|ibgda_sdma]   (default: hp_sdma)"; exit 2 ;;
esac
# Platform = node pair + NIC fabric. PLATFORM=mi300x_mlx5 (default) or mi355x_ainic.
# Any of MASTER/WORKER/MASTER_IP/IFACE/NCCL_IB_GID_INDEX/MORI_RDMA_DEVICES still
# override individually; the platform only fills the ones left unset.
PLATFORM="${PLATFORM:-mi300x_mlx5}"
case "$PLATFORM" in
  mi300x_mlx5)   # MI300X + Mellanox mlx5 (RoCEv2 on GID 3)
    : "${MASTER:=<master>}"; : "${WORKER:=<worker>}"; : "${MASTER_IP:=<master-ip>}"
    : "${IFACE:=eth0}"; : "${NCCL_IB_GID_INDEX:=3}"
    : "${MORI_RDMA_DEVICES:=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}" ;;
  mi355x_ainic)  # MI355X + AINIC (ionic RoCEv2 on GID 1)
    : "${MASTER:=<ionic-master>}"
    : "${WORKER:=<ionic-worker>}"; : "${MASTER_IP:=<ionic-master-ip>}"
    : "${IFACE:=enp81s0f1}"; : "${NCCL_IB_GID_INDEX:=1}"
    : "${MORI_RDMA_DEVICES:=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}" ;;
  *) echo "PLATFORM must be mi300x_mlx5 or mi355x_ainic"; exit 2 ;;
esac
CTR="${CTR:-<container>}"
WT="${MORI_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
EX="$WT/examples/fsdp_sdma"
CFG="${QWEN_CFG:-$(cd "$(dirname "$0")" && pwd)/qwen7b_vocab32000}"
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/results/$PLATFORM/raw}"
IFACE="$IFACE"
IB="$MORI_RDMA_DEVICES"
GID="$NCCL_IB_GID_INDEX"
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

run() {  # <native|mori> <logtag> <mori_env>
  local mode="$1" logtag="$2" menv="$3" port=$(( 29500 + RANDOM % 300 ))
  local base="export HIP_VISIBLE_DEVICES=$DEVS PYTHONPATH=$WT/python:$EX $FABRIC $menv; cd $EX"
  local tr="torchrun --nnodes=2 --nproc_per_node=$NPROC --master_addr=$MASTER_IP --master_port=$port"
  local log="$OUT/e2e_${WORLD}_${logtag}.log"
  echo "== E2E $WORLD $logtag ($STEPS steps, $(date -u +%T)) -> $log =="
  for n in "$MASTER" "$WORKER"; do ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench.py; pkill -9 -f torchrun; true'" 2>/dev/null; done; sleep 3
  ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$base && $tr --node_rank=1 bench.py --mode $mode $ARGS >/tmp/e2e_${WORLD}_${logtag}_w.log 2>&1'" &
  local wp=$!; sleep 5
  ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$base && $tr --node_rank=0 bench.py --mode $mode $ARGS 2>&1'" | tee "$log" | grep -E 'tflops_per_gpu|last_loss' | tail -3
  wait "$wp" 2>/dev/null || true
}

run native RCCL ""
run mori "$VARIANT" "$COMMON $MORI_ENV"
echo "== E2E done: $OUT/e2e_${WORLD}_{RCCL,$VARIANT}.log =="
