#!/usr/bin/env bash
# ==========================================================================
# UT — cross-node AllGather bandwidth UNDER A CONCURRENT GEMM (overlap test).
# Measures, per message size, HierAllGather(SDMA) vs RCCL bandwidth both in
# isolation and while a GEMM runs on the compute units at the same time.
#
#   bash run_ut_overlap.sh            # world=16 (default)
#   WORLD=w8 bash run_ut_overlap.sh   # world=8
#
# Node pair + repo come from env (defaults below). Writes the raw log to
# ../results/ut_overlap_<world>.log .
# ==========================================================================
set -u
MASTER="${MASTER:-useocpm2m-097-040}"
WORKER="${WORKER:-useocpm2m-097-083}"
MASTER_IP="${MASTER_IP:-10.158.213.159}"
CTR="${CTR:-mori-sglang-mingzhi}"
WT="${MORI_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/results}"
IFACE="${IFACE:-eth0}"
IB="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
GID="${NCCL_IB_GID_INDEX:-3}"
WORLD="${WORLD:-w16}"
SIZES="${SIZES:-32 64 128 256}"

case "$WORLD" in
  w8)  NPROC=4; DEVS=0,1,2,3;         ENV="MORI_HIER_FUSE_LOCAL=1 MORI_HIER_FUSE_REMOTE=1 MORI_HIER_LOCAL_PUSHONLY=1 MORI_HIER_DEEP_PIPE=auto MORI_SDMA_NUM_CHANNELS=8 MORI_HIER_NIC_NUMA_LOCAL=1" ;;
  w16) NPROC=8; DEVS=0,1,2,3,4,5,6,7; ENV="MORI_HIER_CROWN=1 MORI_HIER_DEEP_PIPE=auto MORI_HIER_NIC_NUMA_LOCAL=1" ;;
  *) echo "WORLD must be w8 or w16"; exit 2 ;;
esac

FABRIC="GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE \
NCCL_IB_HCA=$IB NCCL_IB_GID_INDEX=$GID MORI_RDMA_DEVICES=$IB"
HEAP="MORI_SHMEM_HEAP_SIZE=34359738368"
PORT=$(( 29900 + RANDOM % 300 ))
BASE="export HIP_VISIBLE_DEVICES=$DEVS PYTHONPATH=$WT/python MORI_ENABLE_SDMA=1 $HEAP $FABRIC $ENV; cd $WT"
TR="torchrun --nnodes=2 --nproc_per_node=$NPROC --master_addr=$MASTER_IP --master_port=$PORT"
OV="tests/python/ccl/bench_gemm_overlap.py --sizes-mb $SIZES --dtypes bf16 --reps 5 --warmup 3"
mkdir -p "$OUT"; LOG="$OUT/ut_overlap_${WORLD}.log"

echo "== UT overlap $WORLD (AllGather under a concurrent GEMM, $(date -u +%T)) -> $LOG =="
for n in "$MASTER" "$WORKER"; do ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench_gemm_overlap; pkill -9 -f torchrun; true'" 2>/dev/null; done; sleep 2
ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$BASE && $TR --node_rank=1 $OV >/tmp/ovl_${WORLD}_w.log 2>&1'" &
wp=$!; sleep 4
ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$BASE && $TR --node_rank=0 $OV 2>&1'" | tee "$LOG"
wait "$wp" 2>/dev/null || true
echo "== done -> $LOG =="
