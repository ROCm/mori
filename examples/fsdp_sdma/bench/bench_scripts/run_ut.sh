#!/usr/bin/env bash
# ==========================================================================
# One-key standalone cross-node AllGather bandwidth UT (2 nodes).
# Each run sweeps sizes and reports, per size, HierAllGather (SDMA) GB/s vs
# RCCL GB/s + ratio + bit-exact, for world=8 AND world=16.
#
#   bash run_ut.sh             # AllGather bandwidth sweep, world=8 AND 16
#   bash run_ut.sh --mori      # same (the sweep always compares mori vs RCCL)
#
# Node pair + repo from env (defaults below); per-world config + sizes fixed.
# ==========================================================================
set -u

MASTER="${MASTER:-useocpm2m-097-094}"
WORKER="${WORKER:-useocpm2m-097-115}"
MASTER_IP="${MASTER_IP:-10.158.213.63}"
CTR="${CTR:-mori-sglang-mingzhi}"
WT="${MORI_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
OUT="${OUT:-$WT/examples/fsdp_sdma/bench/bench_results}"
IFACE="${IFACE:-eth0}"
IB="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
GID="${NCCL_IB_GID_INDEX:-3}"
: "${1:-}"  # --mori accepted (the sweep always measures mori vs RCCL)

FABRIC="GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE \
NCCL_IB_HCA=$IB NCCL_IB_GID_INDEX=$GID MORI_RDMA_DEVICES=$IB"
HEAP="MORI_SHMEM_HEAP_SIZE=34359738368"
# per-world shipped config (the exact env the adapter selects per topology)
W8_ENV="MORI_HIER_FUSE_LOCAL=1 MORI_HIER_FUSE_REMOTE=1 MORI_HIER_LOCAL_PUSHONLY=1 MORI_HIER_DEEP_PIPE=auto MORI_SDMA_NUM_CHANNELS=8 MORI_HIER_NIC_NUMA_LOCAL=1"
W16_ENV="MORI_HIER_CROWN=1 MORI_HIER_DEEP_PIPE=auto MORI_HIER_NIC_NUMA_LOCAL=1"

run_world() {  # <w8|w16> <nproc> <devs> <sizes> <env>
  local world="$1" nproc="$2" devs="$3" sizes="$4" env="$5"
  local port=$(( 29700 + RANDOM % 300 ))
  local base="export HIP_VISIBLE_DEVICES=$devs PYTHONPATH=$WT/python MORI_ENABLE_SDMA=1 $HEAP $FABRIC $env; cd $WT"
  local tr="torchrun --nnodes=2 --nproc_per_node=$nproc --master_addr=$MASTER_IP --master_port=$port"
  local sw="tests/python/ccl/bench_sweep.py --sizes-mb $sizes --dtypes fp32 bf16 --reps 4 --warmup 3"
  echo "== UT $world ($(date -u +%T)) =="
  for n in "$MASTER" "$WORKER"; do ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench_sweep; pkill -9 -f torchrun; true'" 2>/dev/null; done; sleep 2
  ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$base && $tr --node_rank=1 $sw >/tmp/ut_${world}_w.log 2>&1'" &
  local wp=$!; sleep 4
  ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$base && $tr --node_rank=0 $sw 2>&1'" | grep -E '\[sweep\]' | tail -18
  wait "$wp" 2>/dev/null || true
}

run_overlap() {  # <w8|w16> <nproc> <devs> <env>
  local world="$1" nproc="$2" devs="$3" env="$4"
  local port=$(( 29900 + RANDOM % 300 ))
  local base="export HIP_VISIBLE_DEVICES=$devs PYTHONPATH=$WT/python MORI_ENABLE_SDMA=1 $HEAP $FABRIC $env; cd $WT"
  local tr="torchrun --nnodes=2 --nproc_per_node=$nproc --master_addr=$MASTER_IP --master_port=$port"
  local ov="tests/python/ccl/bench_gemm_overlap.py --sizes-mb 32 64 128 256 512 --dtypes bf16 --reps 5 --warmup 3"
  echo "== overlap UT $world (AllGather under a concurrent GEMM, $(date -u +%T)) =="
  for n in "$MASTER" "$WORKER"; do ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench_gemm_overlap; pkill -9 -f torchrun; true'" 2>/dev/null; done; sleep 2
  ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$base && $tr --node_rank=1 $ov >/tmp/ovl_${world}_w.log 2>&1'" &
  local wp=$!; sleep 4
  ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$base && $tr --node_rank=0 $ov 2>&1'" | grep -E 'gemm-ovlp|per_rank|ratio' | tail -12
  wait "$wp" 2>/dev/null || true
}

echo "### standalone AllGather UT (HierAllGather SDMA vs RCCL) — nodes $MASTER/$WORKER ###"
run_world w8  4 0,1,2,3         "4 8 16 32 64 128 256 512" "$W8_ENV"
run_world w16 8 0,1,2,3,4,5,6,7 "8 16 32 64 128 256 512"   "$W16_ENV"
echo "### GEMM-overlap UT (no-CU-contention dividend) ###"
run_overlap w8  4 0,1,2,3         "$W8_ENV"
run_overlap w16 8 0,1,2,3,4,5,6,7 "$W16_ENV"
echo "### UT done ###"
