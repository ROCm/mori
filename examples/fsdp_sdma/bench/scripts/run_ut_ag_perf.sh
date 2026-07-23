#!/usr/bin/env bash
# Cross-node (w16, 2 node x 8 GPU) standalone AllGather-perf UT runner with TWO
# switch presets for the mori device (ibgda_sdma) handle:
#
#   perf : standalone_fast fan-out + all tuning knobs (MORI_HIER_UT_FAST=1 +
#          DEEP_PIPE=auto + SDMA_NUM_CHANNELS=8 + NIC_NUMA_LOCAL,
#          debug_sync OFF). Fast but not E2E-legal: the E2E FSDP adapter never
#          constructs HierAllGather with standalone_fast.
#
#   e2e  : the exact construction the w16 E2E FSDP run uses (MORI_HIER_UT_FAST=0 +
#          fuse knobs + DEBUG_SYNC=1 + CUDA_GRAPH=0). Bit-exact and E2E-safe; the
#          representative UT.
#
# RCCL (all_gather_into_tensor) is measured inline in both presets as the reference.
# usage: bash run_ut_ag_perf.sh <perf|e2e> [sizes_mb...]
set -u
PRESET="${1:?usage: run_ut_ag_perf.sh <perf|e2e> [sizes_mb...]}"; shift || true
SIZES="${*:-64 128 512}"

MASTER="${MASTER:-<master>}" ; MASTER_IP="${MASTER_IP:-<master-ip>}" ; WORKER="${WORKER:-<worker>}"
CTR="${CTR:-<container>}"
WT="${MORI_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"   # repo root (from bench/scripts/)
OUT="${OUT:-$WT/examples/fsdp_sdma/bench/results/mi300x_mlx5/raw}" ; mkdir -p "$OUT"
IFACE="${IFACE:-eth0}"
IB="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
GID="${NCCL_IB_GID_INDEX:-3}"
FABRIC="GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE \
NCCL_IB_HCA=$IB NCCL_IB_GID_INDEX=$GID MORI_RDMA_DEVICES=$IB"
FUSE="MORI_HIER_FUSE_LOCAL=1 MORI_HIER_FUSE_REMOTE=1 MORI_HIER_LOCAL_PUSHONLY=1"

case "$PRESET" in
  perf) ENVSET="MORI_ENABLE_SDMA=1 $FUSE MORI_HIER_UT_FAST=1 MORI_HIER_DEBUG_SYNC=0 \
MORI_HIER_DEEP_PIPE=auto MORI_SDMA_NUM_CHANNELS=8 MORI_HIER_NIC_NUMA_LOCAL=1" ;;
  e2e)  ENVSET="MORI_ENABLE_SDMA=1 $FUSE MORI_HIER_UT_FAST=0 MORI_HIER_DEBUG_SYNC=1 MORI_HIER_CUDA_GRAPH=0" ;;
  *) echo "preset must be perf|e2e"; exit 2 ;;
esac

PORT=$(( 29500 + RANDOM % 400 ))
BASE="export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$WT/python $FABRIC $ENVSET; cd $WT/tests/python/ccl"
TR="torchrun --nnodes=2 --nproc_per_node=8 --master_addr=$MASTER_IP --master_port=$PORT"
RUN="bench_ag_perf_w16.py --handle device --sizes-mb $SIZES --reps 10 --warmup 5"
echo "[$PRESET] sizes='$SIZES' port=$PORT env='$ENVSET'"
for n in "$MASTER" "$WORKER"; do
  ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench_ag_perf; pkill -9 -f torchrun; true'" 2>/dev/null
done; sleep 3
ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$BASE && $TR --node_rank=1 $RUN > /tmp/ut_${PRESET}_w.log 2>&1'" &
wp=$!; sleep 5
timeout 400 ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$BASE && $TR --node_rank=0 $RUN 2>&1'" | tee "$OUT/ut_${PRESET}_m.log"
wait "$wp" 2>/dev/null || true
echo "[$PRESET] result lines:"; grep -E "\[ag-perf\] [0-9]+MB" "$OUT/ut_${PRESET}_m.log"
echo "[$PRESET] -> python $OUT/plot_ag_perf.py  to (re)generate ag_perf_e2e.csv + ag_perf_e2e_stable_w16.png"
