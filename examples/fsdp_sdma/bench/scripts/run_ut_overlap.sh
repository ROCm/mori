#!/usr/bin/env bash
# Cross-node (w16, 2 node x 8 GPU) compute/comm OVERLAP UT launcher for
# tests/python/ccl/test_overlap_w16.py.
#
# Measures the GEMMs' OWN completion time while N AllGathers run concurrently,
# RCCL (CU-resident) vs hp_sdma (host-proxy: cross-node CPU-posted + intra-node
# SDMA copy engine, both CU-free). LOWER hp_sdma GEMM time = the collective
# steals less GPU from compute. HARD bit-exact gate (torch.equal vs RCCL).
#
# Env matches the original hp_sdma run recipe (host-proxy async + SDMA-intra +
# NUMA-local NIC), i.e. the same construction the w16 E2E hp_sdma FSDP run uses.
#
# usage: bash run_ut_overlap.sh [gemm_n] [size_mb] [nops]
#   e.g. bash run_ut_overlap.sh 2048        # defaults: size_mb=8 nops=50
#        bash run_ut_overlap.sh 4096 8 50
set -u
GEMM_N="${1:-2048}" ; SIZE_MB="${2:-8}" ; NOPS="${3:-50}"

MASTER="${MASTER:-useocpm2m-097-040}" ; MASTER_IP="${MASTER_IP:-10.158.213.159}" ; WORKER="${WORKER:-useocpm2m-097-083}"
CTR="${CTR:-mori-sglang-mingzhi}"
WT="${MORI_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"   # repo root (from bench/scripts/)
OUT="${OUT:-$WT/examples/fsdp_sdma/bench/results/mi300x_mlx5/raw}" ; mkdir -p "$OUT"
IFACE="${IFACE:-eth0}"
IB="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
GID="${NCCL_IB_GID_INDEX:-3}"
FABRIC="GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE \
NCCL_IB_HCA=$IB NCCL_IB_GID_INDEX=$GID MORI_RDMA_DEVICES=$IB"

# hp_sdma overlap recipe (host-proxy async + SDMA-intra + NUMA-local), verbatim.
ENVSET="MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 MORI_FSDP_HOST_PROXY=1 \
MORI_FSDP_HOSTPROXY_CAP_MB=512 MORI_SHMEM_HEAP_SIZE=17179869184 \
MORI_HOSTPROXY_ASYNC=1 MORI_HOSTPROXY_SDMA_INTRA=1 MORI_HIER_NIC_NUMA_LOCAL=1"

PORT=$(( 29500 + RANDOM % 400 ))
BASE="export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$WT/python $FABRIC $ENVSET; cd $WT/tests/python/ccl"
TR="torchrun --nnodes=2 --nproc_per_node=8 --master_addr=$MASTER_IP --master_port=$PORT"
RUN="test_overlap_w16.py --size-mb $SIZE_MB --nops $NOPS --gemm-n $GEMM_N --reps 8 --warmup 5"
echo "[overlap] gemm_n=$GEMM_N size_mb=$SIZE_MB nops=$NOPS port=$PORT"
echo "[overlap] env='$ENVSET'"
for n in "$MASTER" "$WORKER"; do
  ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f test_overlap_w16; pkill -9 -f torchrun; true'" 2>/dev/null
done; sleep 3
ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$BASE && $TR --node_rank=1 $RUN > /tmp/ut_overlap_w.log 2>&1'" &
wp=$!; sleep 5
timeout 400 ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$BASE && $TR --node_rank=0 $RUN 2>&1'" | tee "$OUT/ut_overlap_gemm${GEMM_N}_m.log"
wait "$wp" 2>/dev/null || true
echo "[overlap] result:"; grep -E "\[overlap-w16\]" "$OUT/ut_overlap_gemm${GEMM_N}_m.log"
grep -qiE "MISMATCH|Traceback|Aborted|Slow wait|Memory access fault" "$OUT/ut_overlap_gemm${GEMM_N}_m.log" && echo "[overlap] !!! ANOMALY"
