#!/usr/bin/env bash
# Parametrized w16 E2E driver for Team A pair 040/083 (mandate: no-inline-host-drain).
#   usage: EXTRA_ENV="..." bash A_w16_drv.sh <native|mori> <TAG>
# native draws the same-config GT; mori applies EXTRA_ENV on top of MORI_ENABLE_SDMA.
set -u
MASTER=${MASTER:-useocpm2m-097-040} ; MASTER_IP=${MASTER_IP:-10.158.213.159}
WORKER=${WORKER:-useocpm2m-097-083}
CTR=mori-sglang-mingzhi
WT=/home/mingzliu/sdma/mori ; EX=$WT/examples/fsdp_sdma
CFG=/home/mingzliu/sdma/qwen32k
OUT=/home/mingzliu/sdma/fsdp_hier
IFACE=eth0
IB=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
FABRIC="GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE \
NCCL_IB_HCA=$IB NCCL_IB_GID_INDEX=3 MORI_RDMA_DEVICES=$IB"
BENCH_ARGS="--model-name-or-path $CFG --seq-len ${SEQLEN:-2048} --steps ${STEPS:-20} --warmup ${WARMUP:-6} \
--micro-batch-size 1 --dtype bf16 --print-every 5"

MODE="${1:?native|mori}" ; TAG="${2:?tag}"
NPROC=${NPROC:-8} ; DEVS=${DEVS:-0,1,2,3,4,5,6,7}
MORI_ENV=""
if [ "$MODE" = mori ]; then MORI_ENV="MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 ${EXTRA_ENV:-}"; fi
PORT=$(( 29500 + RANDOM % 300 ))

BASE="export HIP_VISIBLE_DEVICES=$DEVS PYTHONPATH=$WT/python:$EX $FABRIC $MORI_ENV; cd $EX"
TR="torchrun --nnodes=2 --nproc_per_node=$NPROC --master_addr=$MASTER_IP --master_port=$PORT"
RUN="bench.py --mode $MODE $BENCH_ARGS"
echo "[$TAG] $(date -u +%FT%TZ) mode=$MODE port=$PORT mori_env='$MORI_ENV'"
for n in "$MASTER" "$WORKER"; do
  ssh -o BatchMode=yes "$n" "docker exec $CTR bash -lc 'pkill -9 -f bench.py; pkill -9 -f torchrun; true'" 2>/dev/null
done; sleep 2
ssh -o BatchMode=yes "$WORKER" "docker exec $CTR bash -lc '$BASE && $TR --node_rank=1 $RUN > /tmp/${TAG}_w.log 2>&1'" &
wp=$!; sleep 5
ssh -o BatchMode=yes "$MASTER" "docker exec $CTR bash -lc '$BASE && $TR --node_rank=0 $RUN 2>&1'" > "$OUT/${TAG}_m.log" 2>&1
wait "$wp" 2>/dev/null || true
echo "[$TAG] result:"; grep -E 'avg_tflops_per_gpu|avg_step_time_s|last_loss' "$OUT/${TAG}_m.log" | tail -4
grep -qi 'Slow wait' "$OUT/${TAG}_m.log" && echo "[$TAG] WARNING: Slow-wait detected"
echo "[$TAG] DONE $(date -u +%T)"
