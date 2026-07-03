#!/usr/bin/env bash
# 2-node FSDP bench driver with container-watchdog + retry (beats the node reaper).
set -uo pipefail

N33=smci355-ccs-aus-n09-21.prov.aus.ccs.cpe.ice.amd.com
N29=smci355-ccs-aus-n09-29.prov.aus.ccs.cpe.ice.amd.com
IMG=rocm/pytorch-private:fsdp_sdma_mingzhi
CTR=mori-sglang-mingzhi
MASTER_IP=10.235.192.87
PORT="${PORT:-29570}"
IFACE=enp81s0f1

ensure_ctr() {  # ensure_ctr <node>  -- (re)create container AND ensure ionic RDMA provider installed
  local n="$1"
  ssh -o BatchMode=yes "$n" "docker ps --format '{{.Names}}' | grep -qx $CTR || { \
docker rm -f $CTR >/dev/null 2>&1; \
docker run -d --name $CTR --restart always --device=/dev/kfd --device=/dev/dri \
--device=/dev/infiniband --network host --ipc host --privileged --shm-size=256g \
--group-add video --security-opt seccomp=unconfined --security-opt label=disable \
--cap-add CAP_SYS_PTRACE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
-v /apps/mingzliu:/apps/mingzliu --entrypoint tail $IMG -f /dev/null >/dev/null 2>&1; }; \
docker exec $CTR bash -lc 'test -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so || \
{ dpkg -i /apps/mingzliu/ainic_debs/ionic-common_*.deb /apps/mingzliu/ainic_debs/libionic1_*.deb >/dev/null 2>&1; ldconfig; }' 2>/dev/null" 2>/dev/null
}

watchdog() {  # background loop
  while true; do ensure_ctr "$N33"; ensure_ctr "$N29"; sleep 8; done
}

run_once() {  # run_once <tag> <mode> <envstr>
  local tag="$1" mode="$2" envstr="$3"
  local args="bench.py --mode $mode --seq-len 1024 --steps ${STEPS:-30} --warmup ${WARMUP:-5}"
  local tr="torchrun --nnodes=2 --nproc_per_node=4 --master_addr=$MASTER_IP --master_port=$PORT"
  # env vars MUST attach to torchrun (via `env`), not to the preceding cd.
  local base="export HIP_VISIBLE_DEVICES=0,1,2,3 GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE MORI_SOCKET_IFNAME=$IFACE; cd /apps/mingzliu/fsdp_hier"
  ensure_ctr "$N33"; ensure_ctr "$N29"; sleep 2
  ssh -o BatchMode=yes "$N29" "docker exec $CTR bash -lc '$base && env $envstr $tr --node_rank=1 $args > /tmp/fsdp2_${tag}_w.log 2>&1'" &
  local wpid=$!
  sleep 4
  ssh -o BatchMode=yes "$N33" "docker exec $CTR bash -lc '$base && env $envstr $tr --node_rank=0 $args 2>&1'" > "/apps/mingzliu/fsdp_hier/fsdp2_${tag}_m.log" 2>&1
  wait "$wpid" 2>/dev/null || true
}

retry_run() {  # retry_run <tag> <mode> <envstr>  (retries until JSON summary appears)
  local tag="$1" mode="$2" envstr="$3" i
  for i in 1 2 3 4 5 6; do
    echo "[run] $tag attempt $i ($(date -u +%T))"
    PORT=$((PORT+1))
    run_once "$tag" "$mode" "$envstr"
    if grep -q 'avg_step_time_s' "/apps/mingzliu/fsdp_hier/fsdp2_${tag}_m.log" 2>/dev/null; then
      echo "[run] $tag SUCCESS on attempt $i"; return 0
    fi
    echo "[run] $tag attempt $i failed; tail:"; tail -3 "/apps/mingzliu/fsdp_hier/fsdp2_${tag}_m.log" 2>/dev/null
    sleep 3
  done
  echo "[run] $tag EXHAUSTED"; return 1
}

case "${1:-all}" in
  watchdog) watchdog ;;
  native) retry_run native native "" ;;
  hier)   retry_run hier mori "PYTHONPATH=/apps/mingzliu/mori_fsdp722/python:/apps/mingzliu/fsdp_hier MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 MORI_DISABLE_TOPO=1 MORI_SHMEM_HEAP_SIZE=17179869184 MORI_APP_LOG_LEVEL=warn" ;;
  *) retry_run native native ""; retry_run hier mori "PYTHONPATH=/apps/mingzliu/mori_fsdp722/python:/apps/mingzliu/fsdp_hier MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 MORI_DISABLE_TOPO=1" ;;
esac
