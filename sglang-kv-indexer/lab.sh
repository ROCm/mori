#!/usr/bin/env bash
# lab.sh - one-command control for the SGLang <-> KV-indexer joint-debug lab.
#
#   ./lab.sh up            start indexer -> sglang (wait ready) -> bridge
#   ./lab.sh test [N]      send N (default 3) generate requests to make KV events
#   ./lab.sh logs          tail -f all three logs in one window (prefixed)
#   ./lab.sh status        show process + port state
#   ./lab.sh down          stop everything cleanly
#   ./lab.sh replay-test   stop bridge, send traffic (gap), restart bridge
#
set -u

# ---------------- config (override via env if needed) ----------------
GPU="${GPU:-0}"                       # single id "0" or TP/DP list "0,1,2,3,4,5,6,7"
TP="${TP:-1}"                         # tensor-parallel size (>=2 for multi-GPU)
DP_SIZE="${DP_SIZE:-1}"               # data-parallel replicas (>=2 => multi-worker)
MODEL="${MODEL:-/nfs/data/Qwen3-0.6B}"
MEM_FRACTION="${MEM_FRACTION:-0.6}"   # --mem-fraction-static
PAGE_SIZE="${PAGE_SIZE:-64}"          # --page-size
EXTRA_ARGS="${EXTRA_ARGS:-}"          # extra sglang flags, e.g. speculative/fp8 options
READY_ITERS="${READY_ITERS:-40}"      # readiness poll iterations (x8s each)
SGLANG_PORT="${SGLANG_PORT:-30010}"
PUB_PORT="${PUB_PORT:-5567}"
# sglang offsets each dp-rank's event port by +rank (offset_endpoint_port),
# so leave a wide gap between PUB and REPLAY bases to avoid cross-rank collisions.
REPLAY_PORT="${REPLAY_PORT:-5590}"
INDEXER_PORT="${INDEXER_PORT:-50051}"

CRATE_DIR="/root/wuyl/mori/sglang-kv-indexer"
SGLANG_DIR="/sgl-workspace/sglang"
LOG_DIR="/var/log/kvlab"
PID_DIR="/run/kvlab"
mkdir -p "$LOG_DIR" "$PID_DIR"

BIN_SERVER="$CRATE_DIR/target/debug/kv-indexer-server"
BIN_BRIDGE="$CRATE_DIR/target/debug/kv-indexer-bridge"

# ---------------- helpers ----------------
c_grn() { printf '\033[32m%s\033[0m\n' "$*"; }
c_red() { printf '\033[31m%s\033[0m\n' "$*"; }
c_yel() { printf '\033[33m%s\033[0m\n' "$*"; }

_alive() { # pidfile -> 0 if the recorded pid is running
  local pf="$PID_DIR/$1.pid"
  [ -f "$pf" ] || return 1
  local pid; pid=$(cat "$pf" 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

_descendants() { # pid -> print the pid and every descendant, children before parents
  local p="$1" c
  for c in $(pgrep -P "$p" 2>/dev/null); do _descendants "$c"; done
  echo "$p"
}

_stop() { # name -> TERM then KILL the whole process tree rooted at the recorded pid
  local name="$1" pf="$PID_DIR/$1.pid" pid pids
  pid=$(cat "$pf" 2>/dev/null)
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    # snapshot the tree first: even if the root dies and orphans reparent to
    # init, we already hold their pids and can still -9 them (multi-GPU workers)
    pids=$(_descendants "$pid")
    kill $pids 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 $pids 2>/dev/null
    c_yel "stopped $name (pid $pid; tree: $(echo $pids | tr '\n' ' '))"
  fi
  rm -f "$pf"
}

# ---------------- start primitives ----------------
start_indexer() {
  if _alive indexer; then c_grn "indexer already up (pid $(cat $PID_DIR/indexer.pid))"; return; fi
  RUST_LOG="${RUST_LOG:-info}" KV_INDEXER_LISTEN_ADDR="127.0.0.1:$INDEXER_PORT" \
    nohup "$BIN_SERVER" >"$LOG_DIR/indexer.log" 2>&1 &
  echo $! > "$PID_DIR/indexer.pid"
  c_grn "indexer up (pid $!) -> 127.0.0.1:$INDEXER_PORT  log:$LOG_DIR/indexer.log"
}

start_sglang() {
  if _alive sglang; then c_grn "sglang already up (pid $(cat $PID_DIR/sglang.pid))"; return; fi
  local kv="{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$PUB_PORT\",\"replay_endpoint\":\"tcp://*:$REPLAY_PORT\",\"buffer_steps\":10000}"
  local dp_arg=""; [ "$DP_SIZE" -gt 1 ] && dp_arg="--dp-size $DP_SIZE"
  # plain nohup: launch_server's TP/DP/detokenizer workers are children of this pid,
  # so _stop's process-tree walk reaps them without relying on process groups
  ( cd "$SGLANG_DIR" && HIP_VISIBLE_DEVICES="$GPU" \
    nohup python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --host 127.0.0.1 --port "$SGLANG_PORT" \
      --tp-size "$TP" $dp_arg \
      --mem-fraction-static "$MEM_FRACTION" --page-size "$PAGE_SIZE" \
      --enable-hierarchical-cache \
      --kv-events-config "$kv" \
      $EXTRA_ARGS \
      >"$LOG_DIR/sglang.log" 2>&1 & echo $! > "$PID_DIR/sglang.pid" )
  c_grn "sglang starting (pid $(cat $PID_DIR/sglang.pid)) GPU=$GPU TP=$TP DP=$DP_SIZE  log:$LOG_DIR/sglang.log"
  printf "waiting for sglang ready"
  for _ in $(seq 1 "$READY_ITERS"); do
    if grep -q "The server is fired up" "$LOG_DIR/sglang.log" 2>/dev/null; then
      echo; c_grn "sglang READY -> 127.0.0.1:$SGLANG_PORT  PUB:$PUB_PORT replay:$REPLAY_PORT"; return 0
    fi
    if ! _alive sglang; then echo; c_red "sglang died during startup, see $LOG_DIR/sglang.log"; return 1; fi
    printf "."; sleep 8
  done
  echo; c_red "sglang not ready after ~$((READY_ITERS * 8))s (still loading?). check $LOG_DIR/sglang.log"
  return 1
}

start_bridge() {
  # one bridge per dp-rank: rank r subscribes PUB_PORT+r / REPLAY_PORT+r (matching
  # sglang's per-rank port offset) and forwards to the single shared indexer.
  local rank name pub replay
  for rank in $(seq 0 $((DP_SIZE - 1))); do
    name="bridge-$rank"
    if _alive "$name"; then c_grn "$name already up (pid $(cat $PID_DIR/$name.pid))"; continue; fi
    pub=$((PUB_PORT + rank)); replay=$((REPLAY_PORT + rank))
    RUST_LOG="${RUST_LOG:-info}" \
    KV_INDEXER_WORKER_ID="worker-$rank" \
    SGLANG_KV_EVENT_ENDPOINT="tcp://127.0.0.1:$pub" \
    SGLANG_KV_EVENT_REPLAY_ENDPOINT="tcp://127.0.0.1:$replay" \
    SGLANG_KV_EVENT_TOPIC="" \
    KV_INDEXER_ENDPOINT="http://127.0.0.1:$INDEXER_PORT" \
      nohup "$BIN_BRIDGE" >"$LOG_DIR/$name.log" 2>&1 &
    echo $! > "$PID_DIR/$name.pid"
    c_grn "$name up (pid $!) SUB:$pub replay:$replay -> indexer:$INDEXER_PORT  log:$LOG_DIR/$name.log"
  done
}

# ---------------- subcommands ----------------
cmd_up() {
  [ -x "$BIN_SERVER" ] && [ -x "$BIN_BRIDGE" ] || { c_red "binaries missing; run: (cd $CRATE_DIR && cargo build --bins)"; exit 1; }
  start_indexer
  start_sglang || { c_red "sglang failed to become ready; aborting up"; exit 1; }
  start_bridge
  echo; cmd_status
}

cmd_down() {
  # stop every bridge instance we may have started (any DP_SIZE) + legacy name
  local pf name
  for pf in "$PID_DIR"/bridge-*.pid; do
    [ -e "$pf" ] || continue; name=$(basename "$pf" .pid); _stop "$name"
  done
  _stop bridge
  _stop sglang
  _stop indexer
  c_grn "all stopped"
}

cmd_status() {
  local pf names="indexer sglang"
  for pf in "$PID_DIR"/bridge-*.pid; do
    [ -e "$pf" ] || continue; names="$names $(basename "$pf" .pid)"
  done
  [ -e "$PID_DIR/bridge.pid" ] && names="$names bridge"
  for n in $names; do
    if _alive "$n"; then c_grn "$n: UP (pid $(cat $PID_DIR/$n.pid))"; else c_red "$n: down"; fi
  done
  echo "ports:"; (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) \
    | grep -E "$SGLANG_PORT|$PUB_PORT|$REPLAY_PORT|$INDEXER_PORT" || echo "  (none listening)"
}

cmd_logs() {
  trap 'kill 0' EXIT INT TERM
  tail -n 5 -F "$LOG_DIR/indexer.log" 2>/dev/null | sed -u 's/^/\x1b[36m[IDX]\x1b[0m /' &
  tail -n 5 -F "$LOG_DIR"/bridge*.log 2>/dev/null | sed -u 's/^/\x1b[35m[BRG]\x1b[0m /' &
  tail -n 5 -F "$LOG_DIR/sglang.log"  2>/dev/null | sed -u 's/^/\x1b[90m[SGL]\x1b[0m /' &
  wait
}

cmd_test() {
  local n="${1:-3}"
  local long=""; for _ in $(seq 1 60); do long="$long San Francisco is a city in California."; done
  for r in $(seq 1 "$n"); do
    curl -s -m 60 "http://127.0.0.1:$SGLANG_PORT/generate" -H "Content-Type: application/json" \
      -d "{\"text\":\"$long req$r:\",\"sampling_params\":{\"max_new_tokens\":32,\"temperature\":0}}" \
      -o /dev/null -w "req$r http=%{http_code}\n"
  done
  c_grn "sent $n requests; check './lab.sh logs' for REPORT/REVOKE on the indexer side"
}

cmd_replay_test() {
  c_yel "[replay-test] stopping bridge(s) to create a sequence gap..."
  local rank; for rank in $(seq 0 $((DP_SIZE - 1))); do _stop "bridge-$rank"; done
  _stop bridge
  c_yel "[replay-test] sending traffic while bridge is down..."
  cmd_test "${1:-4}" >/dev/null
  c_yel "[replay-test] restarting bridge; watch bridge log for replay/gap recovery"
  start_bridge
  sleep 2
  echo "---- bridge.log tail ----"; tail -n 20 "$LOG_DIR/bridge.log"
}

usage() {
  cat <<EOF
lab.sh - SGLang <-> KV-indexer 联调一键控制脚本

用法:
  ./lab.sh <命令> [参数]

命令:
  up                拉起全部服务:indexer -> sglang(等待就绪) -> bridge(全部后台)
  down              干净停止全部服务(按 pidfile 精确 kill,并清理进程组子进程)
  status            查看三个服务进程状态与端口监听
  logs              单窗口跟随三路日志,带彩色前缀 [IDX]青 [BRG]紫 [SGL]灰 (Ctrl-C 退出)
  test [N]          发送 N 条 generate 请求造 KV 事件 (默认 N=3)
  replay-test [N]   停 bridge -> 发 N 条流量造缺口 -> 重启 bridge,验证重放恢复 (默认 N=4)
  help              显示本帮助

典型流程:
  ./lab.sh up          # 起服务
  ./lab.sh logs        # 另一窗口看日志(可选)
  ./lab.sh test 3      # 发请求,indexer 侧应出现 REPORT tier=1/tier=2
  ./lab.sh down        # 收工

从宿主直接调用(无需进容器):
  docker exec kv-indexer-lab2 $CRATE_DIR/lab.sh up

可用环境变量覆盖默认配置:
  GPU=$GPU  TP=$TP  DP_SIZE=$DP_SIZE  MODEL=$MODEL
  MEM_FRACTION=$MEM_FRACTION  PAGE_SIZE=$PAGE_SIZE  READY_ITERS=$READY_ITERS
  EXTRA_ARGS=... (透传额外 sglang 参数,如投机解码/FP8 选项)
  SGLANG_PORT=$SGLANG_PORT  PUB_PORT=$PUB_PORT  REPLAY_PORT=$REPLAY_PORT  INDEXER_PORT=$INDEXER_PORT
  RUST_LOG=info|debug   (bridge/indexer 日志级别; debug 可看每批转发明细)

多 worker 示例 (2 个 DP 副本, 各占 1 卡, 各起一个 bridge):
  GPU=0,1 TP=1 DP_SIZE=2 ./lab.sh up
  # sglang rank0 PUB=$PUB_PORT replay=$REPLAY_PORT; rank1 PUB=+1 replay=+1
  # bridge-0 -> worker-0, bridge-1 -> worker-1, 共享同一 indexer

多卡示例 (DeepSeek-V4, 8 卡 TP + 更长就绪等待):
  GPU=0,1,2,3,4,5,6,7 TP=8 MODEL=/path/to/DeepSeek-V4-Flash-FP8 \\
  MEM_FRACTION=0.8 READY_ITERS=120 \\
  EXTRA_ARGS="--speculative-algorithm EAGLE --trust-remote-code" ./lab.sh up

日志文件位置: $LOG_DIR/{indexer,sglang,bridge}.log
EOF
}

case "${1:-}" in
  up)          cmd_up ;;
  down)        cmd_down ;;
  status)      cmd_status ;;
  logs)        cmd_logs ;;
  test)        shift; cmd_test "${1:-3}" ;;
  replay-test) shift; cmd_replay_test "${1:-4}" ;;
  help|-h|--help) usage ;;
  "") usage; exit 1 ;;
  *) echo "未知命令: $1"; echo; usage; exit 1 ;;
esac
