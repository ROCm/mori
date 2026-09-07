#!/bin/bash
# Run one rank of an internode dispatch/combine test via torchrun.
#
# Usage:
#   run_internode_test.sh --rank <0|1> --master-addr <ip> --ifname <nic> \
#                         --cmd <bench|stress|test|test_sentinel> --max-tokens <N> \
#                         [--master-port <port>] [--kernel-type <v1|v1_ll|async_ll>] \
#                         [--num-qp <N>] [--quant-type <none|...>] [--dtype <bf16|...>] \
#                         [--combine-dtype <bf16|...>] [--hidden-dim <N>] [--topk <N>] \
#                         [--max-recv-total-tokens <N>] [--sentinel-pattern <p>] \
#                         [--nproc-per-node <N>] [--entry <path>]
#
# The optional shape/dtype flags are pass-throughs to the harness, which already
# accepts all of them; they are listed here so the cross-node leg can cover the
# same matrix the single-host pytest file does. Only --nproc-per-node is this
# script's own: it was pinned at 1, which made every cross-node run EP2 and left
# EP16 -- the shape that actually ships -- reachable only by hand-written
# torchrun lines outside this script.
#
# --entry selects which driver torchrun runs, defaulting to the shmem AOT harness.
# The v2 CCO entry takes the same CLI and differs only in installing the JIT
# redirect before the harness builds its first op, so the two are interchangeable
# here; anything else would have to parse these same flags.
#
# Environment variables GLOO_SOCKET_IFNAME and MORI_SOCKET_IFNAME are set
# automatically from --ifname. All other env vars (MORI_RDMA_SL, MORI_SHMEM_MODE,
# SGLANG_USE_AITER, etc.) should be set by the caller via docker exec -e.

set -euo pipefail

RANK=""
MASTER_ADDR=""
MASTER_PORT=1234
IFNAME=""
CMD=""
KERNEL_TYPE="v1"
NUM_QP=2
MAX_TOKENS=""
QUANT_TYPE=""
DTYPE=""
COMBINE_DTYPE=""
TOPK=""
HIDDEN_DIM=""
MAX_RECV_TOTAL_TOKENS=""
SENTINEL_PATTERN=""
NPROC_PER_NODE=1
ENTRY="examples/ops/dispatch_combine/test_dispatch_combine_internode.py"

while [[ $# -gt 0 ]]; do
  case $1 in
    --rank)             RANK="$2";                  shift 2 ;;
    --master-addr)      MASTER_ADDR="$2";           shift 2 ;;
    --master-port)      MASTER_PORT="$2";           shift 2 ;;
    --ifname)           IFNAME="$2";                shift 2 ;;
    --cmd)              CMD="$2";                   shift 2 ;;
    --kernel-type)      KERNEL_TYPE="$2";           shift 2 ;;
    --num-qp)           NUM_QP="$2";                shift 2 ;;
    --max-tokens)       MAX_TOKENS="$2";            shift 2 ;;
    --quant-type)       QUANT_TYPE="$2";            shift 2 ;;
    --dtype)            DTYPE="$2";                 shift 2 ;;
    --combine-dtype)    COMBINE_DTYPE="$2";         shift 2 ;;
    --topk)             TOPK="$2";                  shift 2 ;;
    --hidden-dim)       HIDDEN_DIM="$2";            shift 2 ;;
    --max-recv-total-tokens) MAX_RECV_TOTAL_TOKENS="$2"; shift 2 ;;
    --sentinel-pattern) SENTINEL_PATTERN="$2";      shift 2 ;;
    --nproc-per-node)   NPROC_PER_NODE="$2";        shift 2 ;;
    --entry)            ENTRY="$2";                 shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

for var in RANK MASTER_ADDR IFNAME CMD MAX_TOKENS; do
  [[ -z "${!var}" ]] && { echo "Missing required argument for --${var,,}"; exit 1; }
done

export GLOO_SOCKET_IFNAME="$IFNAME"
export MORI_SOCKET_IFNAME="$IFNAME"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f "$ENTRY" ]] || { echo "--entry not found under $REPO_ROOT: $ENTRY"; exit 1; }

# torchrun puts the entry's own directory on sys.path, not the repo root, so an
# entry that imports "tests.python.*" (the v2 one does, for the shared test
# utils) cannot resolve it. Under pytest the rootdir covers this; here nothing
# does.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXTRA_ARGS=()
[[ -n "$QUANT_TYPE" ]]     && EXTRA_ARGS+=(--quant-type "$QUANT_TYPE")
[[ -n "$DTYPE" ]]          && EXTRA_ARGS+=(--dtype "$DTYPE")
[[ -n "$COMBINE_DTYPE" ]]  && EXTRA_ARGS+=(--combine-dtype "$COMBINE_DTYPE")
[[ -n "$TOPK" ]]           && EXTRA_ARGS+=(--topk "$TOPK")
[[ -n "$HIDDEN_DIM" ]]     && EXTRA_ARGS+=(--hidden-dim "$HIDDEN_DIM")
[[ -n "$MAX_RECV_TOTAL_TOKENS" ]] \
  && EXTRA_ARGS+=(--max-recv-total-tokens "$MAX_RECV_TOTAL_TOKENS")
[[ -n "$SENTINEL_PATTERN" ]] && EXTRA_ARGS+=(--sentinel-pattern "$SENTINEL_PATTERN")

exec timeout "${MORI_INTERNODE_TIMEOUT:-120}" torchrun \
  --nnodes=2 \
  --node_rank="$RANK" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$ENTRY" \
  --cmd "$CMD" \
  --kernel-type "$KERNEL_TYPE" \
  --num-qp "$NUM_QP" \
  --max-tokens "$MAX_TOKENS" \
  "${EXTRA_ARGS[@]}"
