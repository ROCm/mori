#!/bin/bash
# Rigorous dispatch vs combine benchmark for MI350X
# DeepSeek V3 config: BF16 same-type, EP8, 7168 hidden, top-8
#
# Usage:
#   bash tools/bench_dispatch_gap.sh [--label LABEL] [--tokens "4096 8192 16384"]
#
# Prerequisites:
#   - All 8 GPUs must be idle (0% VRAM, 0% GPU)
#   - MORI must be installed (pip install .)

set -euo pipefail

LABEL="${LABEL:-default}"
TOKENS="${TOKENS:-4096 8192 16384 32768}"
ITERS=5
WARMUP=3
GRAPH_REPLAY=10
HEAP_SIZE=6G
WORLD_SIZE=8
DISPATCH_BN=256
DISPATCH_WPB=16
COMBINE_BN=56
COMBINE_WPB=15

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --label) LABEL="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --dispatch-geo) DISPATCH_BN=$(echo "$2" | cut -dx -f1); DISPATCH_WPB=$(echo "$2" | cut -dx -f2); shift 2 ;;
    --combine-geo) COMBINE_BN=$(echo "$2" | cut -dx -f1); COMBINE_WPB=$(echo "$2" | cut -dx -f2); shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

OUTDIR="bench_results/dispatch_gap_$(date +%Y%m%d_%H%M%S)_${LABEL}"
mkdir -p "$OUTDIR/raw"

# Step 1: Verify clean GPU state
echo "=== Step 1: Verify GPU state ==="
gpu_busy=0
while IFS= read -r line; do
  vram=$(echo "$line" | awk '{print $(NF-1)}' | tr -d '%')
  gpu=$(echo "$line" | awk '{print $NF}' | tr -d '%')
  if [ "$vram" -gt 15 ] || [ "$gpu" -gt 5 ]; then
    gpu_busy=1
  fi
done < <(rocm-smi 2>/dev/null | grep -E "^[0-9]")

if [ "$gpu_busy" -eq 1 ]; then
  echo "ERROR: GPUs are not idle. Aborting."
  rocm-smi 2>/dev/null | grep -E "^[0-9]"
  exit 1
fi
echo "All GPUs idle. Proceeding."

# Step 2: Record environment
echo "=== Step 2: Record environment ==="
cat > "$OUTDIR/environment.txt" << EOF
Date: $(date -Iseconds)
Label: $LABEL
Kernel: $(git -C /root/mori log --oneline -1 2>/dev/null || echo "unknown")
Kernel diff: $(git -C /root/mori diff --stat src/ops/dispatch_combine/intranode.hpp 2>/dev/null || echo "none")
MORI version: $(python3 -c "import mori; print(mori.__version__)" 2>/dev/null || echo "unknown")
GPU: $(rocminfo 2>/dev/null | grep "Name:.*gfx" | head -1 | awk '{print $2}')
ROCm: $(python3 -c "import torch; print(torch.version.hip)" 2>/dev/null)
PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
Heap size: $HEAP_SIZE
Dispatch geometry: ${DISPATCH_BN}x${DISPATCH_WPB}
Combine geometry: ${COMBINE_BN}x${COMBINE_WPB}
Iters: $ITERS, Warmup: $WARMUP, Graph replays: $GRAPH_REPLAY
Tokens: $TOKENS
EOF
cat "$OUTDIR/environment.txt"

# Step 3: Run benchmarks
echo ""
echo "=== Step 3: Run benchmarks ==="
echo ""

# Header for summary
printf "%-8s | %10s %10s %10s | %10s %10s %10s | %6s\n" \
  "Tokens" "Disp Mean" "Disp Std" "Disp Best" "Comb Mean" "Comb Std" "Comb Best" "Gap" \
  > "$OUTDIR/summary.txt"
printf "%-8s-+-%10s-%10s-%10s-+-%10s-%10s-%10s-+-%6s\n" \
  "--------" "----------" "----------" "----------" "----------" "----------" "----------" "------" \
  >> "$OUTDIR/summary.txt"

for tokens in $TOKENS; do
  echo "--- $tokens tokens ---"

  # Run benchmark
  PYTHONPATH=. MORI_SHMEM_HEAP_SIZE=$HEAP_SIZE python3 tests/python/ops/bench_dispatch_combine.py \
    --max-tokens $tokens --dtype bf16 --hidden-dim 7168 --world-size $WORLD_SIZE \
    --num-experts-per-token 8 --kernel-type IntraNode --zero-copy 1 \
    --dispatch-block-num $DISPATCH_BN --dispatch-warp-per-block $DISPATCH_WPB \
    --combine-block-num $COMBINE_BN --combine-warp-per-block $COMBINE_WPB \
    --cmd bench \
    > "$OUTDIR/raw/${tokens}t.txt" 2>&1

  # Parse results: extract per-round avg BW for dispatch and combine
  python3 -c "
import sys, statistics

d_bws, c_bws, d_lats, c_lats = [], [], [], []
phase = None
for line in open('$OUTDIR/raw/${tokens}t.txt'):
    if 'Dispatch result' in line: phase='d'
    elif 'Combine result' in line: phase='c'
    elif 'End-to-end' in line: phase=None
    elif phase and line.startswith('Round'):
        p = line.split('lat ')
        b = line.split('bw ')
        if len(p) > 1:
            lat = float(p[1].split()[0])
            bw = float(b[1].split()[0])
            if phase == 'd':
                d_bws.append(bw)
                d_lats.append(lat)
            else:
                c_bws.append(bw)
                c_lats.append(lat)

if d_bws and c_bws:
    d_mean = statistics.mean(d_bws)
    d_std = statistics.stdev(d_bws) if len(d_bws) > 1 else 0
    d_best = max(d_bws)
    c_mean = statistics.mean(c_bws)
    c_std = statistics.stdev(c_bws) if len(c_bws) > 1 else 0
    c_best = max(c_bws)
    gap = (c_mean - d_mean) / d_mean * 100
    toks = '$tokens'
    line = f'{toks:>8} | {d_mean:>10.1f} {d_std:>10.2f} {d_best:>10.1f} | {c_mean:>10.1f} {c_std:>10.2f} {c_best:>10.1f} | {gap:>+5.1f}%'
    print(line)
    with open('$OUTDIR/summary.txt', 'a') as f:
        f.write(line + '\n')
else:
    print(f'{toks}t: ERROR parsing results')
    print(open('$OUTDIR/raw/${tokens}t.txt').read()[-500:])
"
done

echo ""
echo "=== Summary ==="
cat "$OUTDIR/summary.txt"
echo ""
echo "Results saved to: $OUTDIR/"
