#!/usr/bin/env python3
"""Build compare_chart.png: RCCL (native) vs MORI SDMA HierAllGather, 2-node FSDP2 Qwen-7B."""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/apps/mingzliu/fsdp_hier"

def load(paths):
    out = []
    for p in paths:
        fp = os.path.join(D, p)
        if os.path.exists(fp):
            out.append(json.load(open(fp)))
    return out

native = load(["result_native_fair.json", "result_native_fair2.json"])
hier   = load(["result_hier.json", "result_hier2.json"])
assert native and hier, "missing results"

def mean(rows, k): return float(np.mean([r[k] for r in rows]))

n_tf, h_tf = mean(native, "avg_tflops_per_gpu"), mean(hier, "avg_tflops_per_gpu")
n_st, h_st = mean(native, "avg_step_time_s"),     mean(hier, "avg_step_time_s")
n_tok, h_tok = mean(native, "avg_tokens_per_s"),  mean(hier, "avg_tokens_per_s")
n_loss = [r["last_loss"] for r in native]
h_loss = [r["last_loss"] for r in hier]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
colors = ["#1f77b4", "#d62728"]
labels = ["RCCL\n(native)", "SDMA\nHierAllGather"]

# TFLOPS/GPU (higher better)
b = ax[0].bar(labels, [n_tf, h_tf], color=colors)
ax[0].set_title("TFLOPS / GPU  (higher = better)")
ax[0].set_ylabel("TFLOPS/GPU")
for r, v in zip(b, [n_tf, h_tf]): ax[0].text(r.get_x()+r.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom")

# step time (lower better)
b = ax[1].bar(labels, [n_st, h_st], color=colors)
ax[1].set_title("Avg step time (s)  (lower = better)")
ax[1].set_ylabel("seconds/step")
for r, v in zip(b, [n_st, h_st]): ax[1].text(r.get_x()+r.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom")

# throughput (higher better)
b = ax[2].bar(labels, [n_tok, h_tok], color=colors)
ax[2].set_title("Throughput (tokens/s)  (higher = better)")
ax[2].set_ylabel("tokens/s")
for r, v in zip(b, [n_tok, h_tok]): ax[2].text(r.get_x()+r.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom")

fig.suptitle(
    "FSDP2 Qwen-7B, cross-node (2 nodes x 4 GPU = world 8), bf16, seq=1024, steps=10/warmup=3\n"
    f"last_loss  RCCL={n_loss}  SDMA={h_loss}  (match within bf16 noise -> correctness control)",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(D, "compare_chart.png")
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"native tflops/gpu={n_tf:.2f} step={n_st:.4f} tok/s={n_tok:.0f} loss={n_loss}")
print(f"hier   tflops/gpu={h_tf:.2f} step={h_st:.4f} tok/s={h_tok:.0f} loss={h_loss}")
