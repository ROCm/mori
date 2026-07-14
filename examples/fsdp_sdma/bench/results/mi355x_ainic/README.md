# w16 FSDP2 E2E on MI355X + AINIC (ionic RoCEv2)

Cross-node FSDP2 training-step benchmark for the hierarchical AllGather on a
**second hardware platform**: AMD **MI355X** GPUs with the **AINIC (ionic)** RoCEv2
NIC (the sibling results elsewhere in `bench/results/` are MI300X + mlx5).

## Headline (2 nodes x 8 GPU = w16, Qwen-7B, seq2048, bf16, 500 steps)

| backend | avg TFLOPS/GPU | vs native | loss |
|---|---|---|---|
| native (RCCL) | 250.35 | 1.00x | GT |
| **mori host-proxy ASYNC** | **270.20** | **1.079x** | **bit-identical to native** |

- Per-window loss is **bit-identical** to native for all 500 steps
  (last_loss `10.407537460327148` on both backends, Δ=0). See `e2e_w16_loss.png`.
- host-proxy ASYNC is stable at ~270 TFLOPS/GPU across the whole run; native sits at
  ~251 (one transient stall window). See `e2e_w16_tflops.png`.
- Why it wins on ionic: the CPU-posted host-proxy transport keeps the inter-node RDMA
  off the CUs and the deferred completion hides it behind the backward GEMM, so the
  copy-engine/NIC do the bulk moves while the CUs stay on the GEMM. (On ionic the
  GPU-initiated device ring's post/poll occupies CUs and loses this overlap; host-proxy
  recovers it.) Bulk AllGather bytes stay on RDMA/SDMA — no CU copy.

Files: `e2e_w16_native.log`, `e2e_w16_hostproxy.log` (raw run logs),
`e2e_w16_tflops.png`, `e2e_w16_loss.png`, `plot_mi355x_ainic.py` (regenerates the figures).

## How to reproduce

Both use the SAME one-key script `examples/fsdp_sdma/bench/scripts/run_e2e.sh`. It runs
`native` then `mori` and writes `../results/e2e_<world>_{native,mori}.log`. Compare
`last_loss` (must be bit-identical) and `avg_tflops_per_gpu`. The `mori` path is the
host-proxy ASYNC backend (a single env line inside the script, shown below).

### A) Original PR way — MI300X + mlx5 (the script defaults)
No overrides needed; the defaults target the mlx5 cluster:
```bash
cd examples/fsdp_sdma/bench/scripts
bash run_e2e.sh                 # WORLD=w16, 500 steps, mlx5, eth0, GID 3
```
(defaults: `MASTER=useocpm2m-097-040 WORKER=useocpm2m-097-083 IFACE=eth0`
`MORI_RDMA_DEVICES=mlx5_0,2,3,4,5,7,8,9 NCCL_IB_GID_INDEX=3`.)

### B) MI355X + AINIC (ionic) way — this result
Same script, override the node pair + NIC env for ionic. Containers named
`mori-sglang-mingzhi` must be up on both nodes; `/apps/mingzliu` bind-mounted; the
ionic RDMA provider installed in the container.
```bash
cd examples/fsdp_sdma/bench/scripts
MASTER=smci355-ccs-aus-n09-33.prov.aus.ccs.cpe.ice.amd.com \
WORKER=smci355-ccs-aus-n09-29.prov.aus.ccs.cpe.ice.amd.com \
MASTER_IP=10.235.192.86 \
IFACE=enp81s0f1 \
MORI_RDMA_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 \
NCCL_IB_GID_INDEX=1 \
MORI_REPO=<path to this mori checkout> \
WORLD=w16 STEPS=500 \
bash run_e2e.sh
```
Key ionic differences vs the mlx5 defaults: `IFACE=enp81s0f1`, ionic NIC list, and
**`NCCL_IB_GID_INDEX=1`** (ionic exposes RoCEv2 on GID 0/1, not 3).

### The mori backend env (set inside run_e2e.sh for `--mode mori`)
```
MORI_ENABLE_SDMA=1 MORI_FSDP_ENABLE_HIER=1 \
MORI_FSDP_HOST_PROXY=1 MORI_FSDP_HOSTPROXY_CAP_MB=512 \
MORI_SHMEM_HEAP_SIZE=17179869184 MORI_HOSTPROXY_ASYNC=1
```
`MORI_HOSTPROXY_ASYNC=1` is the enable flag; it auto-sets the double-buffered recv
staging (`ASYNC_RING=2`) and the landing drain (`ASYNC_DRAIN=1`) that keep it bit-exact.

Notes: figures were captured at `--warmup 30` (mori is already steady from step 0, so the
shipped `--warmup 6` reproduces the same headline). Regenerate figures with
`python3 plot_mi355x_ainic.py`.
