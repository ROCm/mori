# MI308X / dual-port Thor2 EP16 tuning

Measured on skyriver07 + skyriver04, each with 8 MI308X (80 CUs) and eight Thor2 RDMA bonds, each combining two 200Gbps ports. Base: main `62ee267419f1649ae61c269a9a5ed1aa7c223bed` plus the minimal BNXT completion fix used during measurement in [bnxt-prerequisite.patch](bnxt-prerequisite.patch), extracted from the issue addressed by PR #653. Both reference and candidate use this fix. PR #653 is now merged as `ceac6df6`; the patch is retained as measurement provenance, and this PR takes the transport fix from main. The complete matrix below is historical data from the stated measurement revision, not a new measurement of current main.

ROCm 7.14, RDMA SL=5 / TC=160. Device PCIe links are Gen5 x16. Existing firmware is 235.2.40.0 and lossless TC allocation is 50%; host settings were not changed. These profiles apply to this rig and software environment.

| Model | hidden | top-k | routed experts | experts/rank |
|---|---:|---:|---:|---:|
| DeepSeek R1 | 7168 | 8 | 256 | 16 |
| DeepSeek V4 Flash | 4096 | 6 | 256 | 16 |
| DeepSeek V4 Pro | 7168 | 6 | 384 | 24 |

Both `v2` and `v2_ll`, BF16→BF16 and FP8 E4M3FNUZ→BF16, tokens/rank = 4, 16, 64, 256, 1024, 4096, 8192, 16384. Capacity equals the live token count in each measurement. FP8 carries hidden/128 float32 scales; this does not model V4's native packed UE8M0 scale representation or FP4 expert weights. Shared experts and expert computation are outside this communication benchmark.

## Results

62 of 96 proposed QP/geometry configurations passed all acceptance gates. Among accepted changes, the median reduction in dispatch+combine time was **18.8%**, ranging from **6.4% to 38.8%**, relative to QP=2 and the original 96/64/8 geometry on both phases.

| Model | dtype | accepted / 16 | median reduction among accepted |
|---|---|---:|---:|
| R1 | BF16 | 11 | 15.1% |
| R1 | FP8→BF16 | 11 | 18.8% |
| V4 Flash | BF16 | 15 | 20.1% |
| V4 Flash | FP8→BF16 | 8 | 17.8% |
| V4 Pro | BF16 | 8 | 17.4% |
| V4 Pro | FP8→BF16 | 9 | 23.0% |

[performance.csv](performance.csv) includes rejections and all three ordinary benchmark samples. [validation.csv](validation.csv) contains the per-trial paired deltas, margins, worst timings and ordinary p95 values needed to audit the selection gates. [performance.svg](performance.svg) compares the validated selection with defaults. [qp-performance.csv](qp-performance.csv) records the initial QP sweep. [manifest.json](manifest.json) records official model config URLs/revisions and the measurement environment.

QP=1 used one physical port; QP=2/4/8 already split large-token traffic approximately 50:50 across both ports. More QPs therefore change concurrency and queueing. At 16K tokens, the initial geometry with QP=8 produced substantial PFC and long dispatch tails. Rechecking QPs after geometry tuning changed 12 selections, including cases where QP=8 became useful after tuning the pair of phases.

## Use an exact profile

From the repository root:

```bash
python tools/select_mi308_thor2_profile.py \
  --model r1 --dtype bf16 --kernel v2_ll --tokens 16
```

This prints public `EpDispatchCombineConfig` geometry/QP overrides, the measured capacity, datatype and hardware scope. `validated_change=false` means the proposed replacement failed a gate and the profile retains the reference defaults. The helper refuses unmeasured token counts. Profiles are opt-in; they are not a global tuning-table or `auto` threshold change. Recheck performance if capacity, NICs, EP size, transport dtype or runtime differ.

## Method and verification

- 1152 initial QP benchmarks: QP=1/2/4/8 at fixed dispatch/combine geometry 80/48/8, three interleaved repetitions.
- Coordinate geometry search on the selected QP, judging complete dispatch+combine latency. Candidates stay at or below 80 blocks, with RDMA blocks strictly fewer than total blocks. A block grid larger than CU count is not a general hardware error; 80 is this search's constraint.
- 720 further ordinary benchmarks revisit QP=1/2/4/8 at the tuned geometry for multi-chunk inputs.
- Three fresh two-node process batches validate every point once each, with a fresh CCO communicator per point and reversed/rotated case order. Each point includes five paired repetitions and ordinary benchmarks with one op alive at a time.
- Acceptance requires all three paired tests to win beyond max(1.5µs, 2%) with the tuner tail guard; every candidate ordinary mean to beat the reference median by that margin; and each rank-max-total p95 to remain within 10% of its matched reference.
- Ordinary validation uses 20 warmup / 30 measured / discard first round. HIP event regions include CPU submission gaps. Conversion and expert compute are excluded. The FP8 harness converts the full receive-capacity buffer between phases; its memory traffic and synchronization effects remain part of this workload.
- Every ordinary bench verifies numerical output and weights. FP8→BF16 checking uses combine precision, since the golden already begins with quantized input.
- 28 representative accepted geometry/QP combinations passed strict payload, indices, weight, scale-bit and nonidentity combine checks, including local-only, sentinel, ragged and zero-token cases. Strict checks use 65–67 tokens at capacity 128; full-size numerical checks cover the complete performance matrix.
- 35 existing region/JIT-binding tests passed. Source formatting checks passed.

## Reproduce a comparison

Install matching MORI/ROCm builds on two eight-GPU nodes. Set `NODE_RANK` to 0 and 1 respectively and `MASTER_ADDR` to node 0's reachable address. Network interface and SL/TC values must match the fabric; these were the measurement rig's settings:

```bash
export GLOO_SOCKET_IFNAME=bond0 MORI_SOCKET_IFNAME=bond0
export MORI_RDMA_SL=5 MORI_RDMA_TC=160

torchrun --nnodes=2 --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29651 \
  tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py \
  --cmd compare --kernel-type v2_ll --hidden-dim 7168 --topk 8 \
  --experts-per-rank 16 --max-tokens 16 --dtype bf16 --scale-dim 0 \
  --num-qp 1 --reference-qp 2 --tuning-pair 32,21,8,64,42,4 \
  --tuning-reps 5 --seed 4242 --result-json comparison.json
```

This R1 example compares the selected pair against the ordinary default geometry with QP=2. Repeat with seeds 5219 and 6196, adding `--candidate-first` to the middle run. Clear `MORI_EP_DISP_GEOM` and `MORI_EP_COMB_GEOM` when comparing against defaults; setting them intentionally changes the fixed reference.

For a geometry sweep, use `--cmd tuning --tuning-phase dispatch` or `combine`. `--tuning-candidates '32,16,4;64,32,8;80,48,8'` supplies a restricted search grid. JSON records the actual reference resolved from the op, including table lookup, dtype, overrides and untuned fallback.

The stricter payload/metadata check is [check_internode_payload.py](../../../tests/python/ops/dispatch_combine_v2/check_internode_payload.py). Run it with the same two-node torchrun launcher, `--family v2_ll --dtype bf16 --hidden 7168 --topk 8 --epr 16 --num-qp 1`, and pin the selected geometry through the two `MORI_EP_*_GEOM` environment variables. Scales are checked as opaque dword bits, matching `recv_scales()`'s API.

## Validation on the PR base

After rebasing onto main `ceac6df6` (including #653), a fresh ROCm 7.14 build passed the 35 region/JIT-binding tests and five two-node checks: BF16 paired comparison, FP8 comparison of QPs with identical geometry, strict FP8/v2 payload checks, strict BF16/v2_ll checks with 24 experts/rank, and an explicit two-candidate geometry sweep. Both hosts exited successfully for every check. The 288-trial CSV was also audited against the acceptance rules and reproduces all 62 accepted profiles. These checks validate the updated tools and representative configurations; the historical full performance matrix was not rerun on this revision.
