# HIP V2 tuning configurations

Tables are keyed by GPU architecture/model, kernel family, EP size, total
expert count, top-k, and phase. Each rule selects a launch geometry by dtype,
hidden dimension, and an inclusive token ceiling. Dispatch and combine have
separate tables; tokens above the last ceiling use the last rule.

## MI308X coverage and measurement status — 2026-10-16

MI308X (`gfx942`) matches the expert/family/phase file coverage of MI355X
(`gfx950`):

| Total experts | Top-k | IntraNodeV2 | InterNodeV2 | InterNodeV2LL |
| --- | --- | --- | --- | --- |
| 256 | 8 | EP8 | EP16 | EP16 |
| 256 | 6 | EP8 | EP16 | EP16 |
| 384 | 6 | EP8 | EP16 | EP16 |

Each cell has dispatch and combine tables, for 18 files per GPU model.
The four existing MI308X internode files for E256/K8 retain all 190
pre-existing rules unchanged. The 14 new files contain 189 rules:

| New MI308X files | Adopted measured rows | Rows not yet filled |
| --- | ---: | ---: |
| Six IntraNodeV2 files | 77 | 0 |
| Two InterNodeV2 files, E256/K6 | 14 | 0 |
| Two InterNodeV2LL files, E256/K6 | 12 | 2 |
| Two InterNodeV2 files, E384/K6 | 42 | 0 |
| Two InterNodeV2LL files, E384/K6 | 36 | 6 |
| **Total** | **181** | **8** |

All 119 target cells have been attempted. Accepted stable evidence for 114 cells
supports 181 rows: 114 dispatch and 67 combine. After host RoCE restoration,
all 35 ordinary-V2 E256/K6 and E384/K6 cells were remeasured and their 56 rows
were applied: 48 existing metric rows refreshed and eight missing rows filled.
The other 125 adopted rows retain historical evidence from 79 cells. Five LL
cells remain unadopted. File coverage is complete; global optimality is not
established. No accepted cell lacks a required shared Combine context.

The new files follow MI355X JSON fields and order. Measurement status and
provenance live in the external audit, not additional JSON metadata. Pending
rows have no performance values; MI355X performance was not copied. FP8 uses
`fp8_e4m3_fnuz` on gfx942. E256/K6 has BF16 only; E384/K6 covers H5120
and H7168. All combine rules use BF16. Measured token counts are
T64/128/256/512/1024/2048/4096, excluding the pending groups below.
Pending rows retain their original imported launch values. Attempts of the
corresponding shapes used separately recorded candidates and do not validate
those stored pending geometries.

## Ordinary V2 Dispatch parameter search and application

After the fixed-parameter refresh below, all 35 ordinary InterNodeV2 E256/K6
and E384/K6 contexts underwent Dispatch tuning: 525 coarse candidates followed
by 140 warp-count candidates. Three Dispatch rows passed three independent
ABA confirmations and now use new stored geometry:

| Experts / hidden / dispatch dtype / tokens | Old B/R/W | Applied B/R/W |
| --- | --- | --- |
| 256 / 7168 / BF16 / 2048 | 256/128/8 | 80/64/8 |
| 384 / 7168 / FP8 FNUZ / 2048 | 256/128/8 | 80/53/16 |
| 384 / 5120 / BF16 / 4096 | 256/128/8 | 80/64/8 |

All 35 contexts then passed real AUTO lookup measurements of the selected
tables. These measurements supply the current 35 Dispatch and 21 canonical
Combine metric rows. The other 32 Dispatch geometries and all Combine
geometries are preserved. A fourth nominee, E384/H5120/BF16/T512, missed the
3% Dispatch improvement requirement in one of three confirmations and was
not adopted. This finite search does not establish a global optimum.

The four ordinary-V2 E256/K6 and E384/K6 JSONs are applied; all other 44 JSONs,
kernel code and V2LL are unchanged by this tuning task. Coverage remains
181/189. Current evidence is in `dispatch_tune_20261016/final_auto_audit_final.json`,
`final_stage/change_plan.json` and `application_verification.json` under the
task artifact directory below. The earlier refresh checkpoints are historical
for these four tables. See repository `bw.md` and `dispatch.md` for performance.

## Measurement protocol and metric meaning

Measurements use eight MI308X GPUs per node (80 CUs each), EP8 intra or EP16
across two nodes, from commit `9d43fe32e78e7359c569d8fbe88cd46c14a72ca7`.
The task-private build uses Python 3.12.13, PyTorch 2.12.0+git6bbd260 and HIP
7.2.53211 with matching HIP/HSA binaries. No kernels were changed.

Each accepted cell passes two weighted identity-expert checks and one all-rank
payload check of the exact timed `combine_weights=None` branch. Routing is
uniform, scale dimension is 0, and capacity equals token count. Timing uses
three passes, each with 20 warmups and 30 measured rounds, dropping the first:
87 retained samples per rank and phase. Explicit dtype conversion is outside
phase event windows. Raw metrics, actual launch geometry, runtime identity,
complete dispatch/combine pairing and process ownership are checked.

Bandwidth is the grand mean of per-rank, per-round `bytes/time`; latency is
the mean time, so average bandwidth is not `bytes / average latency`.
The primary bandwidth field represents XGMI for intra, RDMA for ordinary
inter, and the LL accounting metric for V2LL. Inter rows retain separately
named RDMA, XGMI and LL metrics; these are not interchangeable. BF16 contexts
supply canonical BF16 combine metrics. FP8 contexts write dispatch only and
must confirm the same combine geometry.

An otherwise valid measurement is held if either phase has mean/median
latency greater than 1.10 or max/median greater than 5, or the three-pass D+C
mean max/min ratio exceeds 1.20. These are conservative stability gates,
not significance tests. Every prescribed sample is retained; diagnostic
medians do not replace stored grand means.

**Internode results require QP64.** JSON selectors have no QP dimension;
loading a table does not configure QP64. These results do not validate default
QP2, whose T512 local-QP-operation error remains unresolved. Intra records
use QP2 without RDMA payload transfer. Other controls are `CCO_GDR_CAPABLE=0`,
`CCO_UNCACHED_WINDOW=1`, all eight `bnxt_re_bond` devices, TC160/SL5 and
bond0 bootstrap. GDR0 still uses direct GPU RDMA.

## Earlier ordinary V2 refresh after network restoration

The October 16 refresh kept all stored launch parameters and used fresh
measurements for all 35 cells, including all 21 complete shared Combine groups.
The original loader still clamps launch geometry to 80 CUs; for example,
stored 256/128/8 resolves to actual 80/79/8. The JSON retains the stored values,
and the external audit records both stored and effective geometry. Only the
six standard metric fields were updated, in the matching MI355X field order.

All original correctness and noise gates passed. None of the 100,800 raw
timing samples exceeded 50 ms. The selected results are initial runs 000–028
and retry1-029–034. Initial 029 was excluded in full after ownership could
not be confirmed for a disappearing process at exit; initial 030–034 did not
start. The guard and timing thresholds were not relaxed. Earlier diagnostic
metrics were not reused. The final 14:57 CST check found both hosts idle and
all 32 PF / 16 CC configurations correct.

This refresh established the ordinary-V2 baseline before parameter tuning. The 48
rows with previous values have a median latency change of -0.0808%; this is
not a kernel optimization or proof that the fixed geometries are optimal.
Other families and K8 were not remeasured. Runtime network settings must be
restored and read back after reboot.

## Rows not yet filled

All remaining groups use top-k 6. Each E256 group has BF16 dispatch and combine;
each E384 group has BF16/FP8 dispatch and one shared BF16 combine row.

| Family | Experts / hidden / tokens | Pending rows |
| --- | --- | ---: |
| InterNodeV2LL | E256 / H7168 / T2048 | 2 |
| InterNodeV2LL | E384 / H5120 / T4096 | 3 |
| InterNodeV2LL | E384 / H7168 / T2048 | 3 |

## Historical failures and network diagnosis

Before host RoCE restoration, these groups had timeout or unstable timing evidence, including isolated
retries and alternative geometries. D/C64/1/8 also timed out at ordinary
E256/T2048. Ordinary E256/T4096 subsequently passed an isolated confirmation
and was adopted; its older long-tail observation remains excluded. Ordinary
E384/H7168/T4096 passed with BF16 D80/48/8 and unchanged C80/53/8, completing
the group with an earlier stable FP8 context. Failure is not monotonic in T.

No task GPU job is active. Testing resumed after the user authorized stopping
the occupying service; the final resource check found no KFD processes on
either host. Source, driver and kernel code remained unchanged.

Before restoring the network on October 16, four additional ordinary-V2 E384 candidates were tested across
H7168/T2048 and H5120/T4096, with BF16 and FP8 contexts for each. Three reached
the 120-second timeout; one completed with severe dispatch timing tails. No
complete group qualified for adoption, so coverage at that checkpoint remained
173/189. Independent E256/T2048 diagnostics found an 80.3-second
GPU drain for 20 warmup pairs. Added per-operation GPU synchronization and
all-rank phase barriers did not remove repeated approximately 4.2-second
waits. These diagnostics changed observation or scheduling and are excluded
from performance adoption. No LL kernel or configuration was changed.

Later the same day, both hosts were found to have PFC disabled, empty APP
priority mappings and unconfigured ETS on all 32 physical PFs. Restoring the
reference host QoS/PFC/DSCP/mqprio and RoCE ECN/CC configuration resolved the
five original ordinary-V2 cases: four prior timeouts and one severe timing tail
all passed with identical source, driver, geometry and input tables. All
14,400 raw timing samples were below 50 ms, and the original noise gates passed.
ACK timeout remained 20; no timeout hook, quiet probe or LL test was run.

Those five runs were network-diagnosis results, not an adopted tuning set.
Coverage then remained 173/189; the subsequent full refresh above independently
remeasured all ordinary-V2 contexts and brought the adopted count to 181/189.
The E256 diagnostic deliberately retained slow D/C64/1/8; passing it does not
establish an optimal geometry. Future selection and final confirmation should
use the restored network baseline. Both hosts passed configuration readback and
had no KFD processes at 2026-10-16 13:15 CST. The configuration is runtime state
and must be restored and read back after reboot.

## Evidence and continuation

Persistent artifacts are under `/home/zqz/workspace/mori_mi308_tuning_20261014/`:

- `finalization/resumed_final_noise/noise_audit.json` and
  `adopted_evidence_paths.txt`: historical acceptance before this refresh.
- `finalization/clean_merge_{r7small,r7large,r8isolated,resumed_final}/`:
  `merge_audit.json` and `application_verification.json` for each application.
- `selected_manifest_after_resumed_final.json`: frozen historical manifest;
  its ordinary-V2 scope metrics are superseded by this refresh.
- `refresh_pfc_20261016/audit_final_001/noise_audit.json`: all 35 newly accepted
  ordinary-V2 contexts and the exact 56-row metric mapping.
- `refresh_pfc_20261016/application_plan_001/application_checkpoint.json` and
  `application_verify_001/application_verification.json`: earlier refresh application
  checkpoint, with new scope evidence and retained outside-scope history.
- `refresh_pfc_20261016/comparison_applied_001/REPORT.md` and
  `old_new_metrics.csv`: full old/new comparison before Dispatch tuning.
- `evidence/`, `logs/`, `MEASUREMENT_PROGRESS.md` and `RUNSTATE.md`:
  raw records, attempts, runtime provenance and remaining work.
- `resumption_20261016/REPORT.md`: historical candidate failures and isolated
  phase diagnostics before restoring host RoCE configuration.
- `pfc_check_20261016/REPORT.md`: successful five-context reproduction checks
  after network restoration and the diagnostic raw noise/NIC audits.

Resumed applications progressed 99→159→168→170→173, adding 74 measured rows
while preserving all previous 99 row values and MI355X-compatible schema.
The subsequent fixed-parameter refresh replaced 48 historical metric rows and
filled eight more, bringing coverage to 181/189. The later Dispatch tuning
updated three geometries and all 56 ordinary-V2 metric rows. Use its
`dispatch_tune_20261016/application_verification.json` for these four tables
and the historical audit for other tables;
raw `status=valid` and older run-level approval lists alone are insufficient.
Historical r5 audits retain their original 65-cell snapshot and pre-cleanup
repository hashes. Apply only reviewed geometry/metric updates to the clean
JSON schema; old staging output would reintroduce removed metadata.
