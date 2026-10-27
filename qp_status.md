# MI308X EPv2 active-QP × load-first 调优：现状与后续

更新：2026-10-27
分支：`ROCm/mori` 的 `dev/analyze_v1`（PR #696），已合入 PR #698（active QP 作为按档调优维度）

## 1. 机制

PR #698 给每条跨节点规则加了 `"active_qps": N`：该档的 kernel 实际用几个 QP 发送。

- **分配数**：`num_qp_per_pe`，默认 8。
- **默认使用数**：规则里不写时取 `min(2, 分配数)`；写的值超过分配数会被截到分配数。
- **drain 的范围是一个 op**：dispatch 收尾时按这个 op 调度里**最大**的 QP 数做 drain。一个 op 只读和自己 dtype、hidden、topk 匹配的规则，所以 h6144 的行不影响 h7168 的 op。真正的耦合有两种：
  - 同一 hidden 下不同 dispatch dtype 共用 combine 行：fp8 dispatch、fp4 dispatch（h3584）都用 bf16 combine 行。
  - `auto` 模式下一个 op 同时用 V2LL（T ≤ 512）和 V2（T > 512）的行。
- **dispatch 的代价取决于同一档的 (d, c) 组合**：dispatch 末尾要把本档 combine 在它自己不发送的 QP 上留下的完成事件 drain 掉，所以 d < c 时 dispatch 可能比 d、c 单独看都慢。例如 V2 256/top6 T512：D(2,2)=417 µs，D(2,4)=513 µs，D(4,4)=414 µs。
- **空闲 QP 的 drain 不是免费的**：表内最大 QP 为 4 时，本档只用 2 个 QP 的 dispatch 在 T ≤ 64 多付约 1–2 µs（+1–3.5%）。
- **load-first**：只对 `InterNodeV2` combine 生效。实测 dispatch 时间和它无关（比值 1.000±0.6%），combine 时间和 d 无关（中位比值在 ±0.6% 以内），所以 dispatch 和 combine 可以分开建模。

QP 只影响跨节点 kernel（`InterNodeV2`、`InterNodeV2LL`）。`IntraNodeV2` 走 xGMI，没有 QP 维度。

## 2. 结论（2026-10-26/27，已写入仓库工作区，未 commit）

覆盖全部 MI308X 跨节点形状（fp4 dispatch 除外），共 156 个 cell：256/top8 的 h7168/h6144 × bf16/fp8 dispatch、256/top6 h7168 bf16、384/top6 的 h7168/h5120 × bf16/fp8 dispatch，V2 和 V2LL 两个 kernel 族。

整表 A/B（新表对现有表，交替 3 轮，取中位数，D+C 变化）：

| kernel | 形状 | 平均 | 最好 | 最差 |
|---|---|---|---|---|
| V2 | 256/top6 h7168 bf16 | −3.7% | T4096 −15.5% | T512 0.0% |
| V2 | 256/top8 h6144 bf16 / fp8 | −5.6% / −6.8% | T4 −11.1% / T8 −11.8% | T256 −0.3% / −1.1% |
| V2 | 256/top8 h7168 bf16 / fp8 | −5.1% / −5.3% | T4 −11.4% / T8 −12.4% | T128 +0.6% / T64 +1.5% |
| V2 | 384/top6 h5120 bf16 / fp8 | −4.6% / −5.1% | T512 −7.1% / T4096 −7.4% | −3.1% / −2.2% |
| V2 | 384/top6 h7168 bf16 / fp8 | −6.0% / −6.8% | T4096 −15.3% / −17.4% | T2048 +0.2% / T128 −2.1% |
| V2LL | 256/top6 h7168 bf16 | −2.6% | T4096 −6.8% | T64 +0.5% |
| V2LL | 256/top8 h6144 bf16 / fp8 | −1.7% / −2.0% | T4 −4.7% / −5.4% | T32 +2.2% / +0.7% |
| V2LL | 256/top8 h7168 bf16 / fp8 | −1.9% / −1.5% | T4096 −6.5% / T4 −3.9% | T32 +1.7% / T64 +1.1% |
| V2LL | 384/top6 h5120 bf16 / fp8 | −0.8% / −1.7% | T128 −1.7% / T512 −2.8% | T64 +1.3% / +0.4% |
| V2LL | 384/top6 h7168 bf16 / fp8 | −2.1% / −2.7% | T4096 −6.5% / T2048 −7.5% | T64 +0.5% / +0.4% |

合计：V2 平均 −5.5%，V2LL 平均 −1.9%。44 个 cell 快 5% 以上；5 个 cell 慢 1% 以上（最多 +2.2%），都是保持 (2,2) 的档，代价来自空闲 drain。

规律（括号为 (dispatch QP, combine QP)）：

- **V2 小 token（T4–T128）**：combine 用 1 个 QP 并开 load-first，快 5–12%，其中 load-first 贡献大部分（combine 快 10–18%）。
- **V2 大 token（T256–T4096）**：combine 用 4 个 QP，dispatch 用 1 或 4 个（避开 d=2 < c=4 的组合），T4096 快 4–17%。
- **V2LL**：T4–T16 用 (2,1)；T128–T256 dispatch 用 1；T512–T4096 combine 用 4。
- **表内最大 QP 一律为 4**。上限 8 的方案模型预测只多快 0.0–0.4%，没有采用；这也让分配数有机会降到 4。
- 之前单独调的 256/top8 h7168 bf16 结论（V2LL 6 档、V2 2 档）已被这次统一重选取代：那组 combine 行对共用它的 fp8 op 几乎没有收益（模型预测 V2 −0.4%、V2LL −1.5%，新选择是 −5.7%、−2.2%）。

整表 A/B 后回退的改动：V2 256/top6 T512、T1024，V2LL 256/top6 T512，三档回到 (2,2)，确认 A/B 持平。V2 384/top6 h7168 bf16 T256 的 dispatch 从 4 改为 1，确认 A/B −8.5%。

## 3. 方法

脚本和数据在 `/home/zqz/workspace/mori_mi308_retest_20261019/`，新增脚本均以 `fx` 开头。

1. **全组合筛选**（`fx_make.py`、`fx_run.py`、`fx_analyze.py`）：17 个整表变体，每个变体把所有 MI308X 跨节点规则设成 dispatch=d、combine=c（d ∈ {1,2,4}，c ∈ {1,2,4,8}，外加 (8,8)），V2 另有 (q,q)+load-first 的 4 个变体。用 `driver.py` 在和最终 A/B 相同的条件下测（`scale_dim=0`，每个 cell 新建 op，样本取中位数），交替顺序跑 2 轮。每个 cell 估计 D(d,c) 和 C(c,LF)，按共用 combine 行的簇逐档选择。换掉默认需要满足：比 (2,2)、原有 load-first 快 1% 以上，并且每一轮都更快。
   - 不用 harness 的 sweep，是因为它的测量条件不同（`scale_dim=32`、两个 op 同时驻留、取均值），之前很多候选在整表对比里复现不了。
2. **异常运行（stall）**：约 2–6% 的运行会整段落进 5–150 倍慢的模式，常常是同一次 launch 里连续几个 cell 一起中招，各种配置都会碰上，历史数据里也有。dispatch 或 combine 中位数超过该 cell 所有运行中位数 2 倍的运行会被剔除，并用 `fx_repair.py` 补测。
3. **整表 A/B**（`fxab_chain.sh`、`fx_ab_report.py`）：新表对现有表，156 个 cell，交替 3 轮。没有收益的改动回退后，再对改动过的 cell 做确认 A/B（`fxcf`）。
4. **落表**（`fx_apply.py`、`fx_finalize.py`）：metric 字段用 A/B 中通过 README 噪声门限、延迟居中的那次运行刷新。没测到的行保留原值，包括 fp4 dispatch、对称 fp8 combine，以及 384/top6 h5120 fp8 T2048 的 V2 dispatch（三次运行都没过噪声门限）。

注意事项：

- 新建的 manifest 和表目录要先 rsync 到远端节点；传给 `run_job.py` 的路径必须是绝对路径（`driver.py` 在容器里的工作目录不同）。
- `run_job.py` 在 launch 结束、进程退出的那一秒常把自己的进程误报为外部 GPU 进程（返回码 1，但 cell 全部有效），可以忽略。
- 一轮全组合筛选（17 个变体 × 156 个 cell）约 1.5 小时；整表 A/B 3 轮约 25 分钟。

## 4. 产物与仓库状态

| 内容 | 位置 | 状态 |
|---|---|---|
| MI308X 跨节点表（QP、load-first、metric） | `python/mori/ops/dispatch_combine_v2/tuning_configs/gfx942_mi308x_InterNodeV2_ep16.csv`、`..._InterNodeV2LL_ep16.csv` | 已写入工作区，未 commit |
| V2 表改为 CSV：每个型号 × kernel × EP 一个文件，两相合并（48 个 JSON → 9 个 CSV），加载时严格校验；新增 `test_v2_tuning_tables.py` 和对应的 CI 步骤 | `hip_tuning_configs.py`、`tuning_configs/*.csv`、`setup.py`、`.github/workflows/ci.yml` | 未 commit；原 JSON 备份在实验目录 `tables_json_backup/`，实验目录里的 `fx_*` 脚本仍按 JSON 表工作 |
| README：active_qps 说明、MI308X load-first 设置、QP 测量条件 | `tuning_configs/README.md` | 未 commit |
| tuning 脚本 pair 模式 | `tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py` | 未 commit（本次没有用到） |
| 最终表 / 回退前的表 / 选择结果 | `tables_fxfinal/`、`tables_ab/final4`、`fx_picks_q4.json` | 实验目录 |
| 分析和报告 | `fx_analysis_q4.txt`、`fx_analysis_q8.txt`、`fxab_report.txt`、`fxcf_report.txt` | 实验目录 |
| PR #698 合并提交 | `0cba4d02` | 已 push |
| PR #696 回复草稿 | `PR696_REVIEW_REPLIES_DRAFT.md` | 等 QP 工作结束后统一回复 |

## 5. 后续工作

1. **review 并 commit**：12 张表、README、tuning 脚本一起提交。
2. **fp4 dispatch 没有测**：`driver.py` 和 harness 都不支持 fp4。fp4 dispatch 行保持默认 2，但它共用的 h7168 combine 行在 T256+ 已改为 c=4。以 fp8 作参照，V2 在 T256–T1024 用 d=2、c=4 时 dispatch 比 (2,2) 慢约 5%，用 d=4 则更快，所以 fp4 需要补测（同时也是 PR #696 的待办之一）。
3. **`auto` 模式没有直接测**：`driver.py` 只测固定 kernel 族。新表下 auto op 的 drain 也是 4，各档配置和固定族时一样，预计结论成立，但没有实测。
4. **分配数**：表内最大 QP 已是 4，可以考虑把 `num_qp_per_pe` 的默认值从 8 降到 4。需要先确认没有其他使用者需要 8，并用整表 A/B 验证分配 4 时性能不变。
5. **空闲 drain**：dispatch 按整个 op 的最大 QP 数 drain，保持 (2,2) 的小 token 档因此多付 1–2 µs。若在运行时按 max(本档 d, 上一次 combine 的 c) 来 drain，就能去掉这部分，但这是 kernel 改动，要评估正确性。
6. **回复 PR #696 reviewer**：用草稿文件，处理其中「Still open」一节剩下的项。
7. **（可选）MI300X、MI355X**：同样的流程（`fx_*` 脚本）可以复用，但需要对应机器。
