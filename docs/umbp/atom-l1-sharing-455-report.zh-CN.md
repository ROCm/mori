# 455 上 ATOM + UMBP L1 sharing 验证报告

测试日期：2026-11-19；整理日期：2026-11-20。

**结论：修复后的 ATOM + UMBP 已在 455 单 tray 四卡上完成真实模型的跨卡 L1 KV sharing，
所测复制范围零字节差异，应用正确性检查通过。** KV 从其他 GPU 的 HBM 复制到本卡 HBM 后用于推理。

本报告暂存于 `feat/umbp-ualink-kv-sharing`，实际测试使用下述独立版本组合；
不表示该 UALink 分支已完成 ATOM 集成或跨 tray 验证。

## 测试环境

- 硬件：单 tray，4 × gfx1250 / 455，每卡约 432 GiB HBM。
- 模型：DeepSeek-V4.1-Flash，四卡 DP/EP，FP8 KV，GMU 0.43。
- 镜像：`rocm/fw-bringup-mlperf:gfx1250-atom-20261116`。
- ATOM：[PR #3482](https://github.com/ROCm/ATOM/pull/3482)，提交 `bd897325f86d860e811b5c8fa04ad8097d0a776b`，使用 `umbp_pool` connector。
- MORI：recipe 基线 `6661e6ce474fd24166b4a2301567e26552ba9ff4` 加冻结补丁，修复构建为 `mori_l1_fence_455`。

## 跑通结果

| 检查 | 结果 |
| --- | --- |
| 实际跨卡 sharing | 覆盖四卡全部 12 个有向组合 |
| 复制量 | 690 次 Fetch，46,662 个槽，约 8.29 GiB |
| KV 字节校验 | 34,144 个范围，零差异、零读取错误、零 Fetch 错误 |
| 文档检索 needle | 384/384 正确 |
| 多轮负载 | 768/768 请求成功 |
| MORI CPU 回归 | 27/27 通过 |
| ATOM native L1 兼容性 | 91 passed，5 skipped；跳过项为 FakeL1 未实现的异步接口用例 |

日志中同时记录了 UMBP 的 GPU 3 → GPU 0 复制（86 槽、16,401,920 B，零差异），
以及 ATOM 的 `fetched 86 of 86 blocks from peer pools`，确认共享数据已被推理引擎接收并索引。

## 修复说明

旧实现允许外部 Fetch 写入旧 GPU 写入尚未结束的目的槽，旧写入可能覆盖新 KV；
诊断负载复现了 134 个范围的字节差异。

修复在复制前检查目的槽 `Fillable` 状态，遇到未完成槽时返回已复制的连续前缀，
剩余部分由 ATOM 本地计算。修复版实测触发 192 次 defer，校验零差异。
**未增加 mutex、阻塞等待或 HIP 同步。**

修复已提交 [MORI PR #1053](https://github.com/ROCm/mori/pull/1053)，目标分支为
`hyi/umbp-l1-external-fetch`。GPU 数据来自包含同一 guard 和诊断插桩的冻结修复构建
`mori_l1_fence_455`；最小 PR 源码另做了 CPU 回归。

## 质量与开销

Local 与修复版 L1 各运行两遍固定质量集，每臂 864 请求，全部配对 prompt-token hash 一致，无传输错误或截断。

| 项目 | Local 第 1 / 2 遍 | 修复版 L1 第 1 / 2 遍 |
| --- | --- | --- |
| 文档检索 | 144/144 · 144/144 | 144/144 · 144/144 |
| 多轮更新 | 64/64 · 64/64 | 64/64 · 64/64 |
| GSM8K | 191/200 · 190/200 | 190/200 · 190/200 |

未发现同一道题在两遍中均为 Local 正确而 L1 错误；结论仅限所测样本。
L1 第二遍没有新 Fetch，为本地热缓存，不能算第二次跨卡复制验证。

CPU 微基准共 19 个用例、456 个有效样本：最敏感的 16 B 小拷贝增加 **1.17%～1.94%** CPU 耗时；
190,720 B/槽的用例变化为 **−0.43%～+0.35%**，未见稳定退化。
当前数据不需要进一步优化 guard；该测量不包含 GPU 传输、RPC 或模型重算代价。
修复版完整模型吞吐 A/B 尚未完成，因此没有模型级性能增益或损失结论。

## 验证边界与记录

- 已验证路径为 **同 tray 跨 GPU 的 `hipMemcpy`**。gather 因 `d2d-across-devices` 回退，未验证跨卡 gather kernel。
- 本次未进行跨 tray 测试，也未把上述修复和 ATOM 接口移植到本报告所在的 UALink 分支。
- 主结果：`l1_fixed_quality_455_01`；local 质量对照：`ab_local_quality_455_02`；旧行为诊断：`l1_diag_455_01`。
- 原始日志、冻结构建指纹及 CPU 报告保存在 455 工作目录；详细复跑步骤见该机器上的 `RUNBOOK_455.md`。
  本次提交仅包含结果报告，换机复现仍需移交脚本、依赖、测试输入和模型。
