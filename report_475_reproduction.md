# ROCm/mori#475 复现报告：InterNodeV1 跨节点 combine 静默损坏

对 [ROCm/mori#475](https://github.com/ROCm/mori/issues/475) 的实机复现与归因实验。只陈述实测结果，不提根因假说；kernel 层的缺陷分析见 `bugfix/v1_kernel` 分支的 `report.md`（两者有一处出入，见 §4）。

- 基线 `0d05a4d2`（main），无故障注入；2 × (8× MI355X gfx950)，Pensando Pollara / `ionic` RoCE
- **稳定复现**：dispatch 始终正确，损坏只出现在跨节点 combine 返回路径

---

## 一、复现结果

用 issue 报告人 [@Oseltamivir](https://github.com/Oseltamivir) 提供的探针，原样使用。每个 token 是常数行（小整数），故 kernel 实际实现的 combine 乘子可精确反解为 `c_t = combined[t,0] / x_t`，期望值是该 token 的去重目标 rank 数——与 mori 自己的 `check_combine_result`（`input × unique_pes`）语义一致。dispatch 另用 all-gather 路由 oracle 独立校验。

**两节点 world=16**（16 rank 汇总错误 token 数，均 `RESULT: FAIL`）：

| T | run1 | run2 | 报告人 |
|---|---|---|---|
| 16 | 10 | clean | clean |
| 32 | 8 | 2 | 5 |
| 128 | 31 | 21 | 17 |
| 256 | 29 | 37 | 35 |
| 512 | **85** | **107** | **83** |

量级与趋势均与报告人一致：T≤16 边缘，T≥256 每次必现。

**单节点 world=8 对照**：T=16..512 全部 clean（`PASS`）。这条把嫌疑收窄到跨节点路径，也说明 combine kernel 在纯 XGMI 下正常。

**失败特征**：dispatch 累计 240/240 次精确；损坏仅在 combine 输出；`dropped`（远端分量整体缺失）与 `polluted`（混入其他 token 的 payload）两类并存。

---

## 二、损坏 token 走的是哪条路？

原探针用均匀随机路由，几乎每个 token 都同时有本地和远端目标，**无法区分**损坏是跟着网线走、还是也砸到了从未离开本节点的 token。`repro_475_path_attribution.py` 构造路由让每个 token 归属已知类别：`local`（topk 全在本节点，不碰 RDMA）、`remote`（全在对端，每份 partial 都过 RDMA）、`mixed`（各一半）、`uniform`（报告人原始路由，同批次对照）。

T=512、4 轮、dispatch 0 次不匹配：

| 类别 | tokens | bad | dropped | polluted | other |
|---|---|---|---|---|---|
| **local** | 8192 | 5074 | **0** | 3221 | 1853 |
| **mixed** | 8192 | 3198 | **3198** | 0 | 0 |
| **remote** | 8192 | 3178 | **3178** | 0 | 0 |
| uniform | 8192 | 3257 | 3237 | 17 | 3 |

四次独立运行（不同 T、不同轮数）结果一致，分离**零例外**：含跨节点分量的 token（mixed/remote）失败形态 100% 是 `dropped`，从不 polluted；纯本地 token 0 例 `dropped`。后者在判据上是必然的（本地 token 的 `local_only == m`），但实质含义是：**这些从未走过 RDMA 的 token 也被损坏了，形态是"被污染"而非"丢失"**。

**结论：混合，两侧形态不同**——跨节点 token 丢失远端贡献，纯本地 token 被殃及并被污染。损坏不是单纯发生在网线上：本地与远端流量共享同一个 combine kernel、同一套 chunk 记账和缓冲区，记账一旦被破坏两者一起受害。

**佐证**：原探针只查第 0 列，补做全列检查（T=512）得 `col0=98 anycol=98 missed=0`，`mean_bad_frac_of_row=1.000`——既没有漏检，且每个坏 token 是整行 7168 元素统一乘错同一系数。这排除了 payload 撕裂与元素级 DMA 竞争，指向 chunk 级元数据/完成标志被误读。

> **该探针有一处未解偏差**：`uniform` 类与报告人脚本逐字同构，却稳定报 ~40%，而报告人脚本在同 T、同 commit、同机器、无 warmup 下只有 ~1.3%，差 30 倍。已排除 dispatch 正确性、接收容量溢出（`max_total_recv_tokens=0` 不设限）、路由构造、bf16 精度、`cls` 张量长度、warmup ladder、调用参数差异，根因未定位。**故上表绝对百分比不可用作损坏率**；形态分离是同一次运行内的相对比较，不受影响。

---

## 三、已排除的因素

| 假设 | 实验 | 结果 |
|---|---|---|
| **ionic 驱动版本过旧** | 新旧混合栈 | **排除**（见下） |
| **SDMA 路径** | `MORI_ENABLE_SDMA=0` ×2 | **排除**（见下） |
| fabric / 裸 RDMA 故障 | `ib_write_bw` 跨节点 | 排除（352–368 Gb/s） |
| dispatch 侧损坏 | all-gather 路由 oracle | 排除（240/240 精确） |
| 单节点同 kernel | world=8, T=16..512 | 排除（全 clean） |
| 只查第 0 列导致漏检 | 全列检查 | 排除（`missed=0`） |

**驱动版本**。报告人的头号假设是"ionic `25.11.1.001` → `26.03.3.001` 是唯一差异，新驱动可能就是修复"。用一台新栈（n06：ionic `26.03.3.001` / pds a-77 / kernel `6.8.0-124`）与一台旧栈（n08：`25.11.1.001` / a-45 / `6.8.0-84`）组混合栈直接检验：

| T | 旧栈 n08+n09 | 混合栈 n06+n08 |
|---|---|---|
| 128 | 31 / 21 | 23 / 7 |
| 256 | 29 / 37 | 30 / 25 |
| 512 | 85 / 107 | **85 / 76** |

损坏率无下降，且错误两侧均匀分布（T=512 新驱动侧每 rank 2–7 个错，旧驱动侧 3–9 个）。若旧驱动是根因，含新驱动的一侧应明显更干净。严格说尚非终判——混合栈里每条跨节点 QP 仍有一端是旧驱动，"两端都新才修复"未被排除——但结合对称分布，驱动版本作为根因已相当可疑。

**SDMA**。`MORI_ENABLE_SDMA=0` 两次运行：T=128 得 15/24、T=256 得 31/38、T=512 得 **86/75**，落在 SDMA=1 的波动范围内；错误构成同样不变（T=512 dropped 53/44 → 54/42，polluted 24/23 → 21/24）。符合预期：该开关只影响节点内传输选择，而损坏在跨节点路径上。

---

## 四、与 kernel 层分析的一处出入

`bugfix/v1_kernel` 分支 `report.md` §5.1 断言该 bug「**无法靠时序或压力自然复现**……必须主动构造 flag 尚未到达的状态」。

本报告实测与此不符：在 `0d05a4d2` 上、零注入、仅靠加大 token 数即可稳定复现，且损坏率**明确随负载增长**（T=16 边缘 → T=512 每次 75–107 个错），与报告人在 issue 中的描述一致。

两者不必然矛盾：可能是那份分析描述的编译器/缓存机制之外还有一条与规模相关的路径，也可能该机制本身的命中率就与 chunk 数正相关。区分需在真机场景下开 `MORI_DEBUG_COMBINE_TRACE` 观察实际 give-up 事件，尚未进行。在此之前，「必须注入才能复现」不宜作为结论使用——它会让人误以为线上不会自然发生，而实测表明会。

---

## 五、复现方法

| 文件 | 作用 |
|---|---|
| `repro_475_combine_probe.py` | 报告人的原始探针，扫 T=16..512 给出 PASS/FAIL |
| `repro_475_path_attribution.py` | 本报告新增，按传输路径归类损坏（§2） |
| `repro_475_launch_2node.sh` | 两节点启动器 |

```bash
./repro_475_launch_2node.sh                                    # combine 探针，T=16..512
PROBE=path ./repro_475_launch_2node.sh --rounds 4 --tokens 512 # 路径归因
SDMA=0 ./repro_475_launch_2node.sh                             # 关掉 SDMA
torchrun --standalone --nproc_per_node=8 repro_475_combine_probe.py  # 单节点对照，预期 PASS
```

换机器需覆盖 `NODE1` / `MASTER` / `IFACE` / `RDMA_N0` / `RDMA_N1`；`LOGDIR` 必须两节点都可见。

**`ionic_N` → rail 映射两节点不一致**是最容易踩的坑。mori 按设备**索引**配对 peer，而枚举顺序由 PCIe 决定，各机器不同（如 benic1p1 在 n08 上是 `ionic_0`，在 n06/n09 上都是 `ionic_2`）。不校正则每个 rank 被配到对端不同 rail 的 NIC，QP 永远建不起来，表现为**静默挂死在 `shmem_torch_process_group_init`**，无任何报错——极易误判为"复现不出来"。修法是两端各按 rail 顺序显式指定 `MORI_RDMA_DEVICES`（保留列表顺序），映射用 `rdma link | awk '{print $2, $NF}'` 在两台机器上分别生成。启动脚本已内置默认值。

其他注意事项：

- **需要真正两台物理机**。`test_dispatch_combine_internode_v1.py` 只参数化 `world_size=8, gpu_per_node=8`（走 RDMA 的 `gpu_per_node=4` 被注释掉），`max_num_inp_token_per_rank` 上限 128——跨节点路径与 T>128 区间 CI 从未覆盖，本来就漏不出来（issue latent bug 7）。
- **T 需 ≥256** 才稳定可见。
- **不要用 `pkill -9` 清理残留 rank**：容器 PID 1 可能被匹配到一并杀掉；即便没有，强杀会留下 shmem/JIT 状态使后续运行卡在模块加载。用 `docker restart`。
- 启动器已在两节点**串行预编译** kernel，规避多 rank 并发冷启动 JIT 争抢共享 cache 导致的卡死（issue latent bug 5）。
- **容器内 libionic 必须与主机内核驱动匹配**（顺带回答了 issue 中"固件/pds/nicctl 是否要跟驱动一起升"的提问：用户态库必须一起升）。主机升到 a-77 后容器仍用镜像自带的 `libionic 54.0-149`（配 a-45），则容器内一个 RDMA 设备都看不到：`Driver ionic does not support the kernel ABI of 1 (supports 4 to 4)`。需从对应 AINIC bundle 装 `libionic1` / `libionic-dev` `54.0-187-1`。

---

## 六、状态

**已确立**：干净 main 上无需注入即可稳定复现；dispatch 正确，损坏限于跨节点 combine 返回路径；跨节点 token 丢失远端分量、纯本地 token 被污染；坏 token 整行统一乘错系数；驱动版本与 SDMA 已排除。

**未解**：§2 归因探针的 30 倍偏差（绝对数字不可信，形态结论不受影响）；一对**两端都是新驱动**的节点尚未测试，"新驱动是否为修复"缺终判；§4 的出入未澄清。
