# UMBP UALink 优化的实现与验证

基于 `feat/umbp-ualink-kv-sharing` 的 `34d603b`，2026-11-12 在 j07-16 / j07-14
双向验证。机器使用 gfx1250、HIP 7.18 / ROCm 10.3，Fabric link_type 为 UALoE。
本次范围限定为 UMBP。推理引擎原生 KV 分配与 attention 读取远端 KV 不在本次验收范围。
以下区分默认优化、可选路径，以及评估后未启用的方案。

## 已接入的实现

1. **提前合并 GET 连续页片段**：在构造昂贵的端点描述符之前合并；保留相同注册身份、
   源/目标连续性、每 key 的失败 tag、32 bit segment 长度限制和原有 completion pin。
2. **紧凑 range staging**：请求携带规范化的区间并集；DRAM/SSD 的 owner HBM staging
   只复制有效区间，并返回 object offset 到 packed offset 的映射。每段起点保留原始的
   offset modulo 16，避免压缩布局使原本可向量化的拷贝退化。空洞和 padding 不构成可读覆盖。
3. **复用 staging 内容**：在原有 arena 容量内保留不可变内容，按 backend 内容版本、
   对象大小、规范化覆盖和批位置区分。相同请求并发准备只执行一次；release 释放 reader
   引用，缓存继续持有页面；LRU 回收缓存引用时不撤销尚未完成的读者引用。
   Cache key 不使用可被回收复用的页面地址。Clear、淘汰后同名 key 的新内容有不同版本。
4. **SSD 命中绕过读盘**：先查询实时 size/version 元数据；完整批缓存命中可跳过 SSD
   host staging 和 H2D。混合批或不完整覆盖会走正常 resolve。当前缓存单位是整组请求，
   不会把不同批次或不同覆盖自动拼接成命中。
5. **非对齐 kernel**：共同对齐的指针使用短 prefix、16/8/4/2 字节主体和 tail；两端没有
   共同 2 字节对齐条件时保留 byte copy。所有访问严格限制在请求范围内。
6. **可调并行度**：`UMBP_FABRIC_MAX_BLOCKS_PER_SEGMENT` 允许 1–128，默认仍为 8。
   实际 block 数还受 segment 长度限制。保留默认值是因为不同大小和布局的最佳值不同。

7. **Owner-push（可选）**：reader 发布已注册、可通过 FABRIC 导出的最终 GPU 目标，owner
   通过 GPU 可访问的 host alias 直接写入。DRAM 省掉 owner H2D；SSD 保留读盘到 host，
   省掉 owner H2D。仍使用统一 TransferEngine；最多缓存 64 个 host alias 的注册描述符，
   不缓存 host 内容，也不改变 KV placement。请求的目标、区间和 engine descriptor 纳入
   原有操作记录；并发或丢失回复重试只能复用同一次完成结果。确认完成后释放源 pin，
   记录继续保留以拒绝重复写入。未确认完成时 reader 保留目标、owner 保留源和传输资源。
   旧 owner、普通 hipMalloc 目标或尚无 GPU host alias 时使用 pull；BUSY 批也在写入前
   整批退回原来的重试路径。超时本身不允许回退或复用目标。
8. **按 key 分块流水线（可选）**：上一组 pull 与下一组 owner 准备重叠。每组单独持有
   源 pin、目标 admission 和完成事件；聚合 handle 保持结果顺序及整次调用的目标重叠检查。
   在途组数有界，Submit 可能先等待较早的组再继续提交；不会暴露尚未完成的 chunk。
   它可降低所需 staging 容量，但增加路由和 acquire/release RPC，不能假定带宽提高。
9. **设备描述符复用（可选）**：在已经完成的 batch slot 中匹配源/目标地址和长度，
   命中时省去本次描述符上传。只复用已有 slot，不增加 payload 缓存；当前注册和读 pin
   仍须重新验证。host 描述符在同一个 mutex 下发布和比较，未完成的 slot 不可覆盖。

Owner-push、分块流水线和描述符复用均默认关闭；当前测试没有支持将它们作为通用默认值。
使用以下环境变量分别控制（进程初始化前设置）：

| 配置 | 默认 | 用途 |
| --- | --- | --- |
| `UMBP_FABRIC_OWNER_PUSH` | `0` | 两端设为 `1`；reader 目标须为可导出 VMM，否则使用 pull |
| `UMBP_FABRIC_GET_CHUNK_BYTES` | `0` | reader 每组请求字节预算；0 为不分组，单个 key 不拆分 |
| `UMBP_FABRIC_PIPELINE_DEPTH` | `2` | reader 在途组窗口，1–32，并受现有 inflight admission 限制 |
| `UMBP_FABRIC_CACHE_DESCRIPTORS` | `0` | `1` 允许复用已完成的设备描述符 |

`UMBP_FABRIC_READ_STAGING_CACHE=0` 可关闭内容缓存；默认开启。
`UMBP_FABRIC_READ_STAGING_BYTES=0` 仍关闭整个 staging 功能。容量、设备和 placement
语义与原分支相同：staging 缓存不发布新的 KV owner，也不增加 reader 本地 KV 副本。
HBM 路由请求省略 staging 专用 ranges 元数据；路由过期而 owner 实际返回 CPU 存储时，
仍可安全地退回 enclosing extent staging。旧 owner 忽略新增 ranges 字段并返回 enclosing extent；新 owner 只对请求了 ranges 的
reader 发布 packed coverage，因此可以逐步升级。

## 拷贝次数和字节

| 来源与状态 | 实际路径 |
| --- | --- |
| HBM | owner HBM → reader 最终 HBM |
| 冷 DRAM | owner DRAM → owner HBM staging → reader 最终 HBM |
| 冷 SSD | SSD → owner host staging → owner HBM staging → reader 最终 HBM |
| DRAM/SSD 内容缓存命中 | owner HBM cache → reader 最终 HBM |
| 可选 owner-push：DRAM | owner DRAM → reader 最终 HBM |
| 可选 owner-push：SSD | SSD → owner host staging → reader 最终 HBM |

冷读的优化是减少 H2D 字节和 arena 占用；热命中省掉本次 H2D，SSD 热命中还省掉本次读盘。
本次 SSD 冷读仍按对齐后的 enclosing extent 进行 I/O，并没有声称稀疏 SSD I/O 本身已经
去掉空洞。对于 `object = 8P + 128`、三个区间总长 `2P + 313` 的基准，H2D 从接近
`8P` 降到 `2P + 313`；staging 占用还包括每段最多 15 字节 padding 和页尾对齐。

## 端到端结果

下表是有效请求字节除以全部 warm GET 时间，包含路由、acquire/release、规划、提交和等待；
不是物理链路计数。1 GiB 表使用固定 64 KiB 存储页，32 个对象，每对象三个区间，
每点 1 次冷读 + 100 次热读。原分支 baseline 使用同样的固定页基准和计时口径。集成版数值取所有 UMBP 改动后的最终四点复验（`20261112T132732Z-large-final`）；此前
完整 18 点 A/B 和中间版本结果也保存在 CSV 中。可选 owner-push、分块、描述符复用均关闭。

| 约 1 GiB aligned GET | 原分支 | 集成版，8 blocks | 集成版，128 blocks |
| --- | ---: | ---: | ---: |
| 16→14 | 88.98 GB/s | 262.15 GB/s | 256.08 GB/s |
| 14→16 | 90.14 GB/s | 268.64 GB/s | 276.77 GB/s |

| 约 4.2 MB aligned 热读 | 原分支 | 紧凑 staging + 内容缓存 |
| --- | ---: | ---: |
| DRAM，16→14 | 3.28 GB/s | 6.07 GB/s |
| DRAM，14→16 | 3.58 GB/s | 5.35 GB/s |
| SSD，16→14 | 0.55 GB/s | 5.83 GB/s |
| SSD，14→16 | 0.40 GB/s | 6.13 GB/s |

缓存表为 kernel 向量化之前的隔离 A/B，每点 1 次冷读 + 30 次热读；最终集成版另外完成
三种 tier 的跨 tray 正确性回归。缓存收益依赖重复读取：不能把热命中提升等同于冷 SSD
带宽提升。首次导入约 80–105 ms 的开销在原版和集成版均存在。

64 MiB 非对齐 GET 使用 128 blocks 测到 44.15–54.45 GB/s，原分支 20.27–21.72 GB/s；
8 blocks 为 20.30–24.72 GB/s，部分方向有回退。小请求由控制路径和 RPC 占比较高，
应根据实际 workload 选择配置，而不是把大请求的最佳参数当作通用默认值。
补充三轮交替顺序、每轮 200 次 warm GET 的 4.2 MB HBM 复测：16→14 汇总为
4.17→5.08 GB/s，14→16 为 7.23→7.14 GB/s。回退到接近基线后仍有明显单轮波动，
不宣称小请求有稳定吞吐提升。第一次与本机回归重叠的轮次排除，并补跑一轮。
完整冷/热延迟和吞吐见 [CSV](validation/2026-11-12-optimization-results.csv)。

另有同模非对齐的独立 kernel A/B：64 MiB / 96 段、相同 8 blocks、两方向分别从
33.69 / 33.80 GB/s 提升到 428.26 / 425.89 GB/s。128 blocks 为约 480 GB/s。
这是 HIP event 覆盖重复 kernel launch 的数据面结果，不含 UMBP RPC；不能直接与上表 GET 对比。

## 可选路径的最终 A/B

以下使用同一个最终 runtime、相同 reader VMM 目标和固定 64 KiB 存储页；32 个对象，
每次请求 67,118,880 字节，1 次冷连接 + 40 次 warm GET。关闭内容缓存的前三列每次都重新
准备源数据；SSD 每次读盘，表中的 warm 仅指连接/import 已热，不能理解成 HBM 缓存命中。
单元格为有效请求 GB/s。

| owner→reader / 来源 | pull，关闭内容缓存 | owner-push | 分块 pull，8 MiB/2 组 | 内容缓存 pull |
| --- | ---: | ---: | ---: | ---: |
| 16→14 / DRAM | 17.54 | 30.58 | 4.68 | 53.41 |
| 16→14 / SSD | 1.54 | 1.56 | 1.32 | 54.20 |
| 14→16 / DRAM | 21.05 | 32.71 | 7.55 | 81.11 |
| 14→16 / SSD | 2.05 | 2.12 | 1.58 | 72.41 |

Owner-push 对本次 64 MiB 冷 DRAM 有明确收益；SSD 仍受读盘限制，差异小，不宣称稳定的
SSD 带宽提升。热内容复用优先使用默认 HBM staging 缓存。先前 4.2 MB 实测也未显示
owner-push 的通用优势，因此该路径仅在两端显式开启并满足导出/映射条件时使用。

分块 A/B 的 RPC 开销超过重叠收益，但容量测试通过：16 MiB staging arena 下，约 64 MiB
非对齐请求的整批模式被拒绝；8 MiB key 分组、深度 2、关闭内容缓存后，双向各完成
11 次 GET 并校验全部数据。它支持更小的 staging 预留，当前不作为带宽优化默认开启。

描述符复用同配置 A/B（每点 100 次 warm GET）为：4.2 MB，16→14 的 5.11→3.92、
14→16 的 7.64→6.96 GB/s；1 GiB 为 266.69→234.24、268.68→274.80 GB/s。
结果未显示一致收益，默认关闭。减少描述符上传不等于端到端吞吐必然提高。

## 验证内容与复现

- 最终 12 个相关 CTest 组通过：range map、batch codec、transfer engine、page backend、peer pool、
  fabric staging/push、SSD backend、SSD manager、singleflight、eviction、GPU direct ranges 和
  UALink tier sharing。后两组运行真实 GPU；2 项既有 host-gather 子测试因当前内嵌 kernel
  报 `device kernel image is invalid` 而跳过，冷 pull 保留 hipMemcpy 回退。没有将它们记为
  gather 验证通过。FABRIC 动态加载的 kernel 与 owner-push 均另外完成实际字节校验。
- 两台各完成 19 种长度 × 256 种 source/destination modulo-16 偏移 × 4 种 launch 方式；
  校验完整 allocation，包括 payload 和全部未写 guard。额外覆盖缓存准备失败时的并发等待和重试。
- Owner-push：双向 DRAM/SSD 首轮与注册复用版各 40 次 GET；全部 32 个 GPU 配对、
  160 次 GET，通过。最终版另在 64 KiB owner staging 下完成双向 DRAM/SSD 的
  40 次 GET，每次请求 4.2 MB，断言 owner-push 字节等于请求、reader pull 为零。覆盖源/目标越界、并发相同操作只提交一次、目标变化拒绝、release
  后重试拒绝、BUSY 批不写前缀、Clear 与未知完成的资源保护。未用 microbenchmark 代替集成测试。
- 分块：双向 HBM/DRAM/SSD × aligned/unaligned，共 12 个用例、360 次 GET，通过；
  另有跨组目标重叠拒绝和跨线程 Shutdown/Wait 的 GPU 测试。
- 最终集成 runtime 的双向 HBM/DRAM/SSD、aligned/unaligned、stream wait/blocking wait：
  14 个用例、1,400 次 GET，通过。包含每 25 次重建 reader、冷导入、owner 和 Fabric 字节检查。
- 新旧版本混用：两方向 × DRAM/SSD × 旧 owner/旧 reader，共 8 个用例、80 次 GET，通过。
  另外显式开启 owner-push、使用可导出 reader 目标复测相同 8 个组合、80 次 GET，
  验证旧 owner 的能力协商回退和旧 reader 兼容，全部通过。
- 全 GPU 配对：双方向各 4 × 4 组合，共 32 个用例、160 次 GET，通过；双向并发
  2 个用例、400 次 GET，通过。
- 大小/并行度 A/B：18 个完整用例、1,818 次 GET，通过。一次 SSH 收尾超时的尝试保留原始日志，
  该点重跑后全角色正常退出，结果表只计入完整通过的运行。最终代码额外通过四点大请求
  复验、404 次 GET，及 12 个用于比较的小请求用例、2,412 次 GET。扩展实现之后再跑
  最终核心 1,400 次、版本兼容 80 次、大请求 404 次；可选路径 A/B 656 次、描述符 A/B
  808 次均通过。CSV 中 `final-*` 标识最终版本，之前的结果保留原实验标识。

构建使用仓库已有 `umbp_ualink_kv_driver`、`test_fabric_integration`、
`test_umbp_ualink_tier_sharing` 等目标。driver 新增 `--storage-page-bytes`，用于将存储页大小与请求几何分开；省略时保持原行为。
`--fabric-destination 1 --expect-owner-push 1` 分配 reader 可导出 VMM 并断言全部请求字节
由 owner-push 完成。日志单独列出 `owner_push_submitted_bytes`，不把它误报成 reader 的
本地提交计数；两种路径都要求完成与数据校验。

在本次 ROCm 工具链上，`hipcc --genco` 输出 offload bundle，`hipModuleLoad` 需要其中的原始 ELF。
构建后若 `file fabric_copy.hsaco` 未显示 ELF，先执行：

```bash
/opt/rocm/llvm/bin/clang-offload-bundler -unbundle -type=o \
  -targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
  -input=build-ualink/src/io/fabric_copy.hsaco -output=/tmp/fabric_copy.hsaco
export UMBP_FABRIC_KERNEL_PATH=/tmp/fabric_copy.hsaco
build-ualink/tests/cpp/test_fabric_integration --gpu-module "$UMBP_FABRIC_KERNEL_PATH"
ctest --test-dir build-ualink -R 'umbp_(fabric_read_staging|ualink_tier_sharing)' --output-on-failure
```

跨 tray 测试分别启动 master、producer、consumer，`--kernel` 指向各自部署的同一 ELF。
用 `--page-bytes 16777216 --storage-page-bytes 65536 --object-bytes 134217856 --keys 32`
复现大请求，consumer 使用 `--iterations 101 --verify-every 100 --cold-every 0 --async 0`。
更小的正确性测试应使用 `--verify-every 1`，并检查所有进程退出码。

文档检查：所有提交文件的 pre-commit 检查通过。本地 HTML 链接检查为 0 错误。
严格 Sphinx 构建存在 7 条基线文档警告；新增优化文档没有新增警告。

## 评估后保留的限制

- **GDS → FABRIC VMM**：两台均禁止 hipFile fallback 后实测。普通 hipMalloc 成功，
  10 次读取、fallback_reads=0；可导出 VMM、gpuDirectRDMACapable 和 host-access 变体
  均在 hipFileRead 返回 EINVAL。普通 hipMalloc 的 GDS 吞吐为 7.27 / 12.14 GB/s。
  当前安装栈不支持该组合，因此没有加入会失败的生产数据路径。
- **长期路由缓存**：本次保留每次 master 路由。`RouteGetResult` 只有节点、地址、tier
  和 size，owner incarnation/pool generation 也只描述单个 owner；当前协议没有覆盖
  跨 owner 的同 key 替换或迁移的缓存失效通知。因此没有用 TTL 或缓存页地址省略授权。
  已有 engine/buffer/import/session 缓存继续使用，新增描述符复用也不替代实时源 pin。
- **推理工作负载**：按用户确认不纳入本次 UMBP 验收，没有宣称模型 TTFT/TPOT 收益。
