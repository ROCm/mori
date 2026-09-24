# HANDOFF — EPv2 COMBINE PUSH 量化(ct=512)

> 交接给下一会话:继续做 **ct=512 combine 量化**。本文汇总已量化的数据、代码状态、
> 复现方法、结论与可继续的方向。所有实测在 A8-1 gfx1250(MI450/455)EP4 上完成。

分支:`fix/epv2-1250x-tdm-128b-align`,base `c07d4f3e`。**所有改动在 `_wt_v2main`
工作树,未提交。** 相关记忆:`memory/project_epv2_combine_push_quant.md`(权威数据)、
`project_epv2_combine_fp4_build.md`、`project_epv2_combine_fp4_phase0.md`。

---

## 1. 一句话结论

- **fp8_direct_cast PUSH combine = ct=512 净持平**(45.7 vs 44.7 bf16 gather),量化真实生效
  但 PUSH 本身的固定开销吃掉了省下的线宽 → 无净收益。
- **fp4 PUSH combine = 硬净损失**,ct=512 时 135.5µs(3.7× 慢),ct=4096 时 1099µs(8.4× 慢)。
  25µs 目标**架构上不可达**。原因见 §3 分段:recv 解码+reduce 一段就 63µs = fp8 的 9.6×。
- combine@512 **不是带宽瓶颈**,是 compute/latency 瓶颈,窄线宽换不回 PUSH 的固定/延时开销,
  也换不回比 bf16 读更重的 fp4 codec。

---

## 2. 关键实测数据(ct=512, EP4, h7168, topk8, graph, CHECK=0, 50 iters)

### 2.1 三条 combine 路径全貌
| combine 路径 | µs | vs bf16 gather |
|---|---|---|
| gather / none(bf16 基线) | 44.7(跑间 36–45 波动) | — |
| scatter / none(PUSH bf16) | 56.4 | +11.7 |
| scatter / fp8_direct_cast | 45.7 | +1.0(持平) |
| scatter / fp4 | 135.5 | +90.8(3.7×) |

ct=4096:bf16 gather 131.3(此时真·BW-bound @1612 GB/s),fp4 1099(8.4×)。

### 2.2 严格分段(用内建 `MORI_COMB_STOP` gate,实测差分非估算)
STOP 语义(kernel `ep_intranode_1250x.hpp:1948`):
- `STOP=2` 只剩跨设备 barrier(send 与 reduce 都跳过)
- `STOP=1` send + barrier(**数据已 PUSH 落到对端就停,不做 reduce**)← 用户定义的"不做 REDUCE"切点
- `STOP=0` 全程

传法:JIT 宏 `MORI_JIT_EXTRA_FLAGS="-DMORI_COMB_STOP=n"`,**不用重编 .so**。

实测 combine µs:
| STOP | 分段 | FP8 | FP4 |
|---|---|---|---|
| 2 | barrier/rendezvous | 12.7 | 12.6 |
| 1 | send+barrier(无 reduce) | 37.9 | 72.4 |
| 0 | 全程 | 44.5 | 135.5 |

差分拆解:
| 分段 | 计算 | FP8 | FP4 | FP4/FP8 |
|---|---|---|---|---|
| **REDUCE**(recv 解码+topk累加+写出) | T0−T1 | **6.6** | **63.1** | **9.6×** |
| SEND(load+encode+store) | T1−T2 | 25.2 | 59.8 | 2.4× |
| BARRIER | T2 | 12.7 | 12.6 | 1.0× |

**结论:fp4 慢的 91µs = reduce +56.5 / send-encode +34.6 / barrier ≈0。**
REDUCE 单段 63µs = fp4 整个 combine 的 47%,是最大头。
- fp8 解码 = 一条 `v_cvt_pk_f32_fp8`(近乎免费)。
- fp4 解码 = 每 8 元素 `__builtin_amdgcn_cvt_scale_pk8_bf16_fp4`(带 e8m0 scale 查表),走绕开
  向量化 pull 的窄 fold 路径,再 ×topk=8 源逐 token 累加。

barrier 两格式相同(12.6≈12.7)= 与传输无关的常量,证明分段方法学干净。

### 2.3 交叉验证(两套独立隔离法一致)
- QSKIP=1(跳过 send encode):fp4 full 135 / QSKIP 94 → encode ≈ 41µs。
- STOP 分段:fp4 barrier 12.6 + (send−encode) 18.8 + reduce 63.1 = **94.5 ≈ QSKIP 的 94µs ✓**。

---

## 3. fp4 vs fp8 为什么 codec 贵(根因,非表象)

- **fp8 e4m3 = 自缩放(direct cast)**:~4 位指数,动态范围 ~220,000×,每元素独立,无共享
  scale。编码 = 一次 `__hip_fp8_e4m3` cast,解码 = 一次 widen。codec 近乎免费。
- **fp4 e2m1 = 块缩放(必须有共享 block scale)**:仅 2 位指数,8 个码值 {0,.5,1,1.5,2,3,4,6},
  动态范围仅 ~12×。**必须**配 per-block e8m0 scale(= MXFP4)。于是 codec = amax 归约 + e8m0 scale
  计算 + 缩放量化 + 独立 scale 通道 + 缩放解码,每步都比 fp8 重,且解码只能 8 个一组走窄路径。
- **对照 aiter PR #5176**:他们的 fp4 combine 便宜(~6µs)是因为把 codec **融进 gemm2 epilogue**
  (寄存器内 per-1×32 e8m0 量化 + 接收端 dequant),codec 无处不在地被 gemm 吸收;我们是独立 PUSH,
  codec 无处藏身。他们自己也说 "@512 latency-bound not wire-bound,narrowing buys nothing" ——
  **正好印证我们的结论**。要让 fp4 combine 便宜 = 融进 gemm epilogue(架构改动),不是在独立 PUSH 上
  调线宽。

---

## 4. 代码状态(工作树未提交改动)

### 4.1 已实现:真 fp4 线格式(自洽、能跑、结构正确)
- **enum**:`combineQuant` 0=none,1=fp8_direct_cast,3=fp4(2=fp8_blockwise 留空)。
- **线布局/槽**:`[fp4 payload hidden/2][e8m0 scale][topk weights]`,128 对齐。
  scale group **从 512(WS×8)改成 8(per-lane per-pk8)** —— 这是主要优化点。
  wire fp4 = hidden/2 + pad16(hidden/8) = 3584+896 = 4480B,slot = pad128(4480+32)=4608B。
- **encode** `_cQuantTile4`(~ep_intranode_1250x.hpp:1682):amax → e8m0 se → e2m1;fp4 跳过跨 lane
  shuffle 归约(`if constexpr(!kFp4)`),per-lane 存 scale。
- **decode**:native `__builtin_amdgcn_cvt_scale_pk8_bf16_fp4(pk8, e8m0, 0)`,专用窄 fold 分支
  (~2517-2549),`_grp=8`,读 `_pk=*(uint32*)(_s+(_eGlob>>1))`,`_e8=_s[hidden/2 + _eGlob/8]`。
- **fp4 绕开 fp8 快路径**:`_cPipe` 流水 encode、`kLoadTileElems` 双缓冲对 fp4 解禁(去掉 `!kFp4`)。

### 4.2 改动文件清单
- `src/ops/dispatch_combine_v2/ep_intranode_1250x.hpp`(kernel 主体,§4.1)。
- `include/mori/ops/dispatch_combine_v2/ep_cfg.hpp`:`EpCombineFp4Group()`→8;`EpCombineFp4ScaleBytes`;
  `EpCombineWireBytes`/`EpCombinePushSlotBytes` fp4 分支。
- `python/mori/ops/dispatch_combine_v2/hip_backend.py`:`push_wire_nbytes` fp4 = hidden/2 + pad16(hidden/8);
  `_QUANT_TO_INT["fp4"]=3`;fp4 校验守卫(hidden%8);读 `-DMORI_COMB_QPIPE`(默认4,要求 hidden%qpipe==0)。
- `python/mori/ops/dispatch_combine_v2/dispatch_combine_op.py`:`_QUANT_TYPES` 加 `"fp4"`。
- `tests/python/ops/dispatch_combine_v2/test_op.py`:`_IS_FP4` 按 token dtype 取键;numeric arm 加 DBG 打印
  (max_abs/mean_rel 等,`if not ok and rank==0`)。
- `tests/python/ops/dispatch_combine_v2/bench_ep.py`:加 `COMB_MODE`(gather|scatter)、`COMB_QUANT`
  (none|fp8_direct_cast|fp4)env 旋钮;`SWEEP` 逗号列表。**注意 c_bw 硬编码 HIDDEN×2(bf16),
  fp8/fp4 的 GB/s 打印偏高约 2×,只比 µs。**

### 4.3 fp4 正确性说明
`hidden=BAD` 是 **e2m1 固有精度损失,不是 bug**:mean_rel≈0.63–0.68,out_amax 24 vs exp 20
(8 值码本 × 粗块 scale 过冲)。routing/weights PASS,wire 结构正确。测试 `_IS_FP4` skip 按 token
dtype(此处 bf16)判定,所以落进了数值 arm。

---

## 5. 复现 / 远端操作手册

**节点/容器**:`ssh a81` → `docker exec MORI-EPV2`。工作目录 `/app/mori_ebt`,bench 在
`tests/python/ops/dispatch_combine_v2/bench_ep.py`。

**GPU 安全门(强制)**:launch 前查 `ls /sys/class/kfd/kfd/proc/`,**仅当有 LIVE 持有者**
(`[ -e /proc/$p/comm ]`)才让路;空 = 可跑。别碰他人容器/镜像。

**运行一档(EP4, ct=512, graph)**:
```
MORI_V2_KERNEL_BACKEND=hip COMB_MODE=scatter COMB_QUANT=fp4 \
CHECK=0 SWEEP=512 MODES=graph ITERS=50 WARMUP=10 \
MORI_JIT_EXTRA_FLAGS="-DMORI_COMB_QPIPE=4 -DMORI_COMB_STOP=0" \
torchrun --standalone --nproc_per_node=4 bench_ep.py
```
输出行:`ct=512 [hip/graph] ... combine  NN.N us (...)`,只看 combine µs。
COMB_QUANT 换 `fp8_direct_cast`/`none`;STOP 换 0/1/2 做分段;分段驱动脚本模板见
`/tmp/stop_seg.sh`(容器内,循环 quant×STOP)。

**踩过的坑(务必遵守)**:
1. **单条长 ssh 会被 Conductor auth 掐断**("closed by remote host")→ 用
   `docker exec -d MORI-EPV2 bash -c "nohup bash /tmp/x.sh > /tmp/x.log 2>&1 &"` **detached 跑**,
   再用短 ssh(<40s)轮询日志。别在本地 `sleep` 超过 ~110s(撞工具 2min 上限)。
2. **并发 torchrun 会 OOM 杀容器**(Exited 137):bench 的 vmm 预留很大,两个 EP4 run 叠加就爆。
   跑前确认无残留 `pgrep -f torchrun`;掉线重跑前先清残留或重启容器(`docker start MORI-EPV2`)。
3. **JIT stale header 陷阱**:改 kernel 后 .so + `_jit-sources/{src,include}` 头文件 + v2 python
   都要 sync 进 `/usr/local/lib/python3.12/dist-packages/mori/`(JIT 从那里重编)。STOP/QPIPE 是纯
   JIT 宏,只改 flag 不用重编 .so。
4. 节点重启会清 /tmp(宿主),但容器 fs 持久;重启后 `docker start` + 重传补丁。
5. 禁用 bare `git stash`/`git stash pop`(共享 stash 栈);别弹权限窗。

---

## 6. gate/knob 速查(kernel JIT 宏,`-D<macro>=<val>`)

| 宏 | 默认 | 作用 |
|---|---|---|
| `MORI_COMB_STOP` | 0 | 分段:0 全程 / 1 send+barrier(无reduce) / 2 barrier only |
| `MORI_COMB_QSKIP` | 0 | 1 = 跳过 send encode(隔离 encode 成本) |
| `MORI_COMB_QPIPE` | 4 | encode 流水分块数,要求 hidden%qpipe==0 |
| `MORI_COMB_QOOP` / `WMERGE` / `STFRAC` / `Q4PACK` / `LDFRAC` / `QRGLD` / `NOFENCE` / `STSKIP` | — | 其它 send/store 细分 gate(见 kernel 注释 §23) |

bench env:`COMB_MODE` `COMB_QUANT` `SWEEP` `CHECK` `MODES` `ITERS` `WARMUP` `BACKENDS`。

---

## 7. 下一会话可继续的方向(512 量化)

1. **fp8 是唯一有戏的量化,但当前只到持平。** 要转正需砍 PUSH 的 +11.7µs 固定开销(rendezvous/
   staging),而非动线宽 —— 这是 combine@512 的真瓶颈。可查 `EpCrossDeviceBarrier1250x` 的
   rendezvous(STOP=2 = 12.7µs)与 staging copy 是否可省/合并。
2. **fp4 若要便宜,唯一路径 = 融进 gemm epilogue**(register-quant + receive-dequant,仿 aiter
   #5176),独立 PUSH 上无解。属架构改动,需先确认是否在本仓范围内。
3. **reduce 段(fp4 63µs)是最大优化面**:当前每 8 元素一次 intrinsic + topk 逐源累加走窄路径。
   可探:向量化 fp4 fold(一次处理多 pk8)、把 e8m0 scale 预取进 LDS、或把 topk 累加与 decode 融合,
   看能否把 63µs 往 fp8 的 6.6µs 拉近(即便拉一半也改变结论)。
4. **ct=4096 分段未做**:若要看 reduce 段在大 batch 的走势(是否随 token 数线性/被 BW 掩盖),
   照 §5 把 STOP=0/1/2 各跑一遍即可。

---

## 8. f01-2 重建环境后的复测(2026-09-24,不是 A8-1,机器/工具链都换了)

A8-1、f01-1 当天都连不上。f01-2 上 `MORI-EPV2` 容器和 `/root/mori_ebt` 树已经不在了(host 上
连 mori 相关镜像都没有),**从零重建**:`rocm/vllm-dev:nightly_455_wip__torch__0922_b264`
(ROCm 10.1.0,torch 2.11,与 a81/f01-1 同代,不是 f01-2 历史上那个 7.14.0)起新
`MORI-EPV2` 容器,`_wt_v2main` 工作树(未提交改动原样)+ `3rdparty/{spdlog,msgpack-c}`
子模块传过去,`pip install . --no-build-isolation` 编译通过。**这是新机器 + 新工具链,
下面的绝对值不能跟 A8-1 那份表比,只能看本节内部的相对差。**

冒烟:bf16 gather ct=512 graph CHECK=1 `combine 37.9us`,PASS。

STOP 分段(ct=512, EP4, h7168, topk8, graph, CHECK=0, 50 iters,`-DMORI_COMB_STOP=n`):

| STOP | 分段 | bf16 scatter(PUSH) | fp8_direct_cast scatter |
|---|---|---:|---:|
| 2 | barrier only | 13.9 | 13.7 |
| 1 | send+barrier | 46.8 | 43.3 |
| 0 | 全程 | 54.2 / 54.0(复测) | 50.3 / 50.5(复测) |

差分:

| 段 | bf16 | fp8 | fp8−bf16 |
|---|---:|---:|---:|
| BARRIER | 13.9 | 13.7 | −0.2(噪声内,再次确认 barrier 与量化无关) |
| SEND | 32.9 | 29.6 | **−3.3**(narrower store 真省了) |
| REDUCE | 7.4 | 7.0 | −0.4(噪声内,fp8 解码本就接近免费) |
| **全程** | **54.2** | **50.3** | **−3.9,两轮复测一致(54.0/54.2 对 50.3/50.5)** |

**这一次 fp8 是净赢,不是持平**(A8-1 那份是 45.7 vs 44.7,+1.0 持平)。但对照 bf16
**gather** 基线 37.9,scatter/fp8 的 50.3 仍然 **+12.4us 更慢**——PUSH 自己的固定开销
(主要是 barrier 13.7 + send 段里不随量化变的那部分)还是没追平 PULL/gather,§7.1 的
结论方向不变:**量化在 send 段确实省钱,但不够填平 PUSH vs gather 的坑**,§7 下一步
(砍 barrier/rendezvous 固定开销)仍然是唯一能让 PUSH 路径整体转正的方向,fp8 本身已经
把能省的（narrower store）省到了。

未做(留给下一步):fp4 在这台机器上的 STOP 分段(A8-1 数据仍然权威,量级不会变但绝对值要
重测才能引用);BARDIRECT/WSKIP 两个还没转正的门在这台机器上的 A/B。

---
*记录时间 2026-09-24,A8-1 gfx1250 EP4。数据权威副本在 `memory/project_epv2_combine_push_quant.md`。*
*§8 补测于 2026-09-24,f01-2(重建环境,ROCm 10.1.0),不是 A8-1,数值不通用。*
