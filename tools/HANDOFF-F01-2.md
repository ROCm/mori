# ctheliosr-1b114-f01-2 (gfx1250) — 交接

本文件替代原 `ep-partb-debug.mdc` 与 `output-terse.mdc`,开工前读一次即可。

## 1. 输出纪律

1. 只给数据表 + 一句结论。不解释、不复述、不铺垫、不道歉。
2. 不主动写文档/报告。注释只写硬约束。
3. 先动手,做完报数字。时间用 µs(2270 cyc/µs)。
4. 省 token:实验一律 `_ct_go.sh`(WAIT=1 一次出数),补读 `_ct_res.sh`,传文件 `_scp_ct.ps1`,跑脚本 `_send_ct.ps1`。
5. 每个配置只跑 1 次。源码没变别清 JIT 缓存。不为实验新建脚本,传 `EXP=`。
6. 读代码用 Grep 不整文件读。不 `git status` 全量、不列大目录、不 `cat` 日志。
7. PowerShell 不写内联多行命令。报错先修脚本,不原样重试。上下文过大时提醒开新会话。

## 2. 三条铁律

1. **不制造 hang**。debug 内核不加 grid barrier。grid-barrier hang 在本节点 `docker stop`/`--gpureset` 杀不掉,
 只能反复 `pkill -9 -x python3`。**跑 EP 时绝不并发任何 GPU 命令**(连 `rocm-smi` 都不行),单跑,
 跑前确认 VRAM 回落 ~175MB/卡。
2. **先把 dispatch 搞正确再跑 combine**。`acc_check --weightless` 全 PASS 后才接 combine,否则 combine
 读到错的 metadata/slot → 崩或 hang。
3. **改动只在 debug body + `#ifdef` 门控**。clean body(`EpDispatchIntraNodeKernel_clean_body`)一字不动;
 `core.py` 出 `-D`,`cache.py` 加 cache-key 后缀;门控默认关。

## 3. 必须记住的事实

- **wave32**:`warpSize` 宏在 gfx1250 展开为 32(`utils.hpp:34`)。python JIT 路径 `blockDim = 32 × wpb`,
 `--warp-per-block N` = N 个硬件 warp。wpb=8/DBN=64 → blockDim=256、globalWarpNum=512、8 token/warp。
- **改 token 分区必须 COUNT / FINALIZE / payload 三处同改**。payload 前只有 `__syncthreads()`,前提是
 "每个 block 读回它自己写的 `dispDestTokIdMap`"。只改一处 → 跨 block 竞态、acc FAIL。
- `totalRecvTokenNum` 只在 combine 清零;只跑 dispatch 会累加,recv 取新建 op 后第 1 次 dispatch 的值。
- 带宽只用算法带宽;a2a 是 per-rank outbound,dispatch 是去重 recv,口径不同别混。
- 引用带宽必须写明是否带 `MORI_DISP_TIMING`(探针开销 ~5%)。冷编译 ≥500s。
- **`rocm-smi --showuse` 的 100% 是假象,本节点上它不是信号**。实测:连采 5 次,4 张卡全是
 `GPU use (%): 100`,同时 VRAM 一直停在 175251456 B/卡(空闲底线)、KFD 只有 `inb-node-agent`
 (0 VRAM)、`yanbo_fused`/`wjx_moe_ep`/`mori_kc` 三个容器各 0 个活 python。**判节点空闲只看三样**:
 VRAM ≈175MB/卡、`--showpids` 只有 `inb-node-agent`、各容器无活 python。
- **别拿 use% 或 `docker ps` 的 "Up" 去归因性能异常**。容器 Up ≠ 在算,use% 恒 100 也 ≠ 在算。
 归因必须有真实占用证据(VRAM 升高 / showpids 里有别人的 PID / 对方容器里有活进程)。
- **`rocm-smi` 的 power / temp / sclk 同样是冻结值,一并不可用**(间隔 10s 采 3 次,W 一位小数都
 不动;见 §10)。本节点 rocm-smi 只有 VRAM、`--showpids`、`--showtopo`、link 速率可信。
 判性能异常的唯一手段是跑基准:`_ct_epsim.sh MODES=0` 给出与 mori 无关的 a2a 天花板。
- **邻居作业的 VRAM 足迹只有 ~650MB/rank,所以"VRAM 在 175MB 底线"不足以证明空闲**。实测抓到
 另一容器在同 4 张卡上反复起 4 卡作业:
 `torchrun --nproc_per_node=4 test_mega_moe.py -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 --combine gather`
 (4579% CPU),4 个 spawn_main 各 652MB。**唯一可靠的查法是看主机进程表**:
 `ps -eo pid,user,pcpu,etime,args --sort=-pcpu | head` —— 跑着的 trainer 一定在这里,
 `_ct_nodestate.sh` 已包含。`--showpids` 里出现高位 PID(5 位以上,我们容器内 PID 是 4 位)也是信号。
 他们的作业是**短作业反复起**(能看到多批 defunct,间隔十几分钟),所以"跑前空闲"不等于"跑中空闲";
 长测量要跑前跑后各查一次进程表。

## 4. 登录

- SSH `fizhang@ctheliosr-1b114-f01-2.mnb.dcgpu`,key `~/Desktop/my/id_ed25519`。
- 容器 `MORI-EPV2`(`--privileged`,重启后 `docker start`);`SRC=/root/mori_tdm`;
 JIT 源 `/opt/venv/lib/python3.12/site-packages/mori/_jit-sources`;JIT 缓存 `~/.mori/jit`。
- 主机 cmdline 带 `modprobe.blacklist=amdgpu`,重启后先确认 `/dev/kfd` 存在再动容器。
- 基础环境:`MORI_EP_COMM=cco MORI_DISP_BATCH=1 MORI_DISP_TDM=1 MORI_SHMEM_HEAP_SIZE=6G`。

## 5. 测试命令

```
# 一次跑完:部署 -> acc 门 -> bw -> 分段中位数
tools/_send_ct.ps1 -Script tools/_ct_go.sh -Envp "EXP='MORI_DISP_FINLANE=1 MORI_DISP_TIMING=1' ACC=1 WAIT=1" -Tmo 1100
tools/_send_ct.ps1 -Script tools/_ct_res.sh -Tmo 90     # 补读上次结果

# 几何扫描:dbn/wpb 是 launch 参数不是编译期,整个扫描只付一次冷编译
tools/_send_ct.ps1 -Script tools/_ct_geo.sh -Envp "GEOS='128:8 160:8 192:8'" -Tmo 1200

# 单个几何的 acc 门 + 重复 bw(两者同一个 noTIMING 二进制;_ct_go.sh 的 acc 走 TIMING 构建)
tools/_send_ct.ps1 -Script tools/_ct_ver.sh -Envp "DBN=128 WPB=8 REP=2" -Tmo 1500

# 恢复阶梯(节点重启/挂过之后)
tools/_send_ct.ps1 -Script tools/_ct_health2.sh -Tmo 400                                   # VECADD PASS
tools/_send_ct.ps1 -Script tools/_ct_epsim.sh -Envp "GRIDS=64 BLOCK=256 MODES=0" -Tmo 600  # 应 1545-1610
# ^ 也是判"节点是否退化"的唯一可信手段(不经 mori 内核)。2026-07-30 实测只有 1267.5,见 §10;
#   节点处在这种状态时,dispatch 的绝对值不可与 §6 的历史表比较,只能同时段互比。
```

`_ct_go.sh` 现在也部署 `~/dc.py` -> `mori/ops/dispatch_combine.py`(launch 几何在这里,不是内核源码,
不会失效 JIT 缓存)。`_scp_ct.ps1` 的 `-Pairs` 在 `powershell -File` 下只认单个值,多文件要分次调用。

## 6. 当前状态(2026-07-29,EP4-4K bf16 hidden7168,默认路径无 env 门控)

```
默认几何 DBN=64 wpb=8 -> 1280.7 / 1279.2 / 1276.4 GB/s(noTIMING, ACC PASS,重复三次)
                          -> kernel 166.0us,run 间噪声 ±0.3%
meta store drain 延迟已转为无条件默认(不再有 METADRAIN 门控)
```

**已上库 `5d90c102`**(基线 `a12bf9b2` = 1268.6 -> 1275.4 / 1276.2 / 1277.7,ACC 3/3 PASS)。
`_mPend` 是运行时状态不是门控,去掉它等于无条件多等一次。清理时发现 `METAFIELD` 分支还在测
这个已不存在的宏、静默退回了立即等待,已一并改无条件;其余 4 处只是历史注释在引用宏名。

**约束**:默认必须保持 64 block × 8 warp。1366 GB/s 有两个可达点,但都要多占物理资源,
因此都不作默认(见下),1.3T 在严格 64×8 下未达成,实测上限 1278.8。

### 设备事实(实测 `torch.cuda.get_device_properties`,勿再凭代际经验推断)

```
gfx1250: CU=256   LDS=327680B (320KB) per block   max 2048 threads/CU
DBN=64 只占 256 个 CU 中的 64 个 = 25%;wpb=8 的 tile = 8×7168×2 = 114KB,只用掉 320KB 的 36%
```

早期文档写"114KB 几乎占满一个 CU 的 LDS""wpb=16 要 229KB 不可行"是错的,基于旧代际的 128KB
上限。gfx1250 是 320KB,wpb=16 完全可行。

### 两个 1366 GB/s 的点,性质不同

```
DBN=64  wpb=16 -> 1366.4 GB/s   block 数 64 不变、占用 CU 64 个不变;只是把这 64 个 CU 内部
                                 闲置的 LDS(114->229KB)和线程槽(256->512,上限 2048)用起来
DBN=128 wpb=8  -> 1366.6 GB/s   block 64->128,占用 CU 64->128,真的多要了 64 个 CU
```

两者都是 1024 个独立 warp 在飞 —— payload 的瓶颈就是这个数(见 §9 PAYSPLIT)。

DBN 扫描(wpb=8,noTIMING,单次):

```
DBN   32     48     64     96    112    128    144    160    192    256
GB/s  882   1073   1276   1239  1160   1354   1356   1356   1355   1352
```

wpb 扫描(DBN=64 固定):

```
wpb    8     12     16     20     22
GB/s  1278  1218   1365   1255   1239
```

wpb=16 是唯一峰值,因为 64×16×_tpi(4) = 4096 正好一轮分完 token,每 warp 4 个;wpb=8 要两轮,
wpb=12 第二轮只有 1024 个 token(3/4 的 warp 空转),wpb=20/22 一轮就完但大量 warp 拿不到 token。

## 7. 分段成本(差分法,noTIMING,DBN=64 基线 166.0us)

`MORI_DISP_NOPAY` / `NOMETA` / `NOSTG` 三个诊断门控(ACC 必 FAIL,只看 kernel 时间):

```
去 payload  -> 41.0us   (payload 边际 125.0us;引擎下限 212.4MB/1600GB/s = 132.8us,
                         说明约 8us 的非 payload 工作被 payload 掩盖)
去 meta 阶段 -> 159.05us (meta 阶段 = 6.95us,其中仅 ~1.8us 是 2.9MB 的引擎时间)
去 staging  -> 160.55us (FINALIZE 的 gather = 5.45us,大头是从 HBM 读原始输入,不是写)
```

**TIMING 构建的分段不能用于排序**:探针在 per-token 循环内,非 payload 段在 TIMING 下显示 ~87us,
noTIMING 实际只有约 33us(2.6 倍膨胀)。定位分段一律用上面的差分法。

## 8. 已否定的方案(勿重试)

这些门控**已从代码里删除**,只留本表和 `intranode.hpp` meta 段开头的清单注释。要复现某一条得先重写它。

| 方案 | 结果 | 原因 |
|---|---|---|
| `METAFUSE` meta 融进 payload 循环、逐目的地跨卡写 | 462.6 GB/s | 跨卡细粒度写,每 warp 串行 ~116 次 fabric 往返 |
| `METAVEC` staging 批量 vector 直发对端 | 995.5 GB/s | 跨卡 vector 写有效带宽仅 ~54 GB/s(TDM ~1600) |
| `METALDS` gather 直接进 LDS、TDM 从 LDS 直发 | 1253.8 GB/s | 省掉的只是 staging 写+TDM load;抵不过 LDS bank 冲突、额外 barrier、失去延迟 drain |
| `METAFIELD` per-(peer,field) 分配 | -4.3% | meta 成本跟字节走,不跟 op 数走 |
| `METASPLIT=1` | -1.5% | 同上 |
| `SRCVEC` srcmap 跨卡写从 4B 改 16B | htSrc 20.1→25.3us | **跨卡写是延迟受限,产出取决于并发事务数**:58 个 4B 由 32 lane 发出 = 32 个并发事务,改 uint4 后只有 14 个 lane 活跃 = 14 个并发。加宽反而减并发 |
| `PAYSPLIT=2` token 切两段,每 warp 在飞 TDM 翻倍(LDS 不变) | 1272.6(-0.5%) | 引擎不因单个 warp 排更深的队列而提速;它要的是更多**独立 warp** 各自持有的 tile |
| `GRIDFLAG` grid barrier 改每 block 独立 flag + 32 lane 并行轮询 | 1256.7(-1.7%) | 单地址自旋的那条 cacheline 常驻 L2、反复命中极便宜;64 个分散 flag 让每轮轮询变成 64 次跨行读,开销超过省下的原子序列化 |
| `PAYBUF` 每 warp 两个 payload tile,下一 token 的 load 压在本 token 的 store 后面 | 1280.8 vs 1280.7 | 没有气泡可填:partB 已是 1582 GB/s = TDM 引擎天花板(`_ct_tdma2a` 同几何 1569–1590) |

结论:**跨卡 meta 必须走 TDM,TDM 只能以 LDS 为源**,所以 staging + LDS 中转这条结构去不掉。
meta 相关总成本 12.4us 里已无可榨取的部分。

srcmap 为什么永远是标量:它每个 slot 只有 1 个元素,一个 (block,peer) 的 run 在 DBN=64 下是
cc≈58 个元素,而合法 tile 至少要 64 个(`TdmCheapDim1` 的 32×2),`TdmAlignSplit128` 的对齐余数
也只够 1 行(要 2 行)。所以 htSrc 是 metasend 里最大的桶(20.1us / 36.7us,TIMING),但**它没有
可用的 TDM 形式,也不能靠加宽 store 改善**。

## 9. 加倍法(cost isolation by doubling)——已作废,门控已删

删除法(`NOMETA`/`NOPAY`/`NOSTG`)到不了 COUNT / RESERVE / COMPLETION:删 COUNT 会让 RESERVE 拿到
垃圾 s_N、payload 写到 slot 范围外(内存破坏,不只是结果错);跳过 grid barrier 或 waitpeer 会破坏
信号清零不变量,下一次 replay 永久等待。所以当时改用**让该阶段做两遍等量工作**(`DBLCOUNT` /
`DBLRESERVE`),`kernel(2x) - kernel(1x)` 取其成本,好处是结果依然正确、ACC 仍 PASS。

**但实测数据自相矛盾,两个门控已连同代码删除**:`DBLCOUNT` 推出 COUNT = 48.8us,比它所在的非
payload 总时间 41.0us 还大,算术上不可能(§7);`DBLRESERVE` 同批次的 ~989 GB/s 一并作废。原因
未查清。要重做,先解决已知陷阱:第二遍的目标不能用运行时选择的指针(`_cp == 0 ? s_N : s_Ndbl`),
即使两边都在 LDS,编译器也无法证明地址空间,原子退化成 flat 且**两遍都变慢**,差值失去意义;必须
写成两个分别访问静态数组的分支(METALDS 踩过同一个坑),而且循环体不能展开(两份体使寄存器压力
升高、拖慢包括 payload 在内的整个 kernel,于是测到的是编译器的反应)。

方差大、单次不可下结论的项:`mSt` 3.9–14.3、`compl/waitpeer` 3.9–60;整体 run 间抖动约 1.5%。

## 10. 未结项 / 下次注意

- 1.3T 在严格 64×8 下的结论:`PAYSPLIT` 证明加深单 warp 的 TDM 队列无效,payload 缺的是
 **独立 warp 数**(512 -> 1278,1024 -> 1366,与 64×16 / 128×8 哪种来源无关)。要在 64×8 内
 再涨,只剩"减少总工作量"或"打尾巴"两条路,不是加并发。
- payload 已在引擎天花板(partB 134.3us / 212.4MB = 1582 GB/s),所以剩下的 2.6us 只能出自
 ~32us 非 payload 工作。`FASTDEDUP` 是这条路上唯一还活着的门控(默认 OFF,ACC 已 PASS,
 干净机器上的收益还没测到)。
- **纯重构要不要跑基准,可以不占 GPU 就判定**:`tools/_ct_isadiff2.sh` 用同样的 flag 分别编译两份
 源码,unbundle 后逐指令 diff。门控清理那次 559659 条指令只差 10 条,全是同一条 `assert` 的
 `__LINE__` 立即数(0x8e6→0x6cf = 2278→1743),vgpr/sgpr/LDS/spill 全同 ⇒ 性能不可能变。
 注意 `--genco` 出的是 clang offload bundle,不先 unbundle 的话 objdump 输出 0 行,
 diff 会拿两个空文件报"完全一致"(第一版脚本就是这么假通过的,已删)。
- **1001.5 GB/s:确定不是内核造成的,但"节点自身退化"这个归因已被推翻,起因仍未定**。
 实测(2026-07-30):`_ct_epsim.sh MODES=0`(不经 mori 内核)健康 1569–1590,当时三次
 1267.8 / 1259.3 / 1275.3(均值 1267.5,±0.6%)= **−19.8%**;同时段 dispatch 64×8 三次
 1002.7 / 1003.8 / 1001.8(ACC 3/3 PASS)= **−21.6%**。两者同幅下滑,且 §8 已实测 payload 就
 贴在这个天花板上(partB 1582 ≈ epsim 1569–1590),所以**掉的是节点给出的 a2a 天花板,不是内核**
 (再加 ISA diff 见上,双重证据)。
 **但"节点硬件退化"的结论作废**:当时据"VRAM 在底线 + showpids 只有 inb-node-agent"判空闲,
 事后在同一节点抓到邻居容器反复起 4 卡 EP4 MoE 作业,足迹只有 650MB/rank(§3),这种作业**在我们
 的空闲判据下完全隐形**,而且是短作业反复起,极可能就压在测量窗口里。现在的状态:降到 1267 的原因
 未定,竞争是首选假设。已排除的仍然成立:不是门控删除,不是链路(XGMI 1 hop、PCIe 32 GT/s x16
 均在上限)。要复测先按 §3 查主机进程表,且跑前跑后各查一次。
 **这一时段的绝对值不可与历史表比,只能同时段互比**;换算系数 dispatch/epsim ≈ 0.79(健康 0.81)。
- **`rocm-smi` 的功耗/温度/sclk 在本节点也是冻结值,同样不是信号**。间隔 10s 连采 3 次,
 4 张卡的 W 一个小数位都不变(1138.0 / 1130.0 / 1176.0 / 1176.0),真实功耗不可能这么静止;
 "空闲" 却读到 1.1kW、sclk 2354–2371(上限 2400)也自相矛盾。**所以不能用 sclk 读数断言"没降频"**,
 降频这条因此既未证实也未排除。`--showuse` 见 §3,现已知 use%/power/temp/sclk 全组不可用。
- **`.gitignore`**:工作区那份曾把仓库原有规则整体覆盖(`build/` `__pycache__` `*.so`
 `*.egg-info` 全没了),已恢复到 HEAD 且未提交。代价是 `tools/_*` 几百个脚本一直是 untracked,
 别再 `git add -A`。要静音就在原有规则**后面追加**。
- 容器 `MORI-EPV2` 会自己停,`docker exec` 报 `is not running` 就 `docker start MORI-EPV2`,
 不是 SSH 问题。`_send_ct.ps1` 的 keepalive 已放宽到 `ServerAliveCountMax=60`/`ConnectTimeout=900`
 (授权阶段现在要几百秒,曾被 90 秒静默判死)。

## 11. Combine(2026-07-30 首次实测。节点当时有邻居竞争,绝对值偏低约两成,见 §10)

在此之前 combine **一个带宽数字都没有**:`ep4_disp_bw.py:3` 写明 "It NEVER runs combine",而
`_ct_go.sh` 跑的就是它;combine 只在 `ep4_acc_check.py` 里被验过正确性。唯一产出 combine 带宽的是
仓库自带 bench `tests/python/ops/bench_dispatch_combine.py`,口径与 dispatch 同形(
`comb_bw = total_recv_token × hidden × elemsize / comb_duration`,四卡平均,`bench:480-489`)。

```
tools/_send_ct.ps1 -Script tools/_ct_comb.sh -Envp "COMBOS='256:32' DBN=64 DWPB=8" -Tmo 2400
```

几何扫描(dispatch 固定 64×8,同一 run 内 dispatch 都是 955–970 GB/s,作同时段标尺):

```
comb几何   80x8   64x16  128x16  256x16  192x32  128x32  256x32
warp总数    640    1024    2048    4096    6144    4096    8192
GB/s       104.6  180.0   325.9   466.3   468.5   533.9   641.4
lat us     2030   1180     651     455     453     397     331
```

- **shipped 默认严重欠配**。bench 默认 `LaunchConfig(192,32,128,16)`(`bench:904`)的 combine 段是
 128×16 = 326 GB/s,只有实测最优 256×32(641)的 51%;不传参时吃 `config.block_num=80` /
 `warp_num_per_block=8`(`dispatch_combine.py:262-263`)= 104.6,更差 6 倍。dispatch 有
 `_intranode_dispatch_default_launch()` 把默认覆盖成 64×8,**combine 没有任何 per-body 默认**。
- warp 总数不是唯一变量:128×32 和 256×16 都是 4096 warp,却是 533.9 vs 466.3;192×32(6144 warp)
 反而低于 128×32。所以 block/warp 的**切分方式**本身有影响,原因未查。
- **wpb=32 已到上限**:combine 的 LDS 只有 `wpb × topk × 2 × 8` = 4KB(只存指针数组,不是 payload
 tile,`dispatch_combine.py:659-670`),LDS 不是约束;卡的是 blockDim=32×wpb ≤ 1024。wpb=64 未实测
 (推断会直接 launch 失败)。
- **block=320 会挂,能用的上限就是 256×32**(= CU 数 × 最大 wpb)。实测 320×32 十分钟零输出,而
 256×32 只需 18s,`/tmp/comb_320_32.log` 停在 "Benchmarking with combine_block_num=320" 之后。
 **机制未证实**:combine 确有 grid barrier 自旋等 `combineGridBarrier == gridDim.x`
 (`intranode.hpp:327-331`),但每个 block 是"thread0 加完计数就继续走"、会腾出 CU,理论上 320 个
 block 应该能排完 —— 所以别把"超订必挂"当已知结论,要用 >256 得先查清是不是这个 barrier。
 挂了之后:先 `taskkill` 本地 ssh(否则脚本会继续跑下一个配置一个个挂),再 `_ct_stopours.sh`。
- combine 最好点仍比 dispatch 慢 50%(同样 202.47MB:dispatch 220us vs combine 331us),而且到
 8192 warp 都没见平台。**下一步:判它是带宽受限还是 topk=8 归约/延迟受限**,再决定要不要动内核。
 注意先在无竞争时段重测一遍上表,当前绝对值不可信,只有相对排序可信。

## 12. Combine token send(2026-07-30 晚,节点已确认空闲)

节点空闲证据:VRAM 175MB/卡底线、KFD 只有 `inb-node-agent`、进程表无活 trainer(§3 三查全过);
同批 `_ct_epsim.sh MODES=0` = 1591.5,落在健康区 1545–1610,所以本节的绝对值可与 §6/§8 比。

### 12.1 EP SIM:1:1 发送形态本身只值 13%

`_ct_epsim.sh` 新增两个模式(与 mode4 用**完全相同**的 TDM 描述符,只差每个 load 摊几个 store):

```
mode6 = combine 形态,1-load:1-store,散列 dest slot
mode7 = 同 6 但连续 dest slot(隔离散列成本)
```

grid=64 block=256(= 512 warp,与 dispatch 64×8 同):

```
mode  0(a2a)  4(1:N)  6(1:1散列)  7(1:1连续)
GB/s   1591.5  1601.4     1397.5      1441.4
```

- **1:1 只比 1:N 低 13%**(1601→1397),散列再值 3%(1441→1397)。
- 所以"combine token 只有一个目的地、摊不了 load"**不足以**解释真实内核的 771 GB/s。
 1:1 的天花板是 ~1400,不是 ~775。

### 12.2 771 GB/s 是错口径,不是纯发送

dispatch 的 1582 来自 `_pb_maxdur`:`_pbStart` 在 payload 循环前一行、循环后立刻 atomicMax
(`intranode.hpp:1318` / `:1401`),**只括住 payload**。combine 从来没有这个探针,唯一的数是
整个 kernel。而 `MORI_COMB_NOREDUCE` 只删掉 `WarpAccumLF` 本身,**不删**:

1. 跨设备 barrier;
2. `~2012` 起的 gather 准备循环 —— PUSH 走 `UseP2PRead=false` 分支(`:2037`)照跑,
 `curRankNumToken × warpsPerItem` 次,每次 topk 个指针 + 读 `dispDestTokIdMap`。

所以 275.3us / 771 GB/s 是"发送 + barrier + gather 准备",拿它对 dispatch 的孤立 payload 段比,
**结论无效**。已补 `_comb_push_maxdur`(同 `_pb_maxdur` 规矩,`[CSPLIT]` 多一列 `cPush=`)。

**实测(`MORI_COMB_TDM=1 MORI_COMB_TIMING=1`,ZC=0,64×8,四卡)**:

```
call0  cPush = 113.3 / 116.5 / 116.5 / 128.9 us     cRed = 744.2 / 750.5 / 757.8 / 767.4
call1  cPush = 133.0 / 201.6 / 225.1 / 230.6 us     cRed = 776.6 / 781.3 / 787.0 / 799.3
call2  cPush = 131.2 / 202.5 / 226.6 / 255.0 us     cRed = 767.9 / ...
cIssue / cWait 恒 0(PULL 专用桶,PUSH 下应为 0,符合)
```

202.47MB / cPush 得 **832(255.0us) – 1874(113.3us) GB/s**。

**这个数不足以下结论,`cPush` 这个仪器本身不合格**(4 条,偏差方向全都对"已达标"有利):

1. **散布 2.3 倍**(113.3→255.0),取最好那次就是 1874、取最差就是 832。单看一个 call 没有意义。
2. **口径不是 `_pb_maxdur` 的口径**。`_pb_maxdur` 是 thd0 在 `__syncthreads()` 之后按 **block** 取跨度
 (= 整个 block 走完该相位);`cPush` 是每个 **warp** 各自跨度再 atomicMax,两端都没有 syncthreads。
 warp 启动有先后时,max-over-warp 的单段跨度**小于**相位墙钟时间 ⇒ 系统性偏乐观。
3. **没有段和校验**。`_cKern0` 在 push 循环之后才打,所以 cKern 只覆盖 gather 循环、cPush 落在其外,
 没有任何"各段之和 = kernel"的约束。§9 那次翻车(DBLCOUNT 的 48.8us 装不进 41.0us)就是缺这个约束。
 `_BPTS` 天生有(`_totChk = _pt[6]-_pt[0]`),这是它比 `[CSPLIT]` 这种 max 桶强的地方。
4. **违反 §7**:"TIMING 构建的分段不能用于排序,定位分段一律用差分法"。同一跑 bench 报 combine
 1108–1377us(noTIMING 是 720us),TIMING 把 kernel 放大近一倍,却用它算带宽。

`cRed` 的 750–800us 同样只说明"成本重心在 fold/barrier 一侧",不能当定量结论;而且 PUSH 下这个桶里
混了跨设备 barrier 的等待(gather 必须等所有 peer push 完)。

### 12.2b 删除法定价(noTIMING,`MORI_COMB_NOPUSH`)—— **这才是可用的数**

```
基线   (MORI_COMB_TDM=1, ZC=0, 64x8)              712.9 / 724.0 us
NOPUSH (同上 + MORI_COMB_NOPUSH=1)                482.2 / 482.7 us
------------------------------------------------------------------
token send 边际成本                                230.7 / 241.3 us
212.32MB / 230.7us =                              880 – 920 GB/s
```

**`cPush` 低估了一倍**(113us vs 231us),方向与 §12.2 第 2 条预测的一致(按 warp 取 max、
两端无 `__syncthreads` ⇒ 小于相位墙钟)。**以后 combine 分段一律用删除法,`[CSPLIT]` 只用来看重心。**

与同几何各参照物对齐:

```
dispatch payload 边际(§7 NOPAY 删除法) 125.0us  = 1699 GB/s
dispatch partB_maxdur(§8)              134.3us  = 1582 GB/s
EP SIM mode4  1:N 同描述符                       = 1601 GB/s
EP SIM mode6  1:1 同描述符                       = 1397 GB/s   <- 1:1 的结构上限
combine token send(删除法)             230.7us  =  880 GB/s   <- 只有 1:1 上限的 63%
```

所以:**1:1 只解释 13%(1601→1397),剩下 1.5 倍(1397→900)在发送循环自身,仍未解释。**
最大嫌疑仍是 §12.3 表最后一行 —— 循环内每 token 一次 32B 跨卡 `WarpCopy` 写权重(`:1894-1899`),
dispatch 特意把 meta 放在独立阶段。**未验证。**

另外 NOPUSH 剩下的 482us(gather + barrier)本身就比发送大一倍,是更大的鱼。

#### 12.2c 删掉「每 token 重建描述符」之后复测 —— **没有变化,原假设被推翻**

发送路径已改成:整 token 一个 tile、GROUP1 描述符提到循环外只建一次、chunk 循环连同
`_cTileElems` / `_cTdmType` / `_cTdmOk` / `MORI_COMB_PUSH2` 全部删除,循环体只剩
「1 warp 取 1 个整 token → TDM load → wait → TDM store」。当时的假设是:每 token 重建 6 个位域、
一次 min、一次 sub-128B 尾判,乘以 14810 个 token,就是那 1.5 倍缺口的来源。

```
full   (MORI_COMB_TDM=1, ZC=0, 64x8, SKIPCHECK)  717.8 / 716.4 us   (跑1)
                                                  720.4 / 721.1 us   (跑2, spread 0.7%)
NOPUSH (同上 + MORI_COMB_NOPUSH=1)                490.8 / 484.9 us
------------------------------------------------------------------------
token send 边际                                   ~231 us  (区间 225-236)
212.32MB / 231us =                                ~919 GB/s (区间 900-944)
```

对比改动前的 230.7us / 920 GB/s:**完全没动**。所以每 token 的描述符重建 / 循环记账
**不是**瓶颈——它在 `MORI_COMB_TDM=1` 下本来每 token 也只执行一次,删掉省下的是几条标量指令,
而这条循环不是标量指令受限的。别再往「减少 per-token 固定开销」这个方向猜了,PUSH2(减半 wait 次数)
和这次(删掉描述符重建)是同一类改动,两次都是零收益。

顺带确认 TDM 这条路本身是活的:同几何关掉 `MORI_COMB_TDM` 是 2589us / 82 GB/s,
开了是 717.8us,整个 combine 差 3.6 倍。

现在的成本构成(64×8,ZC=0):send 231us 只占 combine 719us 的 32%,**其余 488us 是 fold+barrier**。

未解的不自洽:1874 高于 §12.1 EP SIM 1:1 的 1397 上限。首选解释是 epsim 低估——它 50 次
back-to-back launch,每次每 warp 只有 8 个 tile(真实内核 29 个),per-launch ramp 占比高;
另有 tile 16KB vs 14KB 的差异。**没验证,别当已知。**

### 12.3 两个循环的实际差异(读码,非推测)

| | dispatch payload(`:1348-1383`) | combine push(`:1837-1901`) |
|---|---|---|
| load:store | 1 : 平均 3.6 | 1 : 1 |
| 循环次数 | 4096 token | 14810 token(同样 14810 次 store) |
| 每 token wait(0) | 2 | 2 |
| token 分区 | `_tpi`=4 连续一段 | 按 `globalWarpNum` 跨步 |
| 循环内跨卡细粒度写 | **无**(meta 在独立阶段) | **有**:`UseWeights` 时每 token 一次 32B 跨卡 `WarpCopy`(`:1894-1899`) |

固定开销("2 wait + 1 load + 索引")combine 要付 3.6 倍次数,这部分 EP SIM 已定价 13%。
**未定价的是最后一行**:§8 已实测跨卡细粒度写是延迟受限那一类(`METAVEC` 995.5、`SRCVEC` 加宽反而更差),
dispatch 特意把 meta 与 payload 分成两阶段,combine 却把它塞进发送循环里。

### 12.4 已否定(勿重试)

| 方案 | 结果 | 原因 |
|---|---|---|
| `MORI_COMB_PUSH2` 两个整 token 配一个 wait(2 tile/warp) | 755.9 / 738.4 vs 771.2 | 与 §8 `PAYBUF` 同一结论:引擎不缺队列深度。等待次数减半 + 占用率减半,总 TDM 流量只动 2%。**门控与代码已删**:发送路径只保留「1 warp 发 1 个整 token 到它的目的 PE」这一条展平循环,要复现得重写 |
| "1.55TB/s 是 load+store 共享的引擎顶" | **作废** | 拿 dispatch 整 kernel 173us 去比 combine 纯发送,口径错。§8 的 partB 134.3us 里 58.7MB load 与 212.4MB store 并行,总流量 2018 GB/s,不存在这个共享顶 |
| 目的地 slot 128B 相位没对齐 | **不成立** | `combXferPadded` 已 pad 到 128B(`:1642-1646`),该 pad 装得进 `MaxXferBytesPerToken()` 时 `combSlotOn128B` 为真,这是 TDM push 唯一的开关条件 |

### 12.5 下一步:先把仪器换成 dispatch 那套,再谈数字

combine 缺的是 dispatch 早就有的**两件配套仪器**,`[CSPLIT]` 的 max 桶两件都不是:

1. **`_BPTS` 式单线程时间轴**(block0/thd0,每个边界紧跟 `__syncthreads()`):给出 push / barrier /
 fold 的**划分**,并带 `段和 == kernel 跨度` 的自检。这是唯一能让分段互相制约的形式。
2. **删除法定价**(noTIMING):`MORI_COMB_NOPUSH` 之于 combine = `MORI_DISP_NOPAY` 之于 dispatch。
 §7 的 payload 边际 125.0us 就是这么来的,`cPush` 那种 TIMING 桶按 §7 不能用来定价。

顺序:先 (2) 拿到 push 的无偏边际成本(ACC 必 FAIL,只看 kernel 时间),再 (1) 看划分。
在这两个数出来之前,**"combine token send 已达标"不成立**,§12.2 的 832–1874 只是区间。

已知但仍待定量的两条(不要拿旧数当结论):

- PUSH 的 fold 从**本地** staging 读 topk 路(`:2037` 的 `myPe` 分支),不跨卡 ⇒ 它慢只能是
 LDS/HBM 读放大或指令瓶颈,不是 fabric;
- `MORI_COMB_NOREDUCE` 的 A/B(918→968,+5.3%)只删了 `WarpAccumLF` 本身、留下整个准备循环,
 所以 5.3% 是 fold 算术的**下界**,不是 fold+barrier 的全部。别再用它论证 "fold 很便宜"。

## 14. 64×8 塌陷到 965(2026-08-01,节点确认健康)

一句话:**掉的只有 payload 相位,且只在 DBN<128 时掉;DBN>=128 一切正常(-3%)。节点没问题,
内核源码没变,是 <128 block 这个几何今天拿不到 fabric 带宽。**

### 14.1 结论表(全部 noTIMING,bytes 恒为 212.4MB = recv 14819 × 7168 × 2,t = 212.4e6/bw)

```
几何      full GB/s  full us   NOPAY us   payload us   payload GB/s   历史 full   差
64×8        964.9     220.1      41.5       178.6         1189         1276     -24%
64×16       987.7     215.0      35.6       179.4         1184         1365     -28%
128×8      1313.0     161.8      39.9       121.9         1743         1354     -3.0%
256×8      1307.2     162.5      42.6       119.9         1772         1352     -3.3%
```

DBN 扫描(wpb=8)今天 vs §6 历史,**64→112 是一条平台,128 处阶跃**:

```
DBN     64    80    96   112    128    192    256
今天   965   965   966   948   1313   1312   1307
历史  1276     -  1239  1160   1354   1355   1352
```

非 payload 全程健康:NOPAY 41.5us(§7 记 41.0)、meta 7.7us(6.95)、staging 5.7us(5.45)。
payload 内部再拆(带探针内核,full 校准 961.3 ≈ 基线 958.7):NOSEND 53.0us ⇒ load 只值 11.2us,
NOLOAD 215.8us ⇒ **store 值 174.0us = 1221 GB/s**。缺口全在跨卡 store。
`PAYRAW` 不可用于此对比:它退化成 1:4 全写,字节更多,读 791.7 比 full 还慢。

### 14.2 节点这次是健康的,而且 64 block 就已饱和

**别再用 BLOCK=32 跑 epsim 判节点**,那是错口径:`GRIDS=64 BLOCK=32` 读 1275,正好落在 §10
"退化态 1267.5" 上,会误判成节点退化。正确口径是 §"恢复阶梯" 的 `GRIDS=64 BLOCK=256`。

同几何(BLOCK=256)、同量级工作集(tiles/peer=14848 = 243MB,每 warp 29 tile,与真实内核一致)
的 epsim mode0 grid 曲线:

```
grid    32     64     96    128    192    256
GB/s  1346   1748   1773   1766   1654   1604
```

**64 block 就到 1748,96 达峰 1773,再往上反而降。** 所以"64 CU 喂不满 fabric"不成立:
硬件在 64 block 上给得出 1748,dispatch 只拿到 1189(68%);到 128 block 拿到 1743(99%)。

epsim 逐项加上 dispatch 的结构特征,全都不解释这 32%:

```
mode0 连续 1:4                      1608.6      工作集 67MB→134→243MB   1616.8→1696.6→1755.5
mode1 散列 dest slot                1613.5      分配 ipc vs cco式VMM     1760.0 vs 1758.2
mode2 散列 + 远程 meta 写           1552.3      mode4 真实 1:N 描述符    1600.1
mode7 1:1 连续 16384B               1676.3      mode6 1:1 散列 16384B    1596.6
mode8 1:1 散列 + 真实 14336B tile   1523.3   <- 结构成本合计只有 -13%
```

### 14.3 已排除(都实测过,别重做)

- **节点/邻居**:epsim 正确口径 1598.8–1773;跑前跑后主机进程表干净(§3 三查),VRAM 175MB 底线。
- **内核源码**:产出 1271.8 的是 host `~/` 那套 07-30 代码(2271 行,md5 332398334e88),
 用它 + `_ct_go.sh` 今天只读 954–967。容器 repo 是 shallow clone,只有 a12bf9b(07-28,1708 行),
 与 07-30 那套差 915 行 —— **拿容器 HEAD 当基线是错的**,它今天读 877.4。
- **host 侧 C++**:`a12bf9b2..d0a90b83` 零 `.cpp/.h` 改动,`.so`(07-29 10:28)无需重编。
- **工具链**:`_rocm_sdk_core/devel` 7.14.0a20260623,bitcode 全部 06-26 11:04,一字未动。
- **JIT 部署**:`_jit-sources` 快照 vs repo 树 0 differ / 134 identical。
- **测量探针**:容器的 `ep4_disp_bw.py` 与 host 原版只差一个默认 7168 的 `--hidden` 参数,
 replay=10/iters=10 一致,行为等价。
- **分配方式 / 散列 / tile 几何 / 工作集**:见 14.2 表,合计 -13%,不解释 -32%。
- **`rocm-smi` 的 power/temp/sclk/GPU%**:仍是冻结值(本轮读到 1141.0/1133.0/1179.0/1179.0W、
 GPU%=100、温度却只有 55–57°C,且重启容器后逐位不变)。§3 已写明,本轮又踩了一次,**别再用**。
 容器里的 `<defunct>` python 也不占 GPU:清零后功耗读数一模一样。

### 14.4 两个把我带偏的口径错误(务必避开)

1. **用 `bench_dispatch_combine.py --cmd bench` 判 dispatch**。`_ct_comb.sh` 开头已写明那里的
 dispatch 会吸收前一个 combine 的 skew,不是 node-state 口径。node-state 数只认 `_ct_go.sh`。
 另外该 bench 有两行读数:`EPV2-timing`(back-to-back replay 共用一对 event,历史 169.1–170.1us /
 1249–1256)和 `Round`(每 replay 一对 event,历史 175–181us / 1174–1214)。只抓 `Round` 却去比
 169.1,基线就错位了。
2. **用 `BLOCK=32` 跑 epsim 判节点**,见 14.2。

### 14.5 现在能用的结论

- **要带宽就用 DBN=128 wpb=8**:1313.0 GB/s,ACC PASS,payload 121.9us = 1743 GB/s 已贴
 引擎天花板(epsim 同几何 1766)。64×8 也 ACC PASS,但只有 964.9。
- **仍未解**:同一份源码、同一工具链、节点更健康的情况下,`DBN<128` 的 payload 效率从
 97%(1699/1748)掉到 68%(1189/1748),而 `DBN>=128` 只掉 3%。
 下一步建议:64/80/96/112 今天读数完全平坦(965/965/966/948),而历史是有结构的
 (1276/-/1239/1160),说明今天 payload 时长与参与的 warp 数无关 —— 这最像 payload 循环的
 轮次同步(1024 个 token-slot / _tpi=4;<128 block 时需要两轮)或 block 间 skew 在主导。
 要定这一条,需要**每 block 的 payload 跨度**,而 `_pb_lo/_pb_hi` 那种全局 atomicMin/Max
 绝对 clock64 不行(跨 launch 污染,本轮读出 5.56e8 us)。用 §12.5 说的 `_BPTS` 式
 block0 单线程时间轴,并带"段和 == kernel 跨度"自检。

---

## 15. Combine PUSH 发送慢在"写到哪儿",不在"查表"(2026-08-01,节点健康)

§14 之后 dispatch 已回到 1250 GB/s 基线,转做 combine。全部 64×8 / ZC=0 / `MORI_COMB_TDM=1` /
SKIPCHECK,workload 恒为 202.47MB(14123 token),同口径可比。

### 15.1 删除法定价

```
full                                              718.4 / 721.9 / 719.6 / 719.1 us   (四轮,spread 0.5%)
NOPUSH  (发送循环 trip=0,几何/LDS/gather 不变)     480.2 us
NOROUTE (localSrcMap 查表换成算术)                617.3 / 617.9 / 618.3 us
--------------------------------------------------------------------------------
发送边际 = full - NOPUSH                          238.2 us  = 850 GB/s
NOROUTE 值                                        101.1 us  ← 发送相位的 42%,combine 的 14%
```

**结论先行**:发送慢的根因是**同一时刻在飞的 token 目的地扎堆在少数几张卡上**,不是查表、不是散射写、
不是每 token 开销。换个遍历顺序(`MORI_COMB_SPREAD`)即可,64×8 combine 降 14%、128×8 降 32%。
15.1/15.2 是我走过的两条错路,留着是为了别人别再走。

### 15.2 **那 101.1us 不是查表的访存**,两面夹击都是空的

对查表本身做了两个**保正确性**的改动,各自独立二进制(见 15.4 的缓存核对):

```
档                                     combine     相对 full
full                                   721.9 us    -
MAPBATCH  每 warp 一次 lane 并行批读     719.3 us    -2.6 us
          + __shfl 分发(dispatch :1412-1435 的形状)
PREBASE   4 个 peer 基址提到寄存器,      719.5 us    -2.4 us
          不再每 token 读 p2pPeerPtrs[destPe]
两个都开                                718.9 us    -3.0 us
```

**3.0us / 101.1us = 3%。**`localSrcMap[tokenIdx]` 的地址只依赖循环计数,编译器本来就能把它提到上一轮
TDM 等待之前;这个空结果说明它已经这么做了,**没有可去掉的串行**,uncached 与否都一样。
"两级依赖 load 卡在 store 地址路径上"这个推断**被推翻**,别再做第三次。
**代码与门控已删**(intranode.hpp / jit/core.py / jit/cache.py),只在发送循环注释里留了空结果记录。

### 15.3 根因:**同一时刻在飞的 token 目的地扎堆**。`MORI_COMB_SPREAD` 换个遍历顺序即可

先排除掉一个错误猜测:我一度把剩下的 98us 归给"散射写 slot"。**这是错的,epsim 早就把散射定价为零**
(§14.2 `mode0 连续 1608.6` vs `mode1 散列 1613.5`)。而且 mode6/mode8 的 `p = slot % nPeers` 里
`slot` 是随机置换,所以 epsim **连"warp 内逐 token 换 peer"也模拟了**。同批复测(节点 mode0 1609):

```
mode6 1:1 散列 16384B          1397.9      mode7 1:1 连续 16384B   1461.5
mode8 1:1 散列 + 真实 14336B   1315.8      内核发送段实测           ~856
PEER=1(每 rank 独占一条链路)  mode6 1336.2 / mode8 1263.6   <- 单链路独占也不慢
```

epsim mode8 与真实循环**同描述符、同 14336B token、同 14336B slot 步长、同散射 slot、同 1:1、
同每 token 两次 wait、同 `t += gwn` 分区**,读 1315.8;内核读 856。唯一没被覆盖的差异是
**瞬时目的地分布**:epsim 的 `slot%nPeers` 每 token 都换 peer,四张卡永远同时被驱动。

内核这边是反过来的。dispatch 的 Phase 2 是**每 block 一次远程 atomic 预留**(`intranode.hpp:911`,
注释 :904-906 原话 "this rank's slots on a peer are one contiguous run PER BLOCK"),所以接收端的
recv 索引空间被切成约 `worldSize × dispatchBlockNum` = 256 段连续区间,**段内 destPe 恒定**,本几何下
每段约 55 个 slot。而 combine 推送用 `tokenIdx += globalWarpNum`,同一时刻在飞的 512 个 token 是索引
空间里一个**连续 512 宽的窗口**,只落在约 9 段上 —— 四条链路里只有少数几条在动。

验证即修复。`MORI_COMB_SPREAD` 用大质数步长走同一批 token
(`tokenIdx = (step × 9973) mod totalRecvTokenNum`,质数不整除计数时是**双射**),token 集合、目的地、
字节数全不变,只改 warp 取 token 的次序:

```
几何      full        SPREAD      Δ          发送段(减 NOPUSH 480.2)      发送带宽
64x8      716.8us     616.7us     -100.1us   236.6 -> 136.5us            856 -> 1483 GB/s
64x8 复跑 719.8us     617.9us     -101.9us
128x8     653.0us     446.1us     -206.9us
对照 NOROUTE(强制 destPe 轮转)   619.6us     139.4us            <- SPREAD 已拿到它的 99%
```

**64×8 的 combine 降 14%,128×8 降 32%**,且 1483 GB/s 已反超 epsim mode8 的 1316。warp 越多窗口越宽、
扎堆越重,所以 128×8 的收益更大。这解释了为什么三次"削减 per-token 开销"的尝试(PUSH2、描述符重建、
MAPBATCH/PREBASE)全部读零:**循环里的每 token 工作从来不是瓶颈**。

注意:`MORI_COMB_SPREAD` 目前仍是门控,**未设为默认**,因为 PUSH 路在这个 bench 上拿不到正确性背书
(见 15.5)。它是保正确性的改动,双射性可证,要转正需要一个能校验 PUSH 的 caller。

### 15.3b 1470 已经贴住 1:1 的天花板,**1600 是 1:N 的数,不是这条路的目标**

SPREAD 之后发送段 1470 GB/s。把它放回参照系里:

```
epsim mode8  1:1 散列 + 真实 14336B         1315.8   <- 与内核同形态
epsim mode6  1:1 散列 16384B                1397.9
epsim mode7  1:1 连续 16384B                1461.5
combine 发送段(SPREAD 后)                  ~1470    <- 已在所有 1:1 参照点之上
------------------------------------------------------------
epsim mode0  a2a 连续 1:4                   1609.0   <- 2D 64x64 tile
epsim mode4  1:N 同真实描述符               1601.4
dispatch payload 边际(§7,212MB)            1699
```

差的这 130 是 **1:N 与 1:1 的结构差**,epsim 早已定价 13%(1601→1397):dispatch 一次
`TdmIssueLoad` 之后在 topk 个 peer 上连发 N 次 store(`:549-551`),本地 HBM 读被摊薄到 1/N;
combine 一个 token 只有一个目的地,**读的字节数恒等于发的字节数**,没有可摊的东西。绕开它只有三条路,
都已被否:去掉 LDS 中转走 `WarpCopy` 是 2589us / 82 GB/s;双缓冲(=`MORI_COMB_PUSH2`)因每 warp 两个
tile 折半占用而更差;换 PULL 只是把远程读换到另一侧,仍是 1:1。

**描述符形状也不是原因(新测,空结果)**。`TdmShape` 默认发 `tensorDim1==1` 的 1×N 楔形,而
`TdmShape2D` 注释说 gfx1250 要求两维都 ≥2,一直有个"楔形导致 a2a 1664 vs payload 1192"的推断。
`MORI_DISP_PAY2D` 从没测过,这次测了:

```
描述符(64x8 + SPREAD)      combine     dispatch
1x7168 楔形                 618.7us     1253.1 GB/s
128x56  (行 256B)           617.8us     1252.0 GB/s
256x28  (行 512B)           618.4us     1255.9 GB/s
64x112  (行 128B,贴下限)   639.5us     1096.7 GB/s   <- 反而更差
```

两个相位同时看,合法 2D 与楔形无差别。**这条推断作废,别再折 2D。**

所以发送这一相位可以收工了。**下一个目标是 fold + barrier 的 480.2us,它现在占 combine 的 78%**
(发送只剩 22%)。按 §12.5:先用 `_BPTS` 式 block0 单线程时间轴(带"段和 == kernel 跨度"自检)把
barrier / gather 准备 / fold 三者分开,再谈优化 —— `MORI_COMB_NOREDUCE` 的 +5.3% 只是 fold 算术的
下界,不能用来论证 fold 便宜。

### 15.4 三个让实验/推理失效的坑(都踩过,务必先排除)

0. **减法归因必须先查"备选项枚举完整"**。我把 101.1us 减掉实测的 3.0us,把余下 98us 扣给"散射写",
 而散射早在 §14.2 就被 epsim 定价为零 —— 减法本身没错,错在没回头核对 epsim 覆盖了哪些形态。
 **动手推理前先把 epsim 已有的模式表读一遍**(`_ct_epsim.sh` 开头 mode0-8 的注释块)。

1. **门控没进缓存键**。`NOPUSH/PUSHONLY/NOWEIGHT/NOROUTE` 当初内核里读了,但 `jit/core.py` 没有
 `-D` 出去、`jit/cache.py` 没进目录名 ⇒ A/B 拿同一个 `.hsaco` 跟自己比,读出"这段是免费的"。
 已修。校验方法:`~/.mori/jit/<arch>_<nic>..._<门控>/<内容哈希>/` 下每档应有**各不相同**的
 `.hsaco` md5,且冷编译时间戳应逐档错开约 90s。
2. **`-Aux` 只 scp 到节点 `/tmp/`,搬进容器是脚本自己的事**(`_send_ct.ps1:20-33`)。回归脚本漏写
 `docker cp` 就会静默复测旧二进制 —— 本轮第一次回归就是这样,`kernel=4` 那个残留计数才暴露出来。
 另注:`python/mori/_jit-sources/` 下那份 `intranode.hpp` 是打包遗留,**JIT 不用它**(内容哈希只随
 仓库树 `src/` 变),它 md5 停在 `94cf9be82e68` 是正常的,别去同步。

### 15.5 顺带修正的两条旧结论

- **`MORI_COMB_NOWEIGHT` 在这个 bench 上定不了价**:`bench_dispatch_combine.py` 调 combine 时
 `weights=None` ⇒ `args.weightsBuf` 为空,那段跨卡写**根本不执行**,门控删的是死代码,读数 0.9us
 全是噪声。§12.3 表最后一行、§12.2b "最大嫌疑" 都建立在它身上,**要验必须换一个真传 weights 的
 caller**。
- **PUSH 路(`--zero-copy 0`)本身过不了校验**:`full`、关掉 TDM 走 WarpCopy 的 `wc_full`,全都停在
 `dispatch_combine_test_utils.py:701` 的 `assert result_match`,和是否 TDM、是否有我的改动无关。
 所以这条路上**任何**改动都只能靠 SKIPCHECK 测性能,拿不到正确性背书;要验等价性得另找 caller。

## 16. Combine barrier(2026-08-01 夜 ~ 08-02 凌晨,节点健康)

目标是 combine 170us / 1.2TB。本轮全部围绕 barrier,**结论是:唯一确定的收益只有 15us(退避),
其余方向全否;而且"barrier 占 25%"这个前提本身口径不对,尚未重测**。

### 16.1 结论表(EP4-4K bf16 h7168,DBN=64×8 / CBN=128×16,`--zero-copy 0`,RUNRR,交替口径)

```
档                                    combine      说明
full,BARSLEEP=1(旧默认)              251.6us
full,BARSLEEP=127                     236.1/236.6  唯一落袋的改进,零风险
空跑栅栏 NOPUSH+PUSHONLY,sl=1          69.3us      <- 口径有问题,见 16.4
空跑栅栏 NOPUSH+PUSHONLY,sl=127        58.0/58.5us
BARFAN(0号块独轮询+缓存广播)          309.2us      正确但更慢,否
BARNOFENCE(删每块 fence,错结果)       58.3us      = 不删的 58.6,fence 免费
```

退避扫描(s_sleep 单位 ~64 clk,`MORI_COMB_BARSLEEP`):

```
sleep     1      8      32     127
barrier  69.3   65.8   61.4   58.5
full    251.6  246.0  240.3  236.1
```

**只值 15.2us,再大不动了**。所以剩下那 ~44us 与轮询频率无关。

### 16.2 fence 免费,但 fence 是正确性必需的(`MORI_COMB_BARNOFENCE`)

每块保留轮询、只把 acquire 从"每块"减到"仅 0 号块":

```
sl=127   58.6 -> 58.3     sl=1   69.3 -> 69.9      127 个 fence < 噪声
不带 SKIPCHECK:  有 fence 236.9 rc=0   /   无 fence rc=1 AssertionError
cache 目录:  ..._barsl127_runrr   vs   ..._barsl127_runrr_barnofence   (确认不是同一个 hsaco)
```

旧结论"BARACQ 证明 fence 免费"是错的——那只把作用域从 system 换成 agent,从没验过"没有"。已改注释。

### 16.3 已否定(勿重试)

- **BARFAN**:0 号块独自轮询非缓存 flag,经 `combineGridBarrier[1]` 广播 epoch,每块保留自己的
 system fence。**正确性回来了**(旧版 acquire 自旋是错的,rank3 token6 差 2.75;这版 rc=0),
 所以旧版的错因确认是自旋里 agent 作用域 acquire 发的是设备域 invalidate,盖不住第三方卡写的行。
 **但更慢:barrier 58.5→110.8,full 236.9→309.2**,而旧版是 112.1 —— 内存序和退避都不影响它。
 贵的是形状:把"128 个块读同一条非缓存 line"换成"127 个块读同一条 L2 line"什么也不省,
 还从并行变成串在 0 号块后面。
- 剩下唯一有依据的方向是**摊开**(0 号块轮询一次,广播到 128 条各自填充的 line,每块读独占的一条)。
 存储流水不往返,写 128 次不该等于读同一条 line 128 次。但 `combineGridBarrier` 只有 worldSize 个
 uint32,装不下 —— **这是第一个必须动 host 侧分配、要重编库的 barrier 改动**,不能只 `docker cp` 头文件。

### 16.4 **口径纠错:"barrier = 58us / 25%" 是空跑数,不是边际成本(未重测)**

所有 barrier 的数都出自 `NOPUSH+PUSHONLY`,即**删掉别的只留 launch + barrier**。这测的是空内核里的
栅栏,和它在真实 combine 里的成本是两回事:

1. **到达模式变了**。全量里 128 个块推完各自 token 后错峰到栅栏;空跑时它们在内核第一微秒同时压上去。
 grid barrier 的 atomicAdd 争用和 flag 轮询并发度完全不同。
2. **launch 分不出来**。那一档整个内核就剩 launch + barrier;"只留 0 号块轮询 = 15.0us"就是这个底噪。
3. **全量里 barrier 含"等最慢的对端推完"**,空跑时没有对端可等。这部分再便宜的栅栏也省不掉,
 只有像 DeepEP / NCCL EP 那样按 chunk 打 flag、不要全局栅栏,才能重叠掉。

同理,块数扫描(128→69.6 / 32→20.9 / 8→14.1,~0.51us/块)也是在空跑档做的,只描述空跑栅栏。

**已加 `MORI_COMB_NOBAR` 反向删除**(`intranode.hpp` 栅栏函数内,只删跨卡等待;arrival 计数 / flag 写 /
flag 自增 / fence 全留 —— 那几样是下次 replay 的前提,删了是挂不是测)。脚本 `tools/_ct_nobar.sh`
**已写好但尚未运行**。判据写在脚本头:`full - NOBAR` 才是边际成本;若远小于 58.5,则 barrier 不是
第二大头,优化方向要改。跑之前不要再引用"barrier 占 25%"。

### 16.5 计时尺子:EPV2 口径 vs 本 bench 交替口径(两个都对,回答不同问题)

`bench_dispatch_combine.py` 原本只给 dispatch 补过 EPV2 口径(`time_graph`:N 次背靠背 replay 共用
一对 event),combine 没有,所以没法和 `dispatch_combine_v2` 同尺对比。**已补**(搜
`Combine rank0 EPV2-timing`)。

```
                  交替口径(每 replay 一对 event)   EPV2 口径(10 次背靠背)
combine full            236.6us                        108.7us
combine 空跑栅栏         58.0us                         57.7us
dispatch                174us                          170.0us
e2e(一张图链 10 次)                401us
```

- **一次真实 step 的 combine 是 236.6**。e2e 是独立仲裁:一张图内链 dispatch+combine,没有
 per-iteration launch 边界、rank 天然锁步,读 401us ≈ 174 + 236.6 = 410,差的 9us 就是两次
 graph launch 边界的全部。它不支持 170 + 108。
- **108.7 不能换算带宽**:202.47MB / 108.7us = 1.86 TB/s,超过本机实测写 1.54 / P2P 读 1.40。
 低的原因是连续 replay 中间不插 dispatch,工作集一直 LLC 常驻。**它只能用来和 v2 的 `time_graph`
 比大小**,已在打印里标注。
- **顺带排除了"CPU 没对齐"**:若是进入不齐,空跑栅栏档(只有 launch+barrier)最该露,而它两个口径
 一致(58.0/57.7);交替口径 10 轮全平(236.1~237.8)、每轮四卡差 <1us、第 0 轮不高。

### 16.6 与 dispatch_combine_v2 的对比(v2 无 TDM,绝对值不用管,但记两个口径事实)

同形状 EP4 / h7168 / topk8 / EPR64 / 4096 tok,v2 `MODE=graph`:

```
v2 comb  80×16   919.8us    disp 1758.2us
v2 comb 128×16   747.9us    disp 1762.8us
v2 comb AUTO     724.2us    disp  446.9us     [disp (192,32) comb (128,4)]
```

AUTO 命中的正是 `tuning_configs.py:143-159` 那条 `RE-TUNED 2026-07-15 EP4` 条目,**不是跑偏了设计点**。

**未解:两边 recv 对不上**。同样声称 4096 tok/rank、topk8、EP4、256 experts:

```
v2      recv 14779   payload 211.87MB     ≈ 理论 4×4096×(1-C(192,8)/C(256,8)) = 14804
本 bench recv 14122   payload 202.47MB     低 4.6%,是采样标准差(~37)的 18 倍,不是噪声
```

本 bench 是 `use_max_token_num=True` + `randperm(256)[:8]`(`dispatch_combine_test_utils.py:384-444`),
`max_total_recv_tokens=0` 无截断,按理应落在 14804。**差的这 682 个 token 没查清**,它同时影响我们所有
GB/s 的分母。下次做跨实现对比前必须先解决。

### 16.7 本轮新增的门控与脚本

```
门控(默认全关,均已进 core.py 白名单 + cache.py 缓存键)
  MORI_COMB_BARSLEEP=N   轮询退避,唯一建议开的一个(127 值 15us)
  MORI_COMB_BARNOFENCE   诊断,错结果:删每块 acquire 只留 0 号块
  MORI_COMB_BARFAN       0号块独轮询+缓存广播,正确但慢,勿开
  MORI_COMB_NOBAR        诊断,错结果:删跨卡等待,用来算边际成本(尚未跑)
脚本
  _ct_barsleep.sh   退避扫描        _ct_barnofence.sh  fence 定价
  _ct_nfvalid.sh    门有效性(靠正确性翻转,不靠时间)
  _ct_barfan.sh     fanout 正确性+成本               _ct_combsteady.sh  两个计时口径
  _ct_v2cmp.sh      跑 v2 bench     _ct_cslog.sh/_ct_v2log.sh  只读日志不占 GPU
  _ct_nobar.sh      **已写未跑**,16.4 的判据在脚本头
```

### 16.8 下一步(按优先级)

1. 跑 `_ct_nobar.sh`。在拿到 `full - NOBAR` 之前,barrier 的所有占比说法都不作数。
2. 若 barrier 边际成本确实大:做 16.3 那个"摊开"方案,需要 host 侧 per-block scratch + 重编库。
3. 若不大:回到 push(131.8us,52%)和 reduce+尾巴(45.8us,19%),combine 离 170 还差 66us。
4. 查 16.6 的 recv 差异。

## 17. 量化 combine 搬到 TDM PULL(2026-08-02,节点健康)

起点:`fp8_blockwise` 在 PUSH 上 **3667.8us**,而不量化的 bf16 只要 417.5us(PUSH)/ 168.4us(PULL+QUAD)。
搬一半的字节反而慢 8.8 倍,量化等于白做。现在 **1386.4us**(64×8),2.6×,全部 rc=0。

### 17.1 根因:量化被钉死在 PUSH,而 PUSH 的发送是逐 lane 跨卡散写

`launch.cpp` 要求 blockwise 必须 `useExternalInpBuffer=true`,而 host 又把这个标志和 `_p2p/_nop2p`
一一绑定,于是 blockwise 只能 `_nop2p`。`_nop2p` 的发送是 `WarpQuantizeToCombineBlockwise` 直接量化进
**对端** staging,每 lane 8 字节跨卡写,拿不到 bf16 那条路的 TDM 批量搬运。

关键在于这两件事**可以拆开**:`useExternalInpBuffer=true` 保持不变(它才是"kernel 自己做量化"的开关),
只把 kernel 换成 `_p2p`。这样量化落到**本地** `combineInp`(intranode.hpp:2234 分支),对端来读。
`_p2p` 的 gather 侧本来就已经会取对端 scales(:3567),两头零件都是现成的,**只缺 `_p2p_fp8bwq` 这个符号**。

### 17.2 几何扫描(`MORI_COMB_QPULL=1`,noTIMING,开校验全部 rc=0)

EP4-4K bf16 hidden 7168,`--zero-copy 0 --quant-type fp8_blockwise`,dispatch 几何跟随 combine
(`DBN=SAME DWPB=SAME`)。GB/s 分母是 harness 固定的 202.47MB(bf16 口径),fp8 实搬约一半,
**这列只能横向比,不能当真实带宽读**。

| blocks | wpb | 总 warp | 传输 | combine us | GB/s | dispatch us |
|---|---|---|---|---|---|---|
| 64  | 8  | 512  | QUAD d4 | 1386.4 | 146 | 174.5 |
| 128 | 4  | 512  | chunk2  | 1393.6 | 145 | 176.8 |
| 96  | 8  | 768  | QUAD d4 | 1001.2 | 202 | 177.9 |
| 112 | 8  | 896  | QUAD d4 |  857.0 | 236 | 192.9 |
| 64  | 16 | 1024 | chunk2  |  771.6 | 262 | 163.2 |
| 128 | 8  | 1024 | QUAD d4 |  739.2 | 274 | 164.6 |
| 128 | 16 | 2048 | chunk2  |  434.8 | 466 | 162.5 |
| 256 | 8  | 2048 | QUAD d4 |  430.2 | 471 | 164.4 |

**决定性能的是总 warp 数,不是块数和 wpb 怎么分。** 三对等 warp 数的点各自吻合到 1% 以内:
512 → 1386.4 / 1393.6,1024 → 771.6 / 739.2,2048 → 434.8 / 430.2。而且几乎线性:warp 翻倍时间减半。
**dispatch 全程平在 162~193us,对几何不敏感**,它早就饱和了。

线性强扩展这个形状本身就是结论:这条路是**延迟受限**(warp 越多,在飞的对端读越多),不是带宽受限,
也不是描述符个数受限 —— 后者曾是我的判断,被这张表推翻,见 17.5。

### 17.3 删除法定价(64×8,chunk2,带 SKIPCHECK)

```
full 1410.7   NOQUANT 1412.9   NOREDUCE 1323.1   NOQUANT+NOREDUCE 1323.9
```

- **本地量化那一遍是免费的**(删掉反而 +2.2us,在噪声内)。它多读 117MB bf16、多写 58.7MB fp8,
  但完全被盖住。注意 bf16 PULL 走 `ZC=1` 时这一遍**根本不执行**,所以 168.4us 那个基线里没有这一项,
  不能拿它去反推这一相 —— `MORI_COMB_NOQUANT` 就是为此加的。
- 反量化 fold **87.6us**。
- 剩下 **1323us 全是对端读**。

### 17.4 QUAD 有正确性问题,当前靠门槛绕开(**未解决**)

- 深度 2 **算错**,深度 4 对(1387.1us)。两者都放得下 LDS(114KB / 229KB vs 320KB 预算),
  host 和 kernel 的预留也都对得上,所以不是布局错位,只可能是 ring 的定序问题。
- `128×4` + 深度 4 也**算错**(wpb=4 时每块只剩一个 QUAD 组)。
- **架构默认给的就是深度 2**,所以裸开 `MORI_COMB_QPULL=1` 会直接踩上错的那档。
  已在 intranode.hpp 的 QUAD 使能条件和 `_combine_shared_mem()` 两处加 `_qBufs >= 4` 门槛,
  两边必须同改否则布局错位。**这是护栏不是修复**,表里 wpb=16 和 128×4 那三行因此走的是分块路。

### 17.5 四个让实验/推理失效的坑(全踩过)

1. **`BASE='MORI_COMB_TDM='` 把被测对象关掉了。** 它是编译期开关,置空 = 整条 tile 路径没编进去。
2. **`else if constexpr` 是编译期链。** reduce 在 `if constexpr (UseFp8BlockwiseQuant)` 处分派,
   TDM 段是它的 `else`,所以 blockwise 匹配上之后那段**对它从未实例化**,任何运行期门控都救不回来。
   指纹是:分配了 tile、扫 chunk 却纹丝不动(4745.8@4 vs 4754.1@8)。chunk 是 tile 尺寸的直接因子,
   它不动就说明代码没跑。修法是把 blockwise 从 A/B 两个编译期分支的条件里摘掉,让它落进 `else`,
   并给运行期拒收留反量化兜底(`WarpAccumLF` 会把 fp8 当裸值加、丢掉全部 scale)。
3. **只换传输不换实现是负优化。** `_cPullType` 还拒绝 1 字节 TokT 时切到 PULL:4748.6us,比 PUSH 还差。
   跨卡散写变成了跨卡散读,而 PUSH 的 gather 至少读本地 staging。**赢的是 TDM 批量搬运,不是方向。**
4. **门控不进 `core.py` 白名单 = 不生效且不改缓存键**,A/B 等于拿同一个二进制跟自己比(§16 已记过一次)。

### 17.6 一条被推翻的旧结论(勿沿用)

曾用 `blockwise push 相 2472.6 − direct_cast push 相 572.6 = 1900us` 把这笔钱记成"量化数学",
据此算出地板 2520us 并判定 PULL+TDM 这个方向没救。**错的**:两条路的跨卡写宽度不同,减出来的是
`QuantizeStore` 每 lane 8 字节跨卡散写的代价,是传输。实测 1386.4 已经在那个"地板"下面。
量化数学的真实代价见 17.3:**≈0**。

### 17.7 本轮新增

```
门控(默认全关,已进 core.py 白名单)
  MORI_COMB_QPULL=1      blockwise 走 PULL gather(_p2p_fp8bwq)而非 PUSH staging
  MORI_COMB_NOQUANT      诊断,错结果:删 PULL 侧本地量化那一遍
符号
  EpCombineIntraNodeKernel_bf16_p2p_fp8bwq   (ep_intranode.hip;fp4 尚无 _p2p)
脚本
  _ct_err.sh             取某档的编译/运行错误(_ct_nobar.sh 的 grep 覆盖不到编译错)
  _ct_nobar.sh           DBN/DWPB 支持 SAME = 跟随 combine 几何
```

`MORI_COMB_QPULL` 仍是 opt-in,两个原因:`_p2p` 没注册 vec8 变体,开了就放弃 weightless top8/top9
快路;fp4 没有 `_p2p` 符号。

### 17.8 下一步(按优先级)

1. **fold 里的对端 scale 读**。现在每源每向量都去对端读一次 `srcScalePtrs[_j][_qSb]` —— 小、远、
   不跨卡缓存、延迟全暴露。这与 17.2 的"延迟受限且随 warp 线性"完全吻合,是当前头号嫌疑。
   改法:每 token 每源把 56 个 float 一次性预取进 LDS/寄存器,而不是每向量读一次。**先测再改**:
   拿一档把 scale 强行换成常量(错结果 + SKIPCHECK),差值就是这项的价。
2. 查 17.4 的 QUAD 深度 2 / wpb=4 定序 bug。在它修好之前 QUAD 只能跑深度 ≥4,而深度 4 在 wpb=16
   放不下,等于把宽块挡在 QUAD 之外。
3. 给 `_p2p` 注册 vec8 变体,把 weightless top8/top9 快路要回来。
4. 差距仍在:fp8 在 512 warp 上 1386.4us,bf16 在同样 512 warp 上 168.4us。搬一半字节还差 8 倍,
   1 修完再重新定位。

## 18. 分支切换 + ZC=0 落地 + 量化现状(2026-08-03)

### 18.1 先读这条:代码在哪个分支,否则会对着错的代码推理

| 分支 | 提交 | 内容 |
|---|---|---|
| `tdm-dispatch` | `521e11b6` | `fix(ep/intranode): separate barrier word for NOTIFY completion counter`,NOTIFY 时期 |
| `debug-aa` | `ee74e3c2` | **所有工作在这条上**,自 `6a4d7475` 起 65 个提交 |

`tdm-dispatch` 按指令回退到了 `521e11b6`,做出 980 GB/s BATCH dispatch 的两个提交(`e6896440` PERTOK、
`6a4d7475` 只留 BATCH)**已不在这条分支的历史里**,但通过 `debug-aa` 仍然可达,随时可切回。

三个克隆,别搞混:

- `c:\Users\fizhang\Desktop\work\mori` —— 本机,已切到 `debug-aa`。
- `C:\work\code\mori` —— SHAFIZHANG01 上的另一个克隆,曾停在 `a12bf9b2`。
- f01-2 容器 `/root/mori_tdm` —— 已同步到 `ee74e3c2`。

**强推到更早的提交之后 `git pull` 会报 "Already up to date" 并且一动不动**,因为远端新 tip 是本地的
祖先,合并祖先是空操作。这不是网络问题也不是没推上去,回退必须显式 `git fetch && git reset --hard
origin/<branch>`。这个现象浪费过一轮排查。

同步节点后**要查文件内容而不是只查 ref**(`tools/_ct_sync_aa.sh` 这么做):这棵树已经两次出现
"分支名对、盘上文件不是那个东西"。当前 f01-2 已验证:`intranode.hpp` 4439 行、两次 shuffle 的
`return _hi ? _r1 : _r0;` 在、`MORI_COMB_SCPRE` 宏与 `core.py` 的 `-D` 发射都在、两个微基准都在。

### 18.2 ZC=0(调用方自带输入 buffer)已落地,默认走 host d2d

`MORI_COMB_PULL` 不再是开关,而是选**谁**把外部输入搬进对端要读的对称 buffer:`host` / `kernel` / `off`。
gfx125x 默认 `host`。实测 64×8 EP4 bf16 hidden 7168 / 4096 tokens,**校验开着,rc=0**:

| 路线 | us | GB/s | 说明 |
|---|---|---|---|
| host d2d + 零拷贝 PULL(现默认) | 236.7 | 896.9 | 调用方 stream 上一次 d2d,内核与零拷贝逐位相同 |
| PUSH(旧默认,`MORI_COMB_PULL=off`) | 318.8 | 666.0 | 逐 lane 跨卡散写 |
| 零拷贝参考(同一次跑) | 169.0 | 1255.9 | 调用方能直接交 buffer 时仍是这个数 |

**开销完全对得上账**:236.7 − 169.0 = 67.7us,正好是那次拷贝本身(424 MB 本地流量 / 6.3 TB/s)。
除拷贝外没有别的损失,所以这条路没有藏着的问题,想再快只能不拷贝。

### 18.3 内核内暂存路线:根因找到了,而且它是个潜伏 bug

`MORI_COMB_PULL=kernel` 起初 rc=1,4 个 rank 错 3 个(max diff 3.75,tol 0.159)。真因:
**跨设备 barrier 的 release 侧只对 block 0 的前 worldSize 个线程 fence**,其余 block 写本 rank
`combineInp` 的 store 还在飞时对端 flag 就抬起来了,对端读到半成品。加 per-block fence 后
**rc=0 / 544.9us**,正确但比 host 路线慢 2.3 倍,所以默认不动。

两点必须记住:

1. 这个 fence 假设之前被判过"已否证"——**那次 A/B 根本没编译进去**,开关的 `-D` 没进构建缓存键,
   带 fence 的那次加载的是不带 fence 的二进制,于是"复现了它本该修掉的失败"。缓存键已改成直接从
   真实 `-D` 列表推导(`beac1b24`),`cache.py` 不再需要手工同步。**§17.5 第 4 条现在有第二个案例。**
2. **blockwise 走同一条暂存臂而今天能过,只是因为它的量化 pass 慢到自己把窗口关上了。**
   也就是说加速量化 pass 会把这个 bug 暴露出来。现在已经修了,但这条因果要记住。

顺带定下来的默认值:`QSTGU=7`(410.5us,对比 `=1` 的 718.8us,过 7 之后持平),`QWIDE=0`(实测为空,
保留只是为了下次不必重编就能再问一次)。

### 18.4 量化现状:**仍是净亏损,没有收益**

64×8 EP4 fp8_blockwise 删除法定价(带 SKIPCHECK):

```
full 1409.5    MORI_COMB_NOQUANT 631.1    →  本地量化/暂存 pass = 778us
内核启动 + 跨设备 barrier 合计 = 15.8us
```

两个病因,**都只在微基准里验证过,真实内核里一次都没验证**:

1. **量化 pass 的 778us 不是带宽**。它只搬 212 MB 读 / 106 MB 写,全本地。它是一条依赖链:
   load 16 B → subwarp 取 max → 广播 scale → 转换 → 存,存完才发下一个 block 的 load。
   14813 token / 512 warp,每 warp 28 个 block = 812 次串行迭代,每次约 1900 cycle,是个 load 延迟。
   **占用率救不了**:gather 的 tile 占了 116 KB LDS,一个 CU 只放得下一个 8-warp 块,一个 SIMD 两个 wave。
   `QSTGU` 就是为此加的(先发 N 个 block 的 load 再归约),718.8 → 410.5us。
   另外这条路上还有个纯浪费:`dstScales[0]` 写完又读回来取负,而 `dstScales` 是 uncached,
   等于每 token 为一个 float 付一次完整往返,已删。
2. **gather 侧每向量都去读一次对端 scale**。`srcScalePtrs[_j]` 是指向 uncached 分配的对端指针,
   这个 4 B 跨卡 load 就卡在最内层循环里,等它的算术全在后面排队。`g_micro`(只有这个 gather、
   四张卡、4096 token、chunk 3584、256 块 ×16 warp、每卡 105.8 MB 对端读)实测:

   | 变体 | us |
   |---|---|
   | 出厂(每向量读一次) | 288.5 |
   | scale 行进寄存器(每 token 每源读一次,shuffle 取出) | 136.4 |
   | scale 行进 LDS | 131.9 |
   | 把 fold 删掉只留传输(地板) | 89.4 |
   | bf16 搬两倍字节,同一个点 | 170.4 |

   **这一项决定量化值不值得做:它把"比 bf16 慢 1.69 倍"变成"快 1.25 倍"。**
   被否掉的变体:只 shuffle 不预取(一个 lane 加载、其余取它的值)是 `g_micro` mode 4,357.3us。
   它发出的 fabric 事务数和预取一样多,所以这是个有用的负结果——**代价不在事务个数,在串行化**。
   LDS 比寄存器快 4us 也否了:它要 `_combine_shared_mem()` 按字节对齐预留一块,
   静默的布局错位比 3% 更难查。

   寄存器版的边界:最多 4 个源、每 lane 最多 2 个 scale 条目,且 wave 级 shuffle 不进 sub-warp 尾巴
   (那里各 lane 的迭代次数不同)。超出边界的照旧直接读对端。

量化 pass 的另一半病因是**几何**,不是分配器:`q_micro` 同样 322 MB,64×8 要 720.5us,1024×16 只要
134.2us;而 uncached 和 cached 目标缓冲区**完全没差**(720.5 vs 720.4),之前怀疑的分配器被排除。

### 18.5 一条旧结论作废

**§17.3 的"本地量化那一遍是免费的"(full 1410.7 / NOQUANT 1412.9)是错的**,作废。
那次 A/B 是在缓存键还不诚实的时候测的,两档很可能加载了同一个二进制。
缓存键修好后重测是 full 1409.5 / NOQUANT 631.1,**这一遍要 778us,是当前最大的单项**。

### 18.6 已改但**一次都没在硬件上验证**的东西(下次开工第一件事)

`debug-aa` 整条分支**从未编译过**。本轮尝试跑第一次编译时被中断,没有结果。按此顺序补:

1. **编译**。`tools/_ct_aa1.sh` 是最小的一次:bf16 ZC=0 64×8 校验开着,对照 236.7us / 896.9 GB/s。
   这一档不碰 fp8,它要是错了就是分支本身的问题,与 scale 预取无关。
   注意 SCPRE 那段代码在 `_cPullBwq` 里,**必须 `QT=fp8_blockwise` 才会被编译到**。
2. **重测寄存器预取**。上表那个 136.4us 是**带 shuffle 索引 bug 那一版**测出来的:当时
   `__shfl(scr[j][sb / warpSize], sb % warpSize)` 广播的是**源 lane** 的寄存器选择,而不是把
   **调用方 lane** 自己的 `_hi` 位用上去。hidden 7168 时 scale 行 56 项跨 32 lane,两个半区都活着,
   所以选错半区就读错 scale。已改成两个半区都先 shuffle 再选(`return _hi ? _r1 : _r0;`)。
   微基准以前用的是常数 scale 行(全 ±2.0),读错也能加出对的和,所以没抓到;现在的行是每项 k+1。
   **修完的数一个都没有。**
3. **`MORI_COMB_PIPE=2` 带校验跑真实内核**。`b45a9d99`(把 blockwise 放进 pipelined gather)的提交
   信息上明确写着 NOT MEASURED——写完两台节点就都不应答 ssh 了。它藏在默认 0 的 `MORI_COMB_PIPE`
   后面并且 `#if` 掉整块,所以出货路径没被动过,**但在它跑出 rc=0 之前不要相信那段推理**。
4. **把量化 pass 从 64×8 挪走**。`q_micro` 说这是 5.4 倍的差距(720.5 → 134.2)。

### 18.7 本轮新增

```
门控
  MORI_COMB_SCPRE=0/1    默认 1。scale 行预取的自我 A/B,0 = 退回每向量读对端。
                         之前只能拿 MORI_COMB_QNOSC(整个删掉 scale,结果必错)去夹逼。
  MORI_COMB_PULL=host|kernel|off   谁来暂存调用方自带的输入 buffer,默认 host
  MORI_COMB_RELFENCE     kernel 暂存臂的 per-block release fence
  MORI_COMB_QPRE         量化拆成独立内核、用它自己的宽度启动
  MORI_COMB_QSTGU=7      量化 pass 一次发几个 block 的 load
  MORI_COMB_QWIDE=0      实测为空,保留只为免重编再问
  MORI_COMB_PIPE         blockwise 走 pipelined gather,默认 0,=2 未验证
工具
  tools/g_micro.cc       只有 gather、四卡的独立微基准。DEQ 模式见文件头;
                         mode 4 = 只 shuffle(已否),5 = 寄存器预取,6 = wave 内 block 号一致的变体
  tools/q_micro.cc       只有量化 pass、单卡。**单卡的活先跑**,四卡挂了也不会赔掉便宜的答案
  tools/_ct_sync_aa.sh   把节点检出切到某个 rev,并**查文件内容**确认
  tools/_ct_aa1.sh       debug-aa 的第一次编译 + bf16 ZC=0 基线
  tools/_ct_recreate.sh  宿主重启后重建容器(见 18.8)
  tools/_ct_bg.sh / _ct_bgread.sh   setsid nohup 脱离 ssh 跑,日志落 $HOME/bg_*.log
  tools/_ct_go1.sh       多段流水:先单卡量化,再四卡 gather
```

### 18.8 节点状态(截至 2026-08-03 22:00)

- **f01-2 可用,已验证**。傍晚经历过一次"docker 全废"(根因与处置见 19.9 / 19.10,**image 一个没丢**)。
  当前:`MORI-EPV2` 已 `docker start` 起来、torch 数到 4 张卡、`rocm-smi` 无 libdrm 报错、
  VRAM 回到 175MB/卡 idle、`/` 有 441G 可用,且**回归点已复现历史值**(19.11)。
- 更早的 13:00 记录:11:52 重启后回来,容器内 32 个 render node 与宿主一致。检出已在 `debug-aa`。
- **f01-1 不通**。轮询已停。
- 这次**不需要**重建容器。但 f01-1 上出现过宿主重启后容器抱着旧设备节点的情况:容器内只有 16 个
  render node、GPU[2] 起 libdrm auth 失败、torch 数到 0 卡。设备清单是容器**创建时**解析的
  `/dev/dri` 目录,`docker start` 不会重新解析,只能用 `_ct_recreate.sh` 提交快照后重建。
- **四卡 g_micro 会把节点跑挂**(进内核 30~60 秒后 ssh 不再应答,15~60 分钟不等)。所以四卡的活一律
  用 `_ct_bg.sh` 脱离 ssh 跑、日志落盘,并且**把单卡的活排在前面**。

### 18.9 一个不要引用的数字

记忆里有个"256×16 = 367.6us"的量化数,**没能从任何提交记录里复核出来,不要引用**,需要时重跑。
§17.2 那张几何表(256×8 = 430.2us)是有记录的。

## 19. op 尺寸定价曲线:量化在 64×8 的传输收益上限只有 15~24%(2026-08-03 下午,节点健康)

本轮只跑 EP SIM,不碰内核。产出是两条**统一口径**的 op 尺寸曲线(跨卡读 / 跨卡写),它们把
"量化到底能省多少传输时间"从推断变成实测。**结论:上限是 PULL −24%、PUSH −15%,不是 −50%。**
这就是 §18.4 那个"量化仍是净亏损"翻不过来的分母侧原因。

### 19.1 先读这条:**别拿 1383 GB/s 当 PUSH 发送的天花板去追**(纠我自己的错)

我本轮一度写下"send 实测 231us、结构上限 152us、差 79us,唯一没排除的候选是 32B 权重写"。
**三处都错**,已在 `intranode.hpp:2472-2487` 的注释里留有痕迹,按此更正:

- `231us` 是 **`MORI_COMB_SPREAD` 之前**的数(§15.1 的 236.6us)。SPREAD 之后发送段是
  **136.5us / 1483 GB/s**(§15.3),**已经比我算的 152us "上限"更快**。缺口不存在了。
- 之所以能反超:mode11 与 mode6/mode8 一样是 `slot%nPeers` **散列**口径,而 SPREAD 用大质数步长
  走 token,目的地轮转比散列更均匀。所以微基准在这里是**下限**,不是上限。§15.3b 已把 1:1 的参照
  系列全了(1316 / 1398 / 1461 → 内核 1470),并且判了"发送相位可以收工"。
- "32B 权重写是最后候选" **早被 §15.5 否掉**:`bench_dispatch_combine.py` 传 `weights=None`,
  那段跨卡写根本不执行,`MORI_COMB_NOWEIGHT` 删的是死代码。别再去测它。

**本章有价值的部分不是绝对带宽,是曲线的形状**——即 op 尺寸如何定价。

### 19.2 两条曲线(全部实测,同一工作集 268 MB / 8192 个 op,只有 op 尺寸在变)

`grid=64`、`block=256`、`rdN=1`、`NT=8192`、`EPSIM_STRIDE=16384`。
**STRIDE 必须与 op 尺寸解耦**,否则工作集随 op 一起变,cache 效应会伪装成 op 效率(见 19.6)。

**单位陷阱**:`EPSIM_RDELEMS` 和 `EPSIM_STRIDE` 都是 **bf16 元素**,不是字节
(`itemBytes = rdElems*2*rdN`,`_ct_epsim.sh:399-402`、`:454`)。所以下表 "op 字节" 一列
对应的参数是 `RDELEMS = 字节/2`(fp8 token 7168 B → `RDELEMS=3584`),而 `STRIDE=16384` 是
**32768 B** 的槽距。照抄字节值会差一倍。

```
 op 字节    READ us   PUSH us    READ GB/s   PUSH GB/s   READ ns/op
   512       56.1       —          74.8         —          6.85
  1024       57.1      65.2       147.0       128.6        6.97
  2048       58.4      66.6       287.5       251.8        7.13
  4096       60.8      68.7       551.8       488.3        7.42
  7168       65.6      72.4       895.6       811.5        8.01   <- fp8 整 token
  8192       68.1      73.1       985.5       918.4        8.31
 14336       85.9      84.9      1367.3      1383.0       10.49   <- bf16 整 token
 16384       94.0      90.5      1428.5      1482.5       11.47
 32768      180.2     163.1      1489.7      1645.9       22.00
```

拟合出的两参数模型,**两条曲线在 14 KB 附近交叉**:

```
              固定开销/op    渐近带宽     拐点(固定开销 = 传输时间)
 READ 跨卡       6.80 ns     1490 GB/s     ~10 KB
 PUSH 合计       7.79 ns     1650 GB/s     ~13 KB
```

外推到零字节:PUSH 是 7.79 ns × 8192 = 63.8us,而 1024 B 实测 65.2us ——
**最小 op 有 98% 的时间是固定开销**。小 op 时 PUSH 更慢(每轮两个描述符、两次排空),
大 op 时 PUSH 更快(渐近高 10%)。用 `PEER=0` 的本地读成本把 PUSH 拆开:

```
 本地 load  ≈ 2.2 ns      跨卡 store ≈ 5.6 ns      跨卡 load = 6.8 ns
```

**跨卡写比跨卡读便宜 18%**(写是 fire-and-forget,读要等往返),但 PUSH 白付一个本地 load。

### 19.3 **量化的传输收益上限**(本章的主结论)

同 `rdN=1`,字节整整少一半,时间只少这么多:

```
 路径    bf16 14336B   fp8 7168B    比值      收益
 PULL      85.9us        65.6us     0.764     −24%
 PUSH      84.9us        72.4us     0.853     −15%
```

原因就是 19.2 的拐点:bf16 token 在拐点之上(占渐近 92%),fp8 token 掉到拐点以下(占 49~60%),
**半个 token 买不到半个时间**。PUSH 拐点比 READ 还靠后(13 KB vs 10 KB),所以量化在 PUSH 上更不划算。

折算到真实字节量(64×8 / EP4 / hidden 7168 / 4096 token,combine payload = bf16 212 MB、fp8 106 MB。
`169.2us × 1254.7 GB/s = 212 MB`,与 fp8 那半的 106 MB 正好 2 倍,见 `intranode.hpp:2776-2778`)。
**下面这列时间是推断**(微基准带宽 × 真实字节量),只代表 payload 下限,不含路由、索引、barrier、权重:

```
 配置                            GB/s(实测)   折算用时(推断)
 bf16 14336B rdN=1               1383.0        153.3 us
 bf16 14336B rdN=1 PIPE=1        1409.8        150.4 us
 fp8   7168B rdN=1                811.5        130.6 us
 fp8   7168B rdN=1 PIPE=1         850.8        124.6 us
 fp8   7168B rdN=2 PIPE=1         968.7        109.4 us   <- 深度不对等,见下
```

最后一行**不能直接和 bf16 比**:bf16 的 `rdN=2 PIPE=1` 要 458 KB LDS,超 320 KB 硬上限,测不了;
`PIPE=0` 下它只要 229 KB 是可行的但**本轮没测**。所以"fp8 靠 LDS 占用小换到更大在飞深度"这个
**结构性优势尚未定价**,别引用 −27%。补法见 19.7。

### 19.4 瓶颈在接收端单卡的 fabric 入向,不在链路数、CU 或 TDM 引擎

`EPSIM_PEER` 把来源拆开实测(默认模式下 `p = slot%nPeers`,4 个来源含自己,所以约 1/4 是本地 HBM):

```
                8192 B/op      16384 B/op    说明
 PEER=0         3670 GB/s      6587 GB/s     只读本地 HBM
 PEER=1          957 GB/s      1330 GB/s     只读 rank+1,单条 xGMI
 默认            986 GB/s      1429 GB/s     4 个来源含自己
```

**单条链路 1330 就已经顶到四路混合 1429 的 93%** —— 四路混合并没有真的用上四条路的带宽,
一条链路单独跑就能把接收端喂满。所以单卡跨卡入向上限约 **1400 GB/s**,而本地 HBM 是它的 4.6 倍。

口径提醒:曲线里报的一律是 **GPU0 单卡的入向带宽**,不是 all2all 聚合(那要 ×4),也不是单链路。
`gb` 只累计本 rank 读进的字节,且只有 `g_rank==0` 打印。

顺带更正 §14 时期对"6.8 ns 固定开销"的归因:其中只有约 2 ns 是 TDM 引擎发射,**约 4.8 ns 是跨卡往返**。

### 19.5 并发摊不平 op 的固定开销(否掉"2 个 7K 等于 1 个 14K")

固定 7168 B、固定工作集,只改在飞数:

```
 rdN     GB/s      ns/op
  1      887.6     8.08
  2      999       ~7.2
  4     1147.3     6.25
```

**4 个 7K 在飞(1147)仍然打不过 1 个 14K 单发(1367)**。并发确实有用但很快饱和,
所以 **op 尺寸比在飞数量更决定效率**。推断(未定性):每 op 的固定成本占用的是某个**串行资源**
(如 TDM 描述符处理槽)而非纯延迟,故并发无法隐藏。

同一结论在 PUSH 侧的独立印证:同一 token 的 load→store 有依赖,但相邻 token 之间没有,
让第 t+1 轮的 load 与第 t 轮的 store 同时在飞(`EPSIM_PIPE=1`,每轮只排空一次),
**只值 2~5%**(72.4→69.0us / 84.9→83.3us),不是预期的量。`PIPE=2`(LDS 双缓冲)无额外收益。
**排空次数不是瓶颈**,跨卡 store 的占用才是。

### 19.6 两个让实验失效的坑(都踩过)

1. **STRIDE 跟着 op 尺寸走 = 工作集跟着变**。原先槽距默认取 `rdElems`,于是 3584 B 的 op 只摸
   29 MB、14336 B 的摸 117 MB,**cache 命中率差异被读成了 op 效率差异**,我因此一度得出"64 block
   已经饱和"的错结论。已加 `EPSIM_STRIDE` 解耦(默认 0 = 沿用旧行为以兼容,新实验一律显式给)。
2. **`WORK` 变量名撞车**:`_ct_bg.sh` 用 `WORK` 传脚本路径,`_ct_epsim.sh` 又用它当 mode10 的 FMA
   数量 ⇒ 每次都传了 `EPSIM_WORK=/tmp/_ct_epsim.sh`,`atoi` 得 0,**mode10 的深度扫描静默失效**。
   已改名 `EPSIM_FMAWORK`。mode9/11 不受影响。

### 19.7 下一步(按优先级)

1. **补 `PIPE=0` 的深度对等对比**,把 19.3 最后一行的不对等修掉:
   `RDELEMS=3584:2`(114 KB)/ `7168:2`(229 KB)/ `3584:4`(229 KB),全在 320 KB 之内。
   这才能回答"fp8 的 LDS 优势值多少"。
2. **给 mode11 加 SPREAD/RR 口径**。内核有质数步长(§15.3)和计数排序 RR(`intranode.hpp:2490-2587`),
   mode11 还是散列,所以它给的是下限。补上才能和 1483 GB/s 同框比较。
3. 分母侧已经清楚了,**量化要翻盘只能在分子侧**(量化 pass 本身 + fold/dequant),见 §18.4。
   传输侧最多还这 15~24%,别再指望 50%。

### 19.8 本轮新增

```
 tools/_ct_epsim.sh
   mode11              COMBINE PUSH:本地 HBM load -> LDS -> 跨卡 store 到 peer staging。
                       无 D2D、无 gather、不量化,纯 payload 发送方向。
   EPSIM_STRIDE=N      槽距,单位 bf16 元素(×2 = 字节),与 op 尺寸解耦(见 19.6)。
                       0 = 沿用 stride==rdElems 的旧行为。
   EPSIM_PIPE=0/1/2    0 = 每轮排空;1 = 相邻 token 的 load/store 重叠;2 = 再加 LDS 双缓冲。
   EPSIM_FMAWORK       原 EPSIM_WORK,改名避开 _ct_bg.sh 的 WORK(见 19.6)。
   EPSIM_PEER=0/1      来源分解:0 = 只读本地,1 = 只读 rank+1 单链路。
   _lds_ok()           bash 侧预检,**在 docker exec 之前**挡掉超 320 KB 的配置(见 19.9)。
```

### 19.9 事故:LDS 超预算时"钳制"而不是"不启动"

红线已进常驻规则(`.cursor/rules/evidence-only.mdc` 第 0 节),此处只留事实与定位。

**直接原因**:host 侧把超预算的 LDS 申请**钳到 320 KB 仍然启动**,而内核 tile 偏移按**未钳制**的
`rdN × rdElems × nbuf` 寻址(`7168:2 PIPE=1` 要 458 KB)⇒ wave 写出边界 ⇒ 挂在**不可杀的 D 态**、
抱着 GPU 不放,并**毒掉排在它后面的整个队列**(push2/push3 全部超时,当时误判成节点退化)。

**连带**:5 个 core dump 共 **115 GB** 落在 `/var/lib/apport/coredump/`,撑满 `/` ⇒ `dockerd` 起不来
(`Unable to get the TempDir under /data/docker: ... no space left on device`)⇒ systemd 判
`start request repeated too quickly` 放弃重试 ⇒ 表现为"docker 全废、image 全没了"。
**image 一个都没丢**(`/data/docker` 完好:11 个仓库、`overlay2` 655 层),只是 daemon 没跑。处置见 19.10。

**已修**:bail 而非 clamp,外加 `_lds_ok()` 在 `docker exec` 之前预检。

```324:334:mori/tools/_ct_epsim.sh
  // MUST bail, not clamp. Clamping used to look harmless because the kernel still launched, but
  // the tile offsets are computed from rdN/rdElems/nbuf and keep addressing the un-clamped size,
  // so the wave writes past the LDS it was given. That hangs the kernel in an unkillable D state,
  // holds the GPU, and silently poisons every configuration queued behind it.
```

### 19.10 "docker image 全没了"的正确处置(实操过,有效)

**不要直接 `systemctl start docker` 就完事,更不要重装或改 `data-root`** ——
dockerd 若在 data-root 不可用时起来,会重新初始化一个空的,**那才会真丢 image**。先只读确认:

```bash
sudo cat /etc/docker/daemon.json                       # data-root 在 /data/docker,不是 /var/lib/docker
sudo ls -la /data/docker                               # 该有 image/ overlay2/ containers/ volumes/
sudo ls /data/docker/overlay2 | wc -l                  # 655 层 = 数据在
sudo python3 -c "import json;print(list(json.load(open('/data/docker/image/overlay2/repositories.json'))['Repositories']))"
sudo journalctl -u docker -n 60                        # 拿到真实失败原因(非 root 看不到,会显示 No entries)
```

确认是空间问题后:

```bash
sudo rm -f /var/lib/apport/coredump/core.*             # 本次回收 115G,/ 从 81% -> 74%
sudo systemctl reset-failed docker.service docker.socket   # 关键:不 reset 就一直是 "repeated too quickly"
sudo systemctl start docker && docker images mori-epcheck
```

要点:
- **`df -h` 会骗人**。事后看 `/` 有 327G 可用,但 dockerd 失败发生在磁盘真满的时刻;
  单元早已放弃重试,所以"现在有空间"和"docker 现在能起"是两件事,必须 `reset-failed`。
- 容器无需恢复:`_ct_*.sh` 都是自己 `docker run` 临时容器(`epchk_*`),只要 image 在就行。
  `docker ps -a` 里一片 `Exited (255)` 是 daemon 重启造成的,不用管。
- `rocm-smi` 报 `GPU% 100` 而 `VRAM% 0` 时先查持有者:本次是 `./inb-node-agent -slot-id=1`
  (集群管理 agent,`futex_wait`),**不是残留的计算内核**,不要因此去 reset GPU。
  已用回归点证实这个 `GPU% 100` **不吃算力**(见 19.11),别再被它误导成"节点在忙"。

### 19.11 恢复后的回归基线(**每次节点重启/挂过之后先跑这一个点**)

绝对值只在同一环境内可比,所以恢复后第一件事是回归而不是新数据。这个点历史上测过三次:

```
 MODES=9 GRIDS=64 BLOCK=256 NT=8192 ITERS=50 RDELEMS=3584 RDN=1 PIPE=0 STRIDE=16384
 期望自检行:tiles/peer=8192  268MB/peer  iter=50  block=256 wpb=8 ldsBytes=131072 stride=32768B

 2026-08-03 早    895.6 GB/s   65.6 us
 2026-08-03 早    887.2 GB/s   66.2 us
 2026-08-03 夜    885.8 GB/s   66.3 us   <- docker 恢复后,与第二次差 0.2%
 2026-08-04 上午  879.9 GB/s   66.7 us   <- 整机 reboot 后,新开机
 2026-08-04 上午  881.2 GB/s   66.6 us   <- 同上,与前一次差 0.15%
```

落在 **879~896 GB/s / 65.6~66.7 us** 就算环境可比,前面所有曲线不用重测;偏出去则先查环境再谈结论。

**窗口是按开机分段的,别跨开机比绝对值**。上面前三次是同一次开机内的,后两次是 8/4 reboot 之后的:
组内彼此 0.15~0.2%,组间差 0.6~0.7%。0.6% 不是退化,是开机之间的正常差异(时钟/显存训练),
两次复测都落在同一个点上就说明环境是稳的。判据用「同一次开机内 ±1%」,别拿今天的数去够昨天的窗口。
**先核对那行 `[cfg]/[cfg2]` 自检**:NT、stride、ldsBytes 任一不同,数字就不可比(尤其 STRIDE 的单位是
元素,见 19.2 的单位陷阱)。注意 `ldsBytes = max(TE 基础 131072, mode9 的 need)`,所以小 rdElems 下
它恒为 128KB,不随 op 尺寸变。

分级验证的顺序(本次实操,推荐照做):先 `NT=2048 ITERS=5` 跑一个探针,只看 `RUN_EXIT=0`、
不看绝对值(该口径下是 709.7 GB/s / 20.7us,**工作集只有 67MB,不可与上表比**),
确认不挂再上标准口径。四卡的活一律用 `_ct_bg.sh` 脱离 ssh 跑。

收尾自检(想看到的样子):`/var/lib/apport/coredump/` 空、`ps -ef | grep [t]dm_epsim` 为空、
`rocm-smi --showmeminfo vram` 回到 **175 MB/卡** 的 idle 基线。

另外 `_ct_epsim.sh` **不能用 `-Script` 直发**:它内联了 CUDA 源码,base64 后超过 Windows 32k 命令行
上限,`ssh.exe` 会报 `The filename or extension is too long`。正确调法是当作 `-Aux` 载荷:

```powershell
powershell -File tools/_send_ct.ps1 -Script tools/_ct_bg.sh -Aux tools/_ct_epsim.sh `
  -Envp "WORK=/tmp/_ct_epsim.sh TAG=reg MODES=9 GRIDS=64 BLOCK=256 NT=8192 ITERS=50 RDELEMS=3584 RDN=1 PIPE=0 STRIDE=16384" -Tmo 150
powershell -File tools/_send_ct.ps1 -Script tools/_ct_bgread.sh -Envp "TAG=reg" -Tmo 90
```

## 20. PUSH combine:64×8 下瓶颈是 fold 的 LDS 读(2026-08-03/04,ab05e05)

bf16 EP4 4K h7168 `ZC=0`,`BASE=MORI_COMB_PULL=off`,全部 rc=0。

> 本节标题原为"瓶颈是占用率,不是 fold 实现"。**那是错的**,照它去调 CBN 会撞规则(见 20.4)。
> 占用率只解释 20.1 的 CBN 扫描;在规则强制的 64×8 上,瓶颈是 fold,而 fold 里最大的一块是
> LDS→寄存器(20.5)。

### 20.1 `CBN` 扫描:PUSH 在 `CBN=256` 反超 PULL(**仅供诊断,不是可用方案**)

```
CBN=64  311.5us   CBN=128  235.8us   CBN=192  228.0us   CBN=256  221.1/220.9us (960 GB/s)
```

PULL 的参照是 236.7us。**PUSH 之前"输给 PULL"完全是几何造成的**,不是传输结构。
fold 的 tile 钉住 115712 B LDS ⇒ 一个 CU 只放一个 block ⇒ `CBN=64` 时 256 个 CU 里 192 个空着。
192→256 只再省 7us,说明已经收敛到 send 段。

### 20.2 分段(删除法,同一几何两档对照)

| 段 | CBN=64 | CBN=256 |
|---|---|---|
| token send (`NOPUSH`) | 139.5 | 150.0 |
| gather+fold (`NOGATHER`) | 157.8 | **45.3** |
| ─ 其中 fp32 算术 (`NOREDUCE`) | 49.1 | 12.4 |
| barrier 等待 (`NOBAR`) | 8.7 | 22.6 |
| 残差 | 5.9 | 3.0 |
| **合计 / 实测** | 311.9 / 311.5 | 217.9 / 220.9 |

- **send 150.0us 已经到顶**:212.3MB / 150.0us = 1415 GB/s,§8 记的 TDM 天花板是 1582。没有空间。
- **barrier 从 8.7 涨到 22.6**:块数翻四倍,栅栏 atomicAdd 争用变贵。这是 CBN=256 唯一的退步,
  也是现在唯一还剩的可摘项 —— 但 §16 已把栅栏方向的招试遍了,别重来。

### 20.3 两个被推翻的假设(**别再试**)

1. **"LDS 中转是浪费,该学 DeepEP 用寄存器直读"** —— 反了。`MORI_COMB_FOLDVEC=1` 把 fold 换成
   16B lane gather:596.1us vs 311.5(64x8)。`tdm_redsim` 同向:mode0 102.19us vs mode2 32.15us。
   wave32 下一次 lane gather 每个 source 只搬 512 B,tile 一次搬 1792 元素。**请求数少而大才赢**,
   省下的那趟 LDS 往返远抵不上。门控保留在 `_cPullOk`,是这条结论的复现路径。
2. **"staging 布局 `destPe*maxTok+tokenId` 让一个 token 的四份相隔 58.7MB,毁了局部性"** ——
   不成立。`tdm_redsim` 的 `RED_ADJ=1` 把布局转成 DeepEP 的 `tokenId*nSrc+destPe`(相隔 14 KB),
   字节量、描述符、fold 全不变,只有地址变:**32.22 vs 32.15us,差 0.2%**。
   TDM gather 的跨步描述符对 pitch 不敏感,所以改 `SendBufSlotOffset` 这种大改**没有收益**。

那个 2701 GB/s 是占用率假象:CBN=64 只有 1/4 的 CU 在跑。同一段在 CBN=256 是 45.3us ≈ 8 TB/s,
和 `tdm_redsim` grid=256/wpb=8 的 30.79us 对得上(差值是 routing 和真 kernel 的尾巴)。

### 20.4 几何是**硬约束**:64 CU × 8 warp,20.1 的 CBN 扫描不算成绩

`.cursor/rules/ep-dispatch.mdc` 第 27 行:**必须用 64 CU × 8 warp 达标,禁止"用满整芯片掩盖延迟"**,
并且"若某方案只有在大量 CU 才达带宽、64 CU 掉很多,则未达标"。

所以 20.1 的 `CBN=256 → 220.9us` **不是成绩,是把账赖给 GEMM** —— 推理时那些 CU 是 GEMM 的。
它只能当诊断用:它证明 fold 随 warp 数线性缩放,即这条路是延迟受限。真正要打的是
**64×8 下的 311.5us**,而 fold 在这个几何下是 157.8us。

### 20.5 fold 在 64×8 下的成本构成(`tdm_redsim`,单卡,COPIES=4)

| 段 | 隔离方式 | us | 折算 |
|---|---|---|---|
| TDM global→LDS | mode 4 | 30.21 | 8947 GB/s,**已打满** |
| LDS→寄存器 | mode 6 − mode 7 | **50.81** | 211MB / 4159 GB/s |
| bf16↔fp32 转换+累加 | mode 5 − mode 6 | 6.31 | 几乎免费 |
| 输出写回 global | mode 7 | 19.80 | 58.7MB / 2965 GB/s |
| 完整 fold | mode 2 | 98.10 | |

**传输不是瓶颈,算术也不是,瓶颈是把数据从 LDS 读回寄存器。** 这条否掉了"换 packed bf16 加法"
(最多省 6.3us)。加深每 warp 在飞深度(mode 3 双缓冲)只动 0.4%,与 §8 `PAYBUF`/`PAYSPLIT` 同结论。

### 20.6 `MORI_COMB_FOLDU`:sim 省 21%,真 kernel 只省 1.8%

把 fold 的 source 循环从运行时上界(`_nRed`,编译器不展开,四次 `ds_read_b128` 串着发)改成
编译期上界 + 运行时门槛(`_cRedSrcMax=4`,照 `_cScSrcMax` 的写法),先读齐再累加。

```
tdm_redsim 64x8:  98.10 -> 77.87us  (-21%)
真 kernel  64x8: 314.6 -> 309.3us  (-1.8%)   两档 rc=0,组内波动 <1us
```

**别拿 redsim 的数当 kernel 的数。** sim 的 fold 是 98.10,真 kernel 的是 157.8,
中间 1.6 倍(约 60us)是 sim 没模拟的:routing、weights、`MultiWarpIter` 分派、与 barrier 的交互。
批量发 `ds_read` 只作用在 sim 覆盖的那部分,所以收益落在更小的基数上。
**那 60us 是下一个该查的东西,目前完全没有拆解。**

### 20.7 那 60us 的去向:不是等待,是**寄存器溢出**

**`MORI_COMB_NOWAIT`(删掉 fold 对自己 TDM load 的 `s_wait_tensorcnt`)**:

```
base   314.7us  (rc=0,带检查)     nowait  295.3us     差 19.4us
```

所以"每 token 串行 issue→wait→fold"**只值 19.4us**。PUSH 路径下 `_cPullBufs` 恒为 1
(双缓冲被 `UseP2PRead` 挡住,PUSH 的 fold 复用 send tile),但把它做成双缓冲的天花板就是这 19.4us(6.2%),
而且要 LDS 翻倍(115712×2 = 231424 < 320KB 装得下)。**先别做,下面这条更大。**

把等待剥掉之后账反而更难看:

| | fold 里不含等待的部分 |
|---|---|
| 真 kernel | 157.8 − 19.4 = **138.4us** |
| `tdm_redsim` | 98.10 − 30.21 = **67.9us** |

同一套 LDS 读 + 算术 + store,真 kernel 慢一倍还多。原因在 hsaco 里(`tools/_ct_hsaco_regs.sh`):

```
scratch=192  spill(v/s)=29/0  vgpr=128  sgpr=94   EpCombineIntraNodeKernel_bf16_nop2p   <- 我们跑的
scratch=176  spill(v/s)=26/0  vgpr=128  sgpr=94   EpCombineIntraNodeKernel_bf16_p2p
scratch=64   spill(v/s)=0/0   vgpr=96   sgpr=90   ..._bf16_nop2p_fp8cast
scratch=0    spill(v/s)=0/0                       EpDispatchIntraNodeBatchKernel_bf16
```

**29 个 VGPR 溢出,192 B scratch**(= `srcPtrs[8]`+`srcWeightsPtr[8]`+`srcScalePtrs[8]` = 3×8×8,正好)。
128 这个上限的来源:没有 `__launch_bounds__` 时 clang 按 `flat_work_group_size(1,1024)` 编,
1024 线程要同时驻留 ⇒ 每 wave 只能 128 个 VGPR。
**但 gfx1250 是 wave32**,8 warp = 256 线程,不是 1024。
`MORI_COMB_LB` 的说明写着"设成 `warp_per_block * 64`" —— 那是 wave64 的算法,给出 512,等于没放宽;
正确值是 **256**。放宽不花占用率:115712 B 动态 LDS 早就把每 CU 钉成一个 block 了。

`:3158` 那条"`__launch_bounds__` 挪不动这个 pin"的旧结论**存疑**,大概率是按 warps×64 试的。

### 20.8 反汇编给出根因:**每个 source 一次全等待**,四个 `ds_load` 根本没并行

先把 fold 隔离出来量绝对值(**用 NOPUSH,别再在 314 的总量上做减法**):

```
base 314.7   nopush 176.2   nopush+nogather 15.6   =>  fold = 160.6us
```

与 §20.2 减法得到的 157.8 互相印证(差 1.8%)。**以后 fold 的 A/B 一律在 nopush 基线上做**:
176.2 的底噪下 5us 是 2.8%,314.7 的底噪下同样 5us 只有 1.6%,FOLDU 那轮差点被噪声吃掉。

反汇编 `EpCombineIntraNodeKernel_bf16_nop2p`(`_ct_hsaco_regs.sh DIS=1`),fold 每个 source 编译成:

```
s_and_saveexec_b32 s12, s4          <- 每个 source 一次 exec mask
s_cbranch_execz 30
s_wait_loadcnt_dscnt 0x0            <- 每个 source 一次**全**等待
v_and_b32_e32     v125, 0xffff0000, v34     \
v_lshlrev_b32_e32 v124, 16, v34              > bf16->f32 就是取高 16 位,这部分没问题
v_alignbit_b32    v1, v35, v34, 16          /
v_pk_add_f32      v[122:123], v[122:123], v[124:125]
s_or_b32 exec_lo, exec_lo, s12
```

`_CROW_DEAD(_j)` 的 `continue` 把每个 source 的读放进各自的 exec-mask 基本块,
编译器于是在**每一块开头**插 `s_wait_loadcnt_dscnt 0x0`(等**全部**未完成的 load,不是部分等待)。
⇒ 四个 `ds_load_b128` 严格串行,**源码里怎么排都没用**。
**这就是 FOLDU 只值 1.8% 而 sim 值 21% 的原因** —— sim 里没有这个 mask,读本来就是连续的。

`MORI_COMB_FOLDB`:读循环不再跳过,dead/越界 slot 一律读 row 0 再在累加时丢掉
(gather 本来就会把 dead slot 取进 LDS,见 `TdmShapeGather` 注释),四个读连续发、一次等待。

**实测(rc=0,带正确性检查)**:

```
fold 隔离(nopush 基线)   160.6 -> 133.6us   -16.8%
整个 combine             314.7 -> 287.9us   -8.5%
```

两条独立路径差 0.2us,互相印证。**是 FOLDU 的五倍,而且来自删一个分支,不是重排读。**
PULL 参照 236.7,差距从 78 缩到 51。

机制复核:反汇编指令数 6684 → 6894,新出现 99 个 `s_wait_dscnt`(只等 LDS 的**部分**等待),
原先清一色 `s_wait_loadcnt_dscnt 0x0`(等全部)。
**注意**:按 `v_pk_add_f32` 定位 dump 出来的那段改前改后逐字节相同 —— 那是
`_nRed > _cRedSrcMax` 的 fallback 串行循环,FOLDB 没碰它。别把它当成"改动没生效"。

其他从直方图读到的事实(6684 行,`EpCombineIntraNodeKernel_bf16_nop2p`):
`s_and_saveexec_b32` 295 / `s_cbranch_execz` 264 / `s_wait_loadcnt_dscnt` 169 / `v_pk_add_f32` 252。
**没有 `v_cvt_*_bf16`** —— bf16→f32 是 `v_and 0xffff0000` + `v_lshl 16`,已经是最优形式,别再往这上面想。

### 20.9 工具

- `tdm_redsim.cc` 加了 `RED_ADJ`(0=线上布局 / 1=相邻)、`RED_UNROLL`(0=原样 / 1=先读齐再累加 / 2=一次两位置)、
  `MODE=6`(读 LDS 但用位或代替算术,隔离读)、`MODE=7`(只写输出,隔离存)。
  `_ct_redsim.sh` 用 `ADJ=` / `UNROLL=` 透传。
  grid 扫描(wpb=8, COPIES=4):`64→98.30us  128→53.27  192→41.70  256→30.79`,近乎线性。
- **`_ct_hsaco_regs.sh`:资源用量 + 反汇编,不跑 GPU,占卡时也能查。**
  `DIS=1 PAT=<助记符> SKIPM=<跳过几个匹配> DIRPAT=<jit 目录 glob>` 定位并 dump 指定代码段。
  **`DIRPAT` 必须给**:删除法跑完后最新的缓存目录是**被删过的**那个构建,反汇编它等于问错问题。
  四个坑,每个都吃过:
  1. `.hsaco` 是 clang offload bundle 不是 ELF,`llvm-readelf` 直接读会说
     "not recognized as a valid object file"(像文件不存在,其实是包着的),要先 `--unbundle`。
  2. metadata **按键名字母序**排:`.vgpr_count`/`.sgpr_count` 在 `.private_segment_fixed_size`
     **之后**,`.group_segment_fixed_size` 在 `.name` **之前**。按出现顺序取值会打出一片空列。
  3. **gfx1250 的助记符不是 CDNA 那套**:LDS 读是 `ds_load_b128` 不是 `ds_read_b128`,
     等待拆成了 `s_wait_dscnt` / `s_wait_loadcnt_dscnt` / `s_wait_xcnt`,没有 `s_waitcnt`。
     按固定名字 grep 会安静地报 0 —— 第一次就把"fold 里一次 LDS 读都没有"当成了结论。
     **先出助记符直方图,别预设名字。**
  4. 脚本体是 `bash -lc '...'` 的单引号参数,**注释里一个撇号(`fold's`)就会截断整个字符串**,
     报错出现在几十行之后的 `done`,完全看不出是注释的问题。
- `_ct_bgwork.sh`:`_ct_bg.sh` 用 `bash $WORK` 启动,不传环境变量,所以参数只能写在这个脚本里。
  **每次改它,别为每个实验新建一个 launcher** —— `tools/` 长到几百个脚本就是这么来的。
- `_ct_bgdiag.sh`:后台 sweep "看着卡住"时先跑它 —— 读 `.ep_test_last`(每档 start/done 各一行,
  有 start 没 done 的那档才是嫌疑)、各档 `/tmp/ep_test_*.log`、容器内完整进程树。**不碰 GPU**。
- **`_ct_aa1.sh` 曾经用 `tail` 收尾,已改成 `tee` + 行缓冲 `grep`。** `tail` 要等管道 EOF 才吐字,
  于是后台跑时 `bg_<tag>.log` 全程停在三行,"跑完了"和"挂死了"从外面看一模一样 ——
  FOLDU 那轮为此白查了一遍,实际它早在两分钟前就写好 `done ... rc=0` 了。全量在容器 `/tmp/aa1_full.log`。
- `_ct_aa1.sh` 现在是通用入口:`REV=`(拉分支)、`CBNS=`(逗号分隔,内部转空格)、`RMCORE=1`(清 core dump)。
  **`REV` 必须按分支名 fetch**:容器是 shallow clone 且 refspec 只有 `tdm-dispatch` 一条,
  `git fetch origin` + `git reset --hard origin/debug-aa` 会 fetch 成功、reset 失败、
  然后**用旧二进制跑出一个看着合理的数**。第一次 FOLDVEC A/B 就这么废了一轮(596 那次之前的 311.6 vs 311.5)。
  永远核对输出里的 `HEAD=`。
