# InterNode V2：Dispatch / Combine 优化结论

本文按两个OP记录本次尝试、效果和是否采用；**带宽、延迟及前后对照统一放在 [bw.md](bw.md)**。

<!-- BEGIN EPV2 HANDOFF 20261008 -->
## 运行进度与换机交接（2026-10-08 06:43 UTC）

**已按用户要求停止扫描。** 控制器已退出，未完成的 trial 保留为 interrupted，不计入有效数据。g17 临时暂停的 MORI CI listener 已恢复；其守护进程和 g09 竞争任务守护均已退出。两机 benchmark 容器保留，方便继续使用。此次只更新交接文档，没有采用新的最优配置或更新默认 JSON。

### 停止时的准确进度

- 数值门槛：12/12；新口径原配置 baseline：36/36，逐样本离线复算通过。数值门槛不等于最终严格正确性验收。
- 全量扫描：**22/36 个完整形状，67/108 个有效阶段**。每个形状包含 Dispatch 扫描、Combine 扫描和该轮检查三个阶段。
- 停在 **FP8 E4M3 → BF16 / K8 / T256 的 Combine**。该形状 Dispatch 已通过；Combine a001 被用户停止，续跑时重新完成 Combine，然后检查，不能采用这次中断的数据。
- 尚未执行完整 shortlist 筛选、独立三组 A0/B/A1 确认、共享配置合并及全36 fresh metric + 全36 strict 验收。**目前没有本轮已确认的新最优配置。**

| 组合（总专家数256） | 已完整扫完的 tokens | 剩余 |
|---|---|---|
| BF16 → BF16，TopK 6 | 16、32、64、128、256、512、1024、2048、4096 | 无 |
| BF16 → BF16，TopK 8 | 16、32、64、128、256、512、1024、2048、4096 | 无 |
| FP8 E4M3 → BF16，TopK 8 | 16、32、64、128 | T256 Combine/检查；T512–4096 全流程 |
| FP8 E4M3 → FP8 E4M3，TopK 8 | 无 | T16–4096 全流程 |

冻结条件：commit `33ae5bf9a49df66a966523a10a7ae2f41684017b`，MI355X/gfx950，g17+g09、EP16、H7168、experts/rank16、QP2、Ionic、CCQE关闭、uniform routing。所有 token 档均显式 `--kernel-type v2`，不是 auto/v2_ll。T16/T32 的 capacity 为64，其余 capacity=T。性能阶段 Combine weights=None；baseline/search 为3×30、warmup20、drop1，每阶段1392个保留样本；确认/final为3×100、每阶段4752个样本。D+C 不含 dtype 转换，主带宽是逐样本平均的 RDMA 算法 payload GB/s。

FP8→BF16/K8/T2048 的异常原基线已另做一次固定原配置刷新：D **457.90 µs / 64.08 RDMA GB/s**，C **834.10 µs / 70.24 RDMA GB/s**，4752样本/阶段。原异常值 D8266.95/C10169.08 µs 保留，不能用它计算优化收益。刷新只更新描述性 baseline，不是新配置收益确认。

### 目录、机器与证据

| 用途 | 实际位置 |
|---|---|
| 用户查看/提交 PR 的仓库 | `/var/zqz/mori`，分支 `dev/analyze_v1` |
| 扫描控制脚本、状态、日志 | `/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930` |
| 共享实验归档（宿主机） | `/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260917` |
| 本轮 runtime（宿主机） | 上述归档下的 `rescan_20260930/` |
| 容器内实验根目录 | `/experiment/rescan_20260930` |
| 本轮实际冻结源码 | `/experiment/rescan_20260930/source` |
| 容器本地运行包 / JIT cache | `/opt/mori-20260930/pkg` / `/opt/mori-20260930/jit-cache` |
| 两机与 node rank | g17=`pit2-p03-g17`/`10.19.0.117`/node0；g09=`pit2-p03-g09`/`10.19.0.109`/node1 |
| 两机容器名 | `mori_epv2_recheck_0917` |

控制目录中的直接入口：

- [scan_state.json](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/scan_state.json)：逐项进度、每次 attempt、metadata 路径及 SHA；[pipeline_state.json](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/pipeline_state.json)：控制器阶段状态。
- [最后一轮队列日志](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/pipeline_logs/scans_a015.log)；每次 trial 的 `logs/<tag>/metadata.json`、`node0.log`、`node1.log`；共享 runtime 下的 `evidence/`、`scan_tables/`、`scan_table_history/`。
- [冻结的36格 baseline audit](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/baseline_audit_20261008.json)；[成功的异常点刷新](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/baseline_refresh_fp8_to_bf16_k8_t2048_after_preflight_20261008.json)。最终报告使用这份冻结 baseline audit，不能改用后续累计的 `audit.json`。
- [完整后续命令](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/post_scan_handoff_commands_20261008.md)：扫完108个有效阶段并完成 audit 后，依次执行 shortlist、独立确认、共享表选择、merged validation 和最终报告。
- [停止及清理核验](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/user_stop_verified_20261008.json)；[CI恢复记录](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/ci_listener_resume_user_stop_20261008.json)；[CPU绑核清理核对](/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930/cpu_isolation_restoration_inventory_20261008_final.json)。此前修改过的131个临时容器已全部删除，无残留绑核需要恢复；当前 observer index 为 `{"schema":1,"hosts":{}}`，两机均使用严格竞争检查。

`bw.md` 保留基线和历史数据，其中旧的7/36进度尚未同步；**当前停止进度以本节和 scan_state.json 为准**。控制脚本、原始大样本不加入 PR，换机时需要保留上述归档目录；只带走仓库无法恢复这些实验状态。

### 原 g17/g09：启动已有容器与续跑

以下是操作方法，本次停止后没有自动执行续跑。容器目前保留；若已停止，在登录机执行：

```bash
mori_ssh=(ssh -F /dev/null
  -i /home/qizhou.zhang@amd.com/.ssh/cluster_id_ed25519
  -o UserKnownHostsFile=/home/qizhou.zhang@amd.com/.ssh/known_hosts
  -o BatchMode=yes -o ConnectTimeout=10 -l qizhou.zhang@amd.com)
for mori_host in pit2-p03-g17 pit2-p03-g09; do
  "${mori_ssh[@]}" "$mori_host" 'docker start mori_epv2_recheck_0917'
done
```

在 **g17 宿主机**，确认两机没有竞争任务后，使用原控制目录续跑；锁和身份校验会阻止重复启动或冻结输入漂移：

```bash
cd /home/qizhou.zhang@amd.com/mori_epv2_recheck_20260930
# 只查看计划，不启动 GPU：
taskset -c 4-7 python3 -B continue_after_baseline.py --resume --stop-after audit_after_scans --dry-run
# 真正续跑；仅在决定继续时执行：
taskset -c 4-7 python3 -B continue_after_baseline.py --resume --stop-after audit_after_scans
```

该命令跳过已有67个有效阶段，从中断的 T256 Combine 继续；不要删除 state/lock/evidence，也不要启动第二个队列。若需停止，在前台 Ctrl-C，让 controller/runner 清理带本次 run ID 的进程。不要用宽泛的 `pkill python` 或删除容器代替清理。

### 新机器：容器创建模板

以下还原自两机实际 inspect。基镜像为 `rocm/mori:ci`，已核实的 image ID 为 `sha256:dbc63edd7fd1dfa879aec8b460b3430fe22fb9e578614ea24800bf7936c0d055`。现有容器使用 root、host network/IPC、`sleep infinity`，未启用 privileged。新机先准备同一镜像和共享归档；不要依赖可变的 `:ci` tag 恰好仍指向该版本。镜像不在新机时，可先在旧机 `docker save` 此 image ID，再在新机 `docker load`。

```bash
mori_image=sha256:dbc63edd7fd1dfa879aec8b460b3430fe22fb9e578614ea24800bf7936c0d055
mori_workspace=/var/zqz/mori
mori_archive=/home/qizhou.zhang@amd.com/mori_epv2_recheck_20260917

docker image inspect "$mori_image" --format '{{.Id}}'
docker run -d --name mori_epv2_recheck_0917 \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add video --security-opt label=disable \
  --ulimit nproc=100000:100000 \
  --mount "type=bind,src=$mori_workspace,dst=/workspace/mori,readonly" \
  --mount "type=bind,src=$mori_archive,dst=/experiment" \
  --mount type=bind,src=/usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0,dst=/lib/x86_64-linux-gnu/libibverbs.so.1 \
  --mount type=bind,src=/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187,dst=/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187 \
  --mount type=bind,src=/usr/lib/x86_64-linux-gnu/libionic.so,dst=/usr/lib/x86_64-linux-gnu/libionic.so \
  --mount type=bind,src=/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so,dst=/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so \
  --mount type=bind,src=/etc/libibverbs.d,dst=/etc/libibverbs.d,readonly \
  -e MORI_RDMA_DEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
  -e MORI_RDMA_SL=3 -e MORI_RDMA_TC=104 \
  -e MORI_SOCKET_IFNAME=eno1 -e GLOO_SOCKET_IFNAME=eno1 \
  -w /experiment "$mori_image" sleep infinity
```

Ionic/ibverbs 库路径和版本必须按新机实际安装情况核实；设备、驱动或 NIC 类型不同，不能直接套用这组挂载。`/workspace/mori` 只是只读工作区；本次实际源码位于 `/experiment/rescan_20260930/source`。

### 新容器：恢复运行包和实际 benchmark 环境

**`/opt/mori-20260930` 在容器可写层，不是宿主机 bind mount。只用上述镜像创建容器不会恢复这份包。** 冻结 wheel 在共享归档 `rescan_20260930/wheels/amd_mori-0.1.0-cp312-cp312-linux_x86_64.whl`，SHA256 为 `cedc72ea35f1789ae358a67b219ffda76b7d88a6af138ba0d3416c5aa7e006ad`。仅对 package 目录不存在的新容器执行：

```bash
docker exec -i mori_epv2_recheck_0917 python3 - <<'PY_RUNTIME'
from pathlib import Path
import hashlib, zipfile
wheel = Path('/experiment/rescan_20260930/wheels/amd_mori-0.1.0-cp312-cp312-linux_x86_64.whl')
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == 'cedc72ea35f1789ae358a67b219ffda76b7d88a6af138ba0d3416c5aa7e006ad'
pkg = Path('/opt/mori-20260930/pkg')
assert not pkg.exists(), '已有 package，先核对身份，不覆盖'
pkg.mkdir(parents=True)
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(pkg)
Path('/opt/mori-20260930/jit-cache').mkdir(exist_ok=True)
print('Package extracted; validate source/runtime/ABI before GPU tests.')
PY_RUNTIME
```

旧 runtime 已审计为 Python3.12.3、Torch `2.12.0+rocm7.14.0`、HIP `7.14.60850`、CXX11 ABI=true。新机重新核对源码/包/表/入口及 ABI；空 JIT cache 需要重新编译、预热和验证。不要原样重跑旧 `deploy_pkg.py` 覆盖共享的 `deployment_node0.json` / `deployment_node1.json`，新机校验记录写入新实验目录。

本轮 `run_trial.py` 每次通过 `docker exec` 注入下列环境；**容器默认网卡是 eno1，正式测试覆盖为 eno0**。手工排查时可在容器 shell 中设置，正式扫描仍交给 runner 管理：

```bash
export PATH=/opt/venv/bin:$PATH
export PYTHONPATH=/opt/mori-20260930/pkg
export MORI_SOURCE_ROOT=/experiment/rescan_20260930/source
export MORI_JIT_CACHE_DIR=/opt/mori-20260930/jit-cache
export MORI_RESCAN_AFFINITY_DIR=/experiment/rescan_20260930/affinity
export MORI_GPU_ARCHS=gfx950 MORI_JIT_ARCH=gfx950 MORI_DEVICE_NIC=ionic
export MORI_SKIP_PRECOMPILE=1 MORI_DISABLE_IONIC_CCQE=1
export MORI_SHMEM_HEAP_SIZE=16G OMP_NUM_THREADS=1 GPU_PER_NODE=8
export MORI_SOCKET_IFNAME=eno0 GLOO_SOCKET_IFNAME=eno0
export MORI_RDMA_DEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export MORI_RDMA_SL=3 MORI_RDMA_TC=104
export MORI_APP_LOG_LEVEL=warn PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MORI_JIT_EXTRA_FLAGS=-ffile-prefix-map=/experiment/rescan_20260930/source=/mori-rescan
unset MORI_EP_TUNING_CONFIG MORI_EP_DISP_GEOM MORI_EP_COMB_GEOM MORI_EP_ROUND_SERIES
```

`MORI_EP_V2_TUNING_DIR` 由每次 trial 指向对应的两阶段表；不能统一指向空目录。旧双机每 GPU 的 CPU mask 是 `0–3、16–19、32–35、48–51、64–67、80–83、96–99、112–115`，新机器须按 PCI/NUMA 重新生成 affinity。

### 换登录机与换测试节点的区别

只换登录/开发机器、仍在 g17/g09 测试：保留共享目录，登录 g17 后按上面的原队列命令续跑即可。

替换 g17/g09 任一 GPU 节点：在**新的实验/证据目录**建立新 campaign，调整 HOSTS、SSH、master IP（旧值10.19.0.117）、socket/RDMA接口、设备与驱动挂载、NUMA/affinity，并重新跑正确性和同机 baseline。旧 observer/隔离记录绑定旧主机，不能作为新机核验结果；不能修改旧 manifest 后直接 `--resume`，也不能把新机器测量混入旧的 A0/B/A1 配对。保留本节67个有效阶段供参考，待新机独立确认后再决定采用。

<!-- END EPV2 HANDOFF 20261008 -->

## 测试方法更新（2026-09-30）

本节记录当前工作区对 V2 调优结果入表方法的修改，适用于 [test_dispatch_combine_v2_internode.py](tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py) 的 `--cmd tuning --tuning-save` 路径。最终指标由 `_row_metrics()` 采样、[_grand_mean_metrics.py](tests/python/ops/dispatch_combine_v2/_grand_mean_metrics.py) 汇总，与 V1 的 grand mean 口径对齐；**搜索选优与最终入表测量仍是两个独立步骤**。下文9月的既有实验保留当时冻结版本、配置和统计口径，不按本节的新定义改写旧数值。

### 计时与采样

1. 选定配置后，用实际将写入表的 Dispatch/Combine 配对建立一个 op，重新测量最终指标，不复用候选对照时的搜索分数。
2. 每个 `--tuning-reps` pass 独立执行 warmup，随后 `torch.cuda.synchronize()`、`dist.barrier()`，再记录首个 GPU event。屏障位于计时循环之前，循环内不加逐轮屏障。
3. 每轮分别记录 Dispatch、显式 dtype 转换、Combine 的 event 边界。两阶段延迟及其和不包含中间的转换区间，但它们是 GPU event 区间，不等于纯 kernel 指令时间，也不等于完整端到端 wall time。
4. 每个 pass 都丢弃前 `--drop-rounds` 轮，再汇总保留样本。当前默认参数为 `--warmup 20 --rounds 30 --drop-rounds 1 --tuning-reps 3`，即每个 rank、每阶段保留87个样本。
5. 所有 pass 完成后，进行一次 CPU `all_gather_rows`，收集各 rank 的 payload 字节数依据和原始延迟。读取设备上的接收 token 数、跨 rank 汇总均不放入计时循环。

### Grand Mean 与字节口径

旧 `_row_metrics()` 先计算每个 pass 的 rank/round 平均延迟，再对 pass 取 median；带宽是 rank 0 的字节数除以该延迟，`bandwidth_gbps` 使用接收 payload 口径。现在先对**每个 rank、每个保留轮次**计算带宽，再跨所有 pass、rank、round 求算术平均，不再使用“字节数除以平均延迟”代替平均带宽。

设 `r` 为 rank、`p` 为 pass、`i` 为保留轮次，`t[r,p,i]` 为该阶段延迟，单位为微秒。`recv[r]` 为本 rank 接收 token 数，`rdma[r]` 为 `_rdma_algo_token_count()` 按所测 family 计算的算法 token 数，`width[r]` 为该 rank、该阶段实际存储的每 token payload 字节数：

```text
avg_latency_us         = mean(t[r,p,i])
avg_rdma_bandwidth_gbps = mean(rdma[r] * width[r] / (1000 * t[r,p,i]))
avg_xgmi_bandwidth_gbps = mean(recv[r] * width[r] / (1000 * t[r,p,i]))
ll_scale_rank0          = max_tokens * topk / (recv[0] + 1)
avg_ll_bandwidth_gbps   = avg_xgmi_bandwidth_gbps * ll_scale_rank0
```

这里的 `mean` 覆盖全部保留样本。LL 与 V1 保存的 Average 行一致：**先求全局 XGMI 带宽均值，再乘 rank 0 的缩放比**，不是各 rank 单独缩放后再平均，也不是另测一条 LL 链路。结果仅在写入指标时保留两位小数。

| 入表字段 | 新定义 |
|---|---|
| `bandwidth_metric` | `grand_mean` |
| `bandwidth_gbps`，普通 `v2` | `avg_rdma_bandwidth_gbps` |
| `bandwidth_gbps`，`v2_ll` | `avg_ll_bandwidth_gbps` |
| `avg_rdma_bandwidth_gbps` / `avg_xgmi_bandwidth_gbps` / `avg_ll_bandwidth_gbps` | 分别保留上式三个指标，不把 XGMI 接收带宽当作普通 V2 的主指标 |

Dispatch 的 `width[r]` 来自输入张量的 `shape[-1] * element_size()`；Combine 来自转换后张量的实际存储宽度。因此 packed FP4 Dispatch 按 packed bytes 计数，BF16 Combine 按解包后的字节数计数，不能两阶段都套用逻辑 hidden dimension 与同一个 dtype。以上是 hidden payload 带宽，不额外把 scale、weights 或控制信息加入字节分子。RDMA 使用 V1 兼容的算法计数，并非 NIC 实测流量：普通 V2 对每个 token 的目标 node 去重，计数可包含本节点；LL 则按 `tokens * node_count` 计数，不能将此列解释为严格的跨节点物理字节率。

`_row_metrics(..., evidence_callback=...)` 可以返回所有 rank 的原始样本、字节数依据和采样参数供离线复算；这些证据不加入精简调优规则。普通 `--tuning-save` **不会自动归档这份 evidence**，需要调用方另行保存。旧表只有汇总值时，不能据此恢复逐 rank、逐轮样本，也不能仅把 `bandwidth_metric` 改名就视为完成迁移。

### 选优与只刷新指标

搜索仍使用 `_timed_pass()`：每个 pass 先求 rank/round 平均延迟，候选与基准配对重复测试，再使用 median、配对差值及最差轮次约束判断。默认 `--tuning-metric total` 比较 Dispatch+Combine 的延迟和；`phase` 只用于单阶段诊断。**本次没有修改这条搜索评分路径**，也没有把最终 `_metric_timing_pass()` 新增的循环前屏障加到搜索路径中。不能把搜索分数直接称为新入表带宽，或把默认 total 分数当成某个阶段的延迟。

只更新已选配置的指标时，先清除 `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM`，在原双机启动命令中使用 `--cmd tuning --tuning-save --tuning-phase dispatch`（或 `combine`），并把 `--tuning-candidate B,R,W` 设为该形状当前解析出的对应阶段 geometry。脚本会排除与基准相同的候选，输出 `no candidate measured (refresh of the shipped geometry)`，保留 geometry 并重新测量；若指定 geometry 与基准不同，则是单候选调优，不是纯刷新。一次运行只保存指定阶段的规则，两个阶段都要更新时分别运行。

使用 `--tuning-config-dir` 时，脚本会同时从该目录读取基准并写入结果；独立复测目录应先放入待验证的两阶段表，不能把空目录当成只改变输出位置。复测必须记录实际 dtype、H、token/capacity、专家数、Top-k、kernel family、QP、CCQE、weights/scale 和采样参数；固定 `--kernel-type v2` 或 `v2_ll`，避免 `auto` 在小 token 档切换 family。双机运行前确认无竞争负载；已有性能结论不外推到不同负载或网络条件。

### MI355X 复测与专家组合

MI355X 默认 JSON 仍保留本次重测前的指标，尚未回填新数据。下面是旧表的历史示例：普通 V2 Dispatch 的 BF16/H7168/T128/K6/epr16 行，`bandwidth_gbps=84.57`、`avg_rdma_bandwidth_gbps=30.58`；前者符合旧的 rank 0 接收字节数除以延迟，而不是新定义的 RDMA 主指标。即便旧行已经标注 `grand_mean`，也不能据此认定与本节可比。

另外，当前 [ep_internode_kernel.hpp](src/ops/dispatch_combine_v2/ep_internode_kernel.hpp) 对普通 V2/V2LL Combine 增加了生产者同步与发布顺序修复，LL 还增加了等待最终 node count 的逻辑；这些修改也覆盖 gfx950。应使用新内核先跑正确性，再刷新现有配置的性能指标，并对代表形状做配对复核。统计口径变化本身不要求从零全量搜索；出现显著回退或候选优势变化时，再重扫受影响项。此前数值验收不代替新内核的 MI355X 硬件回归。

V2 当前要求 `topk`、`experts_per_rank` 显式精确匹配，不再把缺字段规则当 wildcard；下文保留 wildcard 的描述仅是历史记录。EP16 下，总专家数为 `16 * experts_per_rank`：256/8 对应 `--experts-per-rank 16 --topk 8`，384/6 对应 `--experts-per-rank 24 --topk 6`。现有 MI355X Top-k 6 表记录的是256/6，不能作为384/6的实测结果；未覆盖的组合使用默认 geometry，须独立补调优后才能落表。

**2026-10-08 MI355X 复测已完成12/12个数值门槛和36/36个新口径原配置 baseline；按用户要求于06:43 UTC停止。** 全量扫描保留22/36个完整形状、67/108个有效阶段，具体停止点、证据、容器及换机方法见本文顶部“运行进度与换机交接”。尚无独立确认的新最优配置，默认JSON未更新。FP8→BF16/K8/T2048的固定原配置刷新已完成，原异常数据保留；不能据此计算优化收益。完整原始baseline与历史数据见 [bw.md](bw.md#epv2-rescan-20260930)。

## 实现与复测快照（2026-09-22 整理）

当前普通 v2 的 Dispatch 在去重后并行分配目标 slot。普通 Combine 在 gfx950、声明 capacity≤256 且行/staging stride 均为4字节整数倍时，直接调用已有的 `WarpAccumLF<T,4,2>`；其余情况使用 `WarpAccum<T,4>`。当前实现没有独立的向量尾部 helper，也没有额外的类型、topk 或节点拓扑选择条件；其它架构不启用这条 LF 路径。具体代码见 [ep_internode_kernel.hpp](src/ops/dispatch_combine_v2/ep_internode_kernel.hpp)。

9月21日已对当前 kernel 完成 BF16→BF16、FP8→FP8 的[固定 record geometry 对照](bw.md#epv2-recorded-geometry-20260921)，并分别完整构建 stash HEAD 与恢复改动后的 Python/native wheel，完成[默认分支 A–B–A 对照](bw.md#epv2-default-branch-20260921)。后者包含54个 dtype/token case，均通过数值预检，测量中未观察到竞争负载。两项实验均强制普通 v2、使用 g17/g09、EP16、H7168、K8、QP2，关闭 CCQE；详细数据及边界以对应章节为准。FP8 Dispatch 存在部分回退，不能把所有改动概括为全面加速。

HIP op 默认读取已有 geometry 记录，普通 v2 与 v2_ll 分别查表；公开 API 的 family 默认仍是 `auto`。当前包内没有 FP8 Combine 记录，其默认 geometry 保持96/64/8；此前实验补调的 FP8 Combine 配置尚未进入默认表。

下文保留此前实验原始说明，其中“当前”“final”“最终”均指该段当时冻结的版本。9月20日的额外类型/拓扑限制、向量尾部 helper 和 gfx942 策略属于历史实现，不描述上面的当前源码。实验归档目录及其原始日志不包含在本仓库中，指向 `../mori_epv2_recheck_20260917/` 的链接用于本地工作区查阅。

## 历史实验记录

2026-09-21：LF选择和尾部函数原文合回 `ep_internode_kernel.hpp`，独立helper头文件已删除。下述验收记录对应9月20日冻结版本，本次文件整理未重跑GPU测试或性能扫描。

**2026-09-20 历史完整 kernel 对照已完成**：以此前扫描并复测确认的同一份最优 JSON 为两边配置，对比 HEAD `fac2e18e` 的串行 Dispatch slot / 普通 Combine 与当时冻结的并行 slot / 条件 LF4×2。BF16/K8、T16–4096 共九档，每档三组 A–B–A，最终81次均通过数值检查且无并发记录。Dispatch T16–2048、Combine T16–128 三组均改善；T4096 两阶段整体持平。完整带宽、延迟、配置和证据见 [bw.md](bw.md) 顶部。该表的B是当时冻结的小容量LF版本，不能按后续cap≤256策略重新解释。普通Combine的后续off→broad 84组与off→tail 25组性能实验也已汇总；当前final源码已选定cap≤256及局部向量尾部，**39项final strict已通过，两端编译/缓存报告已验证；完整ELF等价不作声明**。最终策略与证据入口见 [bw.md最终策略段](bw.md#epv2-final-policy)。以下保留此前各项实验和JSON调优记录，按各自基线和冻结版本解释。

上述完整对照及早期JSON复核的实测范围：pit2-p03-g17 + pit2-p03-g09，MI355X/gfx950、ionic、EP16、experts/rank16，普通V2、H7168、QP2，**`MORI_DISABLE_IONIC_CCQE=1`**。本轮复核commit `1b3e7dba`的JSON，覆盖BF16→BF16的K6/K8及FP8→BF16的K8；性能测试的实际token数等于声明容量。下方原编号的kernel实验来自上一轮BF16/K8、两阶段96/64/8基线，其数值与本轮配置对照分别记录。LL和其它平台的结果不用于这里的优化判定。

`B/R/W`分别表示总block数、RDMA block数、每block的warp数；`S`是每chunk的分片数，Dispatch和Combine各自独立。下文保留原编号1–9、C1–C14，便于查原始记录。

- **有效**：三组新的A–B–A确认完整Dispatch+Combine周期有收益。
- **无效**：本次候选在所测配置下回退或持平；不推广为整个方向永远无效。
- **未确认**：只有小幅初筛差异，或计时受并发影响；不作为收益。
- **静态排除 / 未重跑**：没有本次GPU性能结论。

## Dispatch

commit `1b3e7dba`已加入MI355X普通V2的Dispatch JSON，按实际token数自动选档，包含T1024/2048/4096的256/128/8。最新独立复测、配置选择及与固定96/64/8的对照见 [bw.md](bw.md#json-recheck)。

### 本轮JSON复核

| 尝试 | 结论 | 处理 |
|---|---|---|
| BF16/K6/T64：96/64/8 → 16/8/8 | **有效**：三次加长配对复测及三组普通表路径ABA均改善 | 追加`experts_per_rank=16`行；保留原wildcard |
| BF16/K8/T64：64/32/8 → 32/21/8 | **未采用**：只有门槛内的小幅pair收益，Dispatch本身回退；普通ABA未确认此项 | 保留原Dispatch行，不与Combine候选叠加推算 |
| 其余七档、三种dtype/K组合的单OP quick扫描 | **没有新的稳定赢家** | 保留commit几何；只代表所扫范围 |
| BF16/K8的非2次幂block、RDMA比例及W6/W12邻域补扫 | **初筛未胜出** | 不修改几何 |

### 上一轮kernel与参数实验

| 原编号 | 尝试 | 上一轮结论 | 当时处理 |
|---|---|---|---|
| 1 | T4096，将B/R/W改成256/128/8，保留S8 | **有效** | 三组独立确认；新commit已将该几何加入对应JSON档位 |
| 1 | T64，缩小为64/32/8或32/16/8 | **未确认收益** | 只有小幅单组差异；64/32/8的前置基线还与短暂CPU缓存检查重叠，未采用 |
| 1 | T4096，固定96/64，仅将W改成4或16 | **无效：回退** | 保留W8；单独增减warp数没有改善完整周期 |
| 2 | T4096、W8，将S8改成S16或S32 | **无效：回退** | 保留S8；增加分片还会增加bid遍历和无payload的warp工作 |
| 2 | T4096，将S/W联合改成4/16，几何96/64/16 | **有效** | 保留[实验patch](docs/patches/epv2_dispatch_s4.patch)；未替换全局S8，优先使用上面的无patch配置 |
| 2 | T4096、W8，只将S8改成S4 | **未确认收益** | 清洁复测只有小幅差异，不单独采用 |
| 2 | T4096，将S/W联合改成16/4，几何96/64/4 | **无效：回退** | 不采用；S×W相同不代表相同延迟 |
| 4 | 将已有并行slot分配改成串行，做反向对照 | **串行方案无收益，保留已有并行实现** | T64/512的Dispatch变慢，T4096完整周期约持平；本轮没有新增slot优化 |

几何由 [hip_tuning_configs.py](python/mori/ops/dispatch_combine_v2/hip_tuning_configs.py) 的 internode API 通过共享的 `mori.ops.tuning_config` 读取分OP的JSON；`MORI_EP_V2_TUNING_DIR`可指定表目录。每相匹配自身dtype，`num_tokens`为包含上界的ceiling，最大档向上延用；同一档位的`experts_per_rank`精确匹配行优先于wildcard。测试自动选档时必须清除 `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM`，任一几何override都会绕过两个OP的表选择。显式使用普通 `v2`；`auto`的小T可能运行LL。

分片与slot代码在 [ep_internode_kernel.hpp](src/ops/dispatch_combine_v2/ep_internode_kernel.hpp) 的 `DispatchInterNodeRecv`。S实验保持 `bid % S` 的token归属；串行slot对照保留了去重、有效性检查、maps与payload copy顺序。它是机制对照，不是历史版本的完整revert。

### 只做了机制复核的历史方向

下列历史实现或完整快照缺失，本机没有重跑，不能标为本次有效或无效。

| 原编号 | 方向 | 复核结论 |
|---|---|---|
| 3 | 缓存远端node完成量，跳过空chunk轮询 | 应关注T远小于容量的场景；不能用本rank的T推断远端工作量 |
| 5 | 读取一次payload tile，复用到多个目标 | 会改变跨peer写流和寄存器使用；V1的收益不能代替V2验证 |
| 6 | 条件省略send结束AMO | 必须证明每个远端node的容量内slot都实际发送；稀疏、尾chunk和空rank仍需完成通知 |
| 7 | 用LDS任务列表摊平warp间copy负载 | 同时改变轮询、任务分配和block同步，不能从旧结果单独否定其中一个机制 |
| 8 | 将全局barrier改成LDS加全局两级计数 | 减少原子次数不等于缩短关键路径，必须保留payload发布和epoch清零顺序 |
| 9 | 按目标node紧凑打包send source，合并put | 需要额外打包和buffer；仅凭put尺寸不能预测完整周期收益 |

## Combine

当前final源码在gfx950为 **K6/K8、实际Combine类型BF16/FP32/OCP FP8、恰好两节点且每节点2/4/8 GPU、声明capacity≤256** 选择LF4×2，并用局部helper补齐单向量尾部；保留4字节stride对齐和短行回退。gfx942仍使用原BF16/H7168/K8/EP16/cap64或128条件。**最终39项strict均通过；两端编译与既有cache检查完成，二进制差异范围见下方。**

此前加入的Combine JSON独立于Dispatch选几何；FP8→BF16和BF16→BF16共用BF16 Combine行，修改该行须验证两种输入。T4096的S16/W8仍是限定配置的历史实验patch，未叠加到JSON复测。

### 本轮JSON复核

| 尝试 | 结论 | 处理 |
|---|---|---|
| K8/T64：96/64/8 → 16/10/8 | **有效，小幅收益**：BF16和FP8输入各三组普通表路径ABA均改善 | 两种输入共用一条BF16/epr16 Combine规则，保留原wildcard |
| K8/T256：64/32/8 → 64/42/8 | **有效**：BF16和FP8输入各三组普通表路径ABA均改善 | 追加BF16/epr16 Combine规则；FP8均值受尖峰影响，完整样本见bw.md |
| FP8/K8/T256：64/32/8 → 256/128/8 | **无效：初筛WIN未复现** | 加长复测回退；初筛差异主要来自原配置Dispatch的异常偏慢，不采用 |
| 其余Combine quick扫描及W6/W12等邻域补扫 | **初筛没有新的稳定赢家** | 保留commit几何 |

### 上一轮kernel与参数实验

| 原编号 | 尝试 | 上一轮结论 | 当时处理 |
|---|---|---|---|
| C1 | 容量64/128，本地和远端普通gather都先读两个相邻4B向量步，再累加 | **有效，已落地** | 两容量各三组独立确认；通过数值、最终包和目标机器码验收 |
| C2 | T4096，将普通gather扩宽到16B，必要时回退8/4B | **无效：回退** | 不采用 |
| C3 | T4096，将普通gather扩宽到8B，必要时回退4B | **未确认收益** | Combine本身变慢，完整周期小幅变化、wall持平；不采用 |
| C4 | T4096，将LF4×2推广到大容量，另新增LF4×4加深实验 | **无效：两者均回退** | LF保留小容量限制 |
| C6 | T4096、W8，将远端chunk分片S8改成S4 | **无效：回退** | 不采用 |
| C7 | T4096、W8，将S8改成S16，两阶段仍96/64/8 | **有效** | 保留[实验patch](docs/patches/epv2_gfx950_combine_s16.patch)，未默认启用 |
| C8 | 增加QP至4/8/16/32，同时影响两个OP | **无效：QP4持平，其余回退** | 继续QP2；旧bnxt平台的QP16结论不适用于本机 |
| C9 | T64，将Combine几何改成64/32/8 | **未确认收益** | 只有小幅单组差异，未采用 |
| C9 | T4096，将Combine几何改成96/48/8或128/64/8 | **无效：前者约持平，后者回退** | 保持96/64/8 |
| C12 | T4096，只在本地或只在远端gather启用LF4×2 | **无效：两者均回退** | 不采用单侧LF |
| C13 | T4096，以S16/W4或S4/W16替换S8/W8 | **无效：两种联合方案均回退** | 不根据S×W相等选配置 |
| C7/C13 | 固定W4，对比S8与S16 | **无效：约持平** | S16的已确认收益只覆盖W8 |
| C7/C13 | 固定W16，对比S8与S16 | **未确认：计时受CI编译影响** | 数值通过，性能样本不用于判断 |

LF调用位于普通 `CombineIntraNodeTyped` / `CombineInterNodeTyped` → `CombineGatherNormal`，具体策略和尾部处理位于 [ep_internode_kernel.hpp](src/ops/dispatch_combine_v2/ep_internode_kernel.hpp)。gfx950取消H7168特例，声明capacity上限扩为256，支持上述三种实际Combine类型和三种两节点拓扑；NIC/QP不参与选择。其它架构或不满足topk、类型、拓扑、capacity、对齐条件时使用原普通gather。FNUZ Combine没有新启用；FNUZ/FP4输入搭配BF16/FP32 Combine按后者判断。声明capacity4096而实际T64仍回退；性能结论限定于当时冻结的Ionic实测配置。

设单步为 `64×4/sizeof(Combine dtype)` 个元素：总行长不足两个步（512字节）时回到原 `WarpAccum`；完整LF双步后若仍有一个完整向量步，就用原4B向量实现处理，再执行不足单步的标量尾部；余量不足单步时保留原LF与标量实现。无需另设hidden下限。helper保持原源指针累加顺序及最后一次类型转换，没有修改公共primitive、LL、权重和最终跨节点归约。

性能扩展已完成：[off→broad统计](../mori_epv2_recheck_20260917/lf_generalize_20260920/results.md)包含84组ABA/252次运行，[off→tail统计](../mori_epv2_recheck_20260917/lf_generalize_20260920/tail_results.md)包含25组/75次运行。核心12格使用已扫描并复测确认的selected baseline，每格3组且A/B几何相同；BF16/K6的T64/128 Combine延迟分别下降15.54%/12.40%。默认/回退几何及tail两阶段96/64/8属于diagnostic，没有为每个扩展shape重新扫最优配置。两轮LF专项的Dispatch均保持并行，仅作控制项。

局部向量尾部使先前BF16 H1152/H2176、OCP FP8 H768及2×2 BF16 H2176四个回退点在独立off→tail复测中转为改善；对应Combine延迟变化为−2.70%/−6.02%/−10.36%/−1.17%，均3组。T256的H7168/BF16 K6/K8各3组off→broad分别下降2.00%/2.22%，但tail T256交叉诊断中的短BF16及2×2 BF16接近持平，不能声称所有cap256形状均稳定受益。具体数值和限制见bw.md；broad/tail仍是实验版本，最终条件以final源码为准。

FP4性能输入为真实E2M1双nibble packed bytes，每token dispatch payload是逻辑H/2字节；公开codec解码后交给BF16或FP32 Combine，没有启用FP4 Combine。本测试不自动应用opaque scale metadata，不等同于完整MXFP4量化；pack/unpack不计入两阶段kernel时间，payload与scale/weights分项记录。

**最终正确性：39/39项通过。** 38项主队列加BF16 H7200原LF非零标量尾部补充均完成，每个有效rank各3轮；共1,704条STRICT记录，逐位检查2,469,003,264个hidden元素，hidden不匹配为0。weights采用atol=rtol=1e-6，实测最大绝对误差为0。6个FP4用例有288条PACKED记录、207,954个接收行、745,307,136个packed字节，完整接收行multiset检查无不匹配。该检查不要求接收顺序一致，数值范围为有限值identity-expert、quant none的普通v2。见[主审计](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_strict_v2_audit.json>)、[H7200补充审计](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_strict_v2_supplement_audit.json>)和[独立复核](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_strict_independent_review.md>)。

**最终编译及缓存检查：两端各105份普通Combine报告均验证成功。** 每端含30份离线编译和75份既有缓存inspect，后者全部命中，未通过补编译填MISS。每端35项final/reference比较中，34项原始.text字节相同，35项descriptor、只读数据和kernel metadata/资源相同；启用路径分别对照broad或tail，回退路径对照off。**完整代码对象等价审计保留EQUIVALENCE_NOT_PROVEN：** 所有对照都有真实导出符号`__hip_cuid_*`及相关ELF查找表差异，个别标识长度还改变DT_STRSZ；H192回退另有4字节指令差异，不能把off的H192性能直接转作final结论。没有抹掉这些差异或将完整ELF判为相同。原始审计见[node0](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_offline_audit_node0_v2.json>)、[node1](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_offline_audit_node1_v3.json>)，逐字节差异分类见[node0诊断](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_elf_cuid_host_diagnostic_v1.json>)、[node1诊断](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_elf_cuid_host_node1_v3.json>)。本次offline只覆盖ordinary Combine，Dispatch/LL仅有源码保持审查和各自历史验证；gfx942/mlx5仅作离线编译与代码对象覆盖，不构成这些平台的硬件正确性或性能结论。

最终状态和全部审计SHA256集中在[final_verification_summary.json](<../mori_epv2_recheck_20260917/lf_generalize_20260920/final_verification_summary.json>)（SHA256 `563bddc4824d3504d6b49239f129d7a7310f013462b2da30f24895b4077a84c0`）。这份结果更新冻结记录及下方实验发布快照中的pending状态，原始证据未改写。g09首次辅助程序构建因本机缺少host headers退出，已保留失败日志并用g17导出的相同头文件在独立v3目录完成；没有修改GPU测试来源或既有cache。结束时两台机器均无本任务活动runner。当前更新为源码及报告，测试显式使用冻结MORI_SOURCE_ROOT，未重新构建wheel或生成包内源码副本。

此前移除NIC/QP限制时的历史验证（对应当时源码，不代替本次final验收）：gfx950/Ionic与mlx5的QP4、容量64普通Combine均通过[离线编译](../mori_epv2_recheck_20260917/lf_portability_20260920/offline/summary.json)。g17/g09上的Ionic数值检查覆盖QP2/T64 BF16、QP4/T64 BF16、QP4/T128 BF16 skewed、QP8/T64 FP8→BF16，每组3轮、16个rank的输出和权重均通过，见[数值结果](../mori_epv2_recheck_20260917/lf_portability_20260920/gpu_checks.json)。额外的QP8/T128 local用例因CI持续占用未启动；mlx5仅有编译验证，未作对应硬件运行或性能声明。

S16改的是普通远端Combine的分片，token归属 `bid % S` 和完成阈值 `S * W`保持一致。实验patch自身没有完整限制NIC/QP/W，只能在上述实测配置下使用；不能全局开启或把它与Dispatch的收益相加。两种实验patch均未进入上一轮已验收wheel，应在独立checkout分别应用；基线header SHA256为 `5b1a92a5ed89b683539ac07590320fca63d568acbf90405ecf0fafdb0ff448d8`。

### 静态排除与待验证项

| 原编号 | 方向 | 本次状态 |
|---|---|---|
| C5 | 压缩非空源指针，再选择累加模板 | **本机未重跑**：历史快照缺失；原路径已跳过空源payload读取，压缩不会减少有效payload字节 |
| C10 | 让同一向量步的expert读取并行 | **静态排除**：常用K8实现已先读取各源再累加；C1改变的是跨两个向量步的调度 |
| C11 | 只改 `WARP_ACCUM_UNROLL` 让普通gather展开 | **静态排除**：普通wrapper没有使用该参数展开，修改宏不能实现目标优化 |
| C14 | 定位gather、传输、peer等待与barrier成本 | **已做阶段profile，优化待验证**：主Combine仍包含这些成本，尚未完成内部等待归因，没有新的代码收益 |

## 验证与记录

上一轮小容量LF、推荐Dispatch几何及V1/V2/LL兼容性已通过对应数值验收。CQ模式一致性、provider生命周期和V1 benchmark入口属于正确性修复，不计入kernel优化收益。此前JSON复核仅更新表和对应配置测试；本次LF扩展另增加普通Combine的局部helper及上述选择条件，保留已有改动。LL、`CombineAll`和共享gather primitive没有新增优化改动；最终源码的运行、编译结果及二进制核对范围见上方。

新加的三条JSON规则只覆盖epr16，其他expert数继续命中原wildcard。token档位按实际输入做ceiling匹配，容量、QP和CQ模式不是匹配键；性能结论限定于实际T=容量、QP2、CCQE关闭的实测条件。独立扫描是固定另一OP的有限候选集，不是全局最优证明。

此前采用的JSON已通过BF16/K6、BF16/K8、FP8/K8各21轮不同token档位的数值检查（每轮16个rank，含uniform/skewed/local），以及每个dtype/K组合在容量64/256/4096各128轮持续测试。数值检查含weights；持续测试检查hang/fault，未逐轮核对数值。

性能数值、复现入口和原始证据索引统一见 [bw.md](bw.md)。
