# SGLang KV Indexer 使用与部署指南

本指南面向**运行、配置、部署** KV Indexer 的同学。设计动机见
[`SGLANG-KV-INDEXER-RFC.md`](./SGLANG-KV-INDEXER-RFC.md),分阶段实现计划见
[`SGLANG-KV-INDEXER-DEVELOPMENT-PLAN.md`](./SGLANG-KV-INDEXER-DEVELOPMENT-PLAN.md)。

---

## 1. 概览

KV Indexer 是一个**全局外部 KV 元数据索引服务**。它把各 SGLang worker 本地
多级 KV cache(GPU / CPU_PINNED / DISK)里"哪些 block 存在于哪个 tier"的信息,
汇聚到一个共享的、内容寻址(block hash)的索引中,供调度/路由侧查询命中率、
做前缀感知路由等。

它本身**不搬运 KV 数据**,只维护元数据(block hash → tier / worker / 掩码 等)。

### 数据链路

```
                 ZMQ PUB (事件流) + ROUTER (重放)
SGLang worker  ───────────────────────────────►  bridge  ──gRPC──►  KV Indexer  ──►  Redis
(radix cache)   BlockStored/Removed/Cleared      (每 worker 一个)              (单机/Dragonfly/Cluster)
```

1. **SGLang**:每个 worker 的 radix cache 在 block 落盘/淘汰时,通过 ZMQ 发布
   KV cache 事件(`BlockStored` / `BlockRemoved` / `AllBlocksCleared`),每条事件带
   一个单调递增的 `seq`。
2. **bridge**(sidecar):订阅某个 worker 的事件流,检测 `seq` 缺口,必要时通过
   ROUTER/DEALER 重放拉回,再翻译成 gRPC 调用转发给 indexer。
3. **KV Indexer**:tonic gRPC 服务,把事件应用到后端存储。
4. **Redis**:后端存储,支持单机 / Dragonfly / Cluster。

### tier 命名对应

| SGLang 语义 | proto TierType | 常见别名 |
|---|---|---|
| GPU (L1)         | `TIER_HBM` | `HBM` |
| CPU_PINNED (L2)  | `TIER_DRAM` | `DRAM` / `CPU` |
| DISK (L3)        | `TIER_SSD` | `SSD` |

### gRPC 接口(3 个 RPC)

| RPC | 用途 |
|---|---|
| `ApplyExternalKvBatch`            | **唯一写入口**:按序批量应用 store/remove/clear 动作 |
| `MatchExternalKv`                 | 查询一段 block hash 的命中情况(不计数) |
| `GetExternalKvHitCounts`          | 查询命中计数 |

资源边界:单个请求/Apply batch 最多携带 16,384 个 hash,单个 Apply batch
最多 256 个 action;超限返回 gRPC `RESOURCE_EXHAUSTED`。Redis 后端以每批 256
个操作分块执行,避免单请求产生无界并发。

另外注册了标准 `grpc.health.v1.Health` 服务(就绪探针):后台每 5s 探一次后端,Redis 可达时报 `SERVING`,否则 `NOT_SERVING`;整体("")与具名服务 `kv_indexer.v1.KVIndexer` 均可查询。k8s/router 用它判断 indexer 是否可服务(尤其降级启动、Redis 未连上时)。

---

## 2. 构建

crate 位于 `sglang-kv-indexer/`,产出两个二进制:`kv-indexer-server` 与
`kv-indexer-bridge`。

```bash
cd sglang-kv-indexer

# 生产 / 启用 Redis 后端(必须带这个 feature)
cargo build --release --features redis-backend
# -> target/release/{kv-indexer-server,kv-indexer-bridge}
```

要点:

- `redis-backend` 是**可选 feature,默认关闭**(标准构建不拉 redis 依赖)。使用
  Redis 后端必须带 `--features redis-backend`,否则 `KV_INDEXER_BACKEND=redis`
  会直接报错 `requires building with --features redis-backend`。
- `cargo` 是 Rust 的构建/包管理器(底层调用 `rustc`);日常只用 `cargo`。
- 联调用 debug,压测/上线用 `--release`。

---

## 3. 配置(全部走环境变量)

服务遵循 12-factor,**只用环境变量配置**,不用命令行 flag —— 便于容器/K8s 注入
与密钥管理。

### 3.1 indexer 服务端(`kv-indexer-server`)

| 变量 | 默认 | 说明 |
|---|---|---|
| `KV_INDEXER_LISTEN_ADDR` | `[::1]:50051` | gRPC 监听地址。容器里通常设 `0.0.0.0:50051` |
| `KV_INDEXER_BACKEND`     | (必填) | 必须显式设为 `redis` 或 `logging`;生产使用 `redis`,避免配置遗漏后静默使用调试后端 |
| `RUST_LOG`               | `info` | 日志级别;`debug` 可看每 RPC 明细 |

Redis 后端(`KV_INDEXER_BACKEND=redis`)追加:

| 变量 | 默认 | 说明 |
|---|---|---|
| `KV_INDEXER_REDIS_URL`           | (无) | 单实例/Dragonfly 地址,如 `redis://127.0.0.1:6379`。非 cluster 时必填 |
| `KV_INDEXER_REDIS_CLUSTER_NODES` | (无) | 逗号分隔的 cluster 节点;设了则走 Cluster,优先于 `URL` |
| `KV_INDEXER_REDIS_NAMESPACE`     | `kvidx` | key 前缀,多租户/多环境隔离 |
| `KV_INDEXER_REDIS_REQUIRED`      | `1` | `1`=启动即连接并 PING,连不上快速失败;`0`=降级启动,首次使用时惰性连接(见 §7) |
| `KV_INDEXER_WORKER_TTL_SECS`     | `120` | worker 存活 TTL:超过该时长未 apply/心跳的 worker 会从 `match` 结果中剔除(避免路由到死节点);`0`=关闭存活判定 |

> 升级注意: generation fencing 将 placement/reverse 值升级为代际格式。
> 新版本可读取旧格式,但旧 binary 不能读取新格式;不要让新旧 indexer 对同一
> namespace 做滚动混写。首次部署应停写升级,或切换新的
> `KV_INDEXER_REDIS_NAMESPACE`。

### 3.2 bridge(`kv-indexer-bridge`)

| 变量 | 默认 | 说明 |
|---|---|---|
| `KV_INDEXER_WORKER_ID`             | (必填) | 该 worker 的唯一标识,如 `worker-0` |
| `SGLANG_KV_EVENT_ENDPOINT`         | (必填) | 订阅的 SGLang PUB 地址,如 `tcp://127.0.0.1:5567` |
| `KV_INDEXER_ENDPOINT`              | (必填) | indexer gRPC 地址,如 `http://127.0.0.1:50051` |
| `SGLANG_KV_EVENT_REPLAY_ENDPOINT`  | (无) | SGLang 重放 ROUTER 地址;缺口重放和可信 worker 心跳都依赖它,生产环境应配置 |
| `SGLANG_KV_EVENT_TOPIC`            | (空) | ZMQ 订阅 topic,与 SGLang upstream 默认一致;非空时须在两侧设置相同值 |
| `KV_INDEXER_WORKER_ADDRESS`        | (空) | 可选,worker 反查地址 |
| `KV_INDEXER_CLEAR_TIERS`           | `HBM,DRAM,SSD` | `AllBlocksCleared` 时清空哪些 tier |
| `KV_INDEXER_HEARTBEAT_SECS`        | `30` | 先严格校验 SGLang replay ROUTER 的 END 响应,成功后才续 worker TTL;探测失败仅抑制本次心跳,不重建 SUB;无 replay endpoint 时周期心跳禁用;`0`=关闭 |
| `KV_INDEXER_WORKER_INCARNATION`    | (自动) | 可选的世代号可观测前缀;bridge 会追加进程内唯一后缀,不会在重启后复用已退休 token。服务端退休旧 token 并拒绝迟到请求;确认 seq 回退后自动生成新世代 |

---

## 4. 快速上手(单机联调)

仓库自带一键联调脚本 `sglang-kv-indexer/lab.sh`,可拉起
`indexer → sglang → bridge` 全链路。

```bash
cd sglang-kv-indexer
cargo build --bins --features redis-backend      # 构建(带 Redis 后端)
redis-server --port 6379 &                        # 起一个本地 Redis

# lab.sh 会把环境变量透传给 indexer,设好 backend/Redis 即可
KV_INDEXER_BACKEND=redis KV_INDEXER_REDIS_URL=redis://127.0.0.1:6379 \
  ./lab.sh up                 # 起 indexer -> sglang(等就绪) -> bridge
./lab.sh logs                 # 另开窗口看三路彩色日志([IDX]/[BRG]/[SGL])
./lab.sh test 12              # 发 12 条 generate 请求造 KV 事件
./lab.sh status               # 看进程与端口
./lab.sh down                 # 收工
```

常用覆盖项(环境变量):

```bash
GPU=0 TP=1 MODEL=/nfs/data/Qwen3-0.6B RUST_LOG=info \
KV_INDEXER_BACKEND=redis KV_INDEXER_REDIS_URL=redis://127.0.0.1:6379 ./lab.sh up
```

验证成功的标志:

- bridge 日志出现 `bridge session established`,全链路无 WARN/ERROR/gap;
- 发流量后 Redis 里出现索引 key:`redis-cli KEYS 'kvidx*' | head`(namespace 默认 `kvidx`),
  `redis-cli DBSIZE` 随流量增长。

---

## 5. 部署拓扑

### 5.1 单 worker(最简)

```
sglang(1 worker) ──PUB:5567/replay:5590──► bridge(worker-0) ──► indexer:50051 ──► redis
```

- sglang:`--kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5567","replay_endpoint":"tcp://*:5590","buffer_steps":10000}'`
- bridge:`SGLANG_KV_EVENT_ENDPOINT=tcp://<sglang>:5567`、`KV_INDEXER_ENDPOINT=http://<indexer>:50051`
- indexer:`KV_INDEXER_BACKEND=redis`、`KV_INDEXER_REDIS_URL=redis://<redis>:6379`

### 5.2 多 DP rank / 多 worker(重点)

**关键机制**:SGLang 对每个独立 KV cache 的事件端口按 `base_port + rank` 偏移
(源码 `srt/disaggregation/kv_events.py::offset_endpoint_port`)。rank 的取值:

- **DP attention**(`--enable-dp-attention`):按 `attn_dp_rank`
- **纯 DP 副本**(`--dp-size N`):按 `dp_rank`(副本序号)

例如配 `endpoint=tcp://*:5567`、`replay_endpoint=tcp://*:5590`、`--dp-size 4`:

| DP rank | PUB 端口 | replay 端口 |
|---|---|---|
| 0 | 5567 | 5590 |
| 1 | 5568 | 5591 |
| 2 | 5569 | 5592 |
| 3 | 5570 | 5593 |

> ⚠️ PUB base 与 replay base 之间要留足间隔,否则 DP 下 `PUB_base+rank` 会撞上
> `replay_base`。本仓库 `lab.sh` 默认 PUB=5567 / replay=5590 即为此预留。

**部署原则:N 个 rank → N 个 bridge → 1 个共享 indexer**

- **一 rank 一 bridge**:每条事件流的 `seq` 缺口检测和重放是 per-rank 的,不能复用
  一个 bridge 处理多路(会串号)。每个 bridge 用唯一 `KV_INDEXER_WORKER_ID` 和
  自己 rank 的 `PUB_base+rank` / `replay_base+rank`。
- **共享一个 indexer / Redis**:block 内容寻址,不同 rank 写入同一 Redis 天然去重合并,
  正是外部索引想要的效果。

用 `lab.sh` 一键起(已内置 DP 支持):

```bash
# 2 个 DP 副本,各占 1 卡,自动起 bridge-0 / bridge-1,共享一个 indexer
GPU=0,1 TP=1 DP_SIZE=2 ./lab.sh up
./lab.sh test 12
```

手动起 N 个 bridge 的等价写法:

```bash
DP_SIZE=4; BASE_PUB=5567; BASE_REPLAY=5590; INDEXER=http://127.0.0.1:50051
for r in $(seq 0 $((DP_SIZE-1))); do
  KV_INDEXER_WORKER_ID="worker-$r" \
  SGLANG_KV_EVENT_ENDPOINT="tcp://127.0.0.1:$((BASE_PUB+r))" \
  SGLANG_KV_EVENT_REPLAY_ENDPOINT="tcp://127.0.0.1:$((BASE_REPLAY+r))" \
  KV_INDEXER_ENDPOINT="$INDEXER" \
    ./target/release/kv-indexer-bridge &
done
```

### 5.3 容器 / K8s

推荐拆分:

- **bridge 作为 sglang 的 sidecar**:与某个 rank 同 Pod,走 `localhost` 通信;rank
  扩缩容时 bridge 随 Pod 一起伸缩。
- **indexer 独立 Deployment + Service**:当前必须运行 **1 个写副本**。进程重启后可
  从 Redis 恢复,但 per-worker apply 串行锁是进程内锁;多个 indexer 副本并发写同一
  Redis 可能打乱 mutation 顺序,因此暂不支持多副本写 HA。
- **Redis 独立 StatefulSet**(或托管 Redis/Dragonfly/Cluster)。

```
┌ Pod: sglang-rank-r ┐        ┌ Deploy: kv-indexer ┐      ┌ StatefulSet: redis ┐
│ sglang  +  bridge  │──gRPC──►│  kv-indexer-server │──────►│      redis         │
└────────────────────┘        └────────────────────┘      └────────────────────┘
```

多机时:sglang 的 `endpoint` 用 `tcp://*:5567` 绑全网卡,bridge 从对端 IP
`connect`(端口同样 `base+rank`)。

内置 gRPC 健康状态反映 Redis readiness。它不应直接用作 liveness,
否则 Redis 故障会导致健康的 indexer 进程被反复重启:

```yaml
readinessProbe:
  grpc: { port: 50051 }          # 查整体 ""; Redis 不可达时 NOT_SERVING → 不接流量
livenessProbe:
  tcpSocket: { port: 50051 }     # 仅检查进程/监听端口,不绑定 Redis readiness
```

---

## 6. replay / 缺口恢复

- SGLang 每条事件带单调 `seq`;发布端保留最近 `buffer_steps` 条到内存重放缓冲。
- bridge 在同一进程生命周期内发现 `seq` 不连续(如 `...,1,4`)时,会通过 DEALER
  请求缺失区间,按序补齐后再 apply。历史已从 replay buffer 淘汰时仍会降级为
  best-effort。
- replay endpoint 还被用作 worker 存活证明:只有 ROUTER 响应后 bridge 才续 TTL,
  避免 SGLang 已死而 sidecar 仍把旧 placement 保活。
- bridge **进程重启**会丢失本地 `next_seq`;当前没有全量 cache dump,因此不能保证
  重建 SGLang 中已存在但不再产生事件的 block。这仍是显式非目标。

---

## 7. 可观测与排障

### 看什么日志

- **bridge**:启动打印 `worker_id / event_endpoint / replay_endpoint / indexer_endpoint`;
  会话建立打印 `bridge session established`。`RUST_LOG=debug` 可看每批转发明细。
- **indexer**:`RUST_LOG=info` 打印启动信息与每个 RPC;`debug` 更详细。
- **Redis**:`redis-cli KEYS 'kvidx*'`、`redis-cli DBSIZE` 直接看索引数据是否在增长
  (namespace 由 `KV_INDEXER_REDIS_NAMESPACE` 决定,默认 `kvidx`)。

### 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 启动 `requires building with --features redis-backend` | 用了 `KV_INDEXER_BACKEND=redis` 但二进制没带该 feature。重新 `cargo build --features redis-backend` |
| Redis 不可达时启动卡住 / 或期望降级却退出 | 用 `KV_INDEXER_REDIS_REQUIRED` 明确语义:`1`=快速失败,`0`=降级惰性连接 |
| 多 worker 只看到一个 worker 的事件 | bridge 端口没按 `base+rank` 偏移,或只起了一个 bridge;确认一 rank 一 bridge |
| DP 下端口冲突 | `PUB_base+rank` 撞到 `replay_base`;拉大两个 base 的间隔 |
| bridge 报 seq gap 且不恢复 | 没配 `SGLANG_KV_EVENT_REPLAY_ENDPOINT`,或 SGLang `buffer_steps` 太小已淘汰 |
| indexer 收不到任何事件 | topic 不匹配(`SGLANG_KV_EVENT_TOPIC`),或 sglang 未开 `--enable-hierarchical-cache` / 未配 `--kv-events-config` |
