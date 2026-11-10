# Mori UMBP 使用手册：本机 KV 缓存

UMBP（Unified Memory/Bandwidth Pool）通过统一的指针 API 存储和恢复 KV 数据。
调用者提供 key、地址和字节数，UMBP 管理对象元数据、存储空间、读写和回收。
在 SGLang 对接中，Linker 将框架的 KV 页面布局转换为 UMBP 对象及其字节范围。

本文于 2026-11-10 按 Mori 源码提交
[`a612473e56f2c509a06cccabef99aed0fff94a6b`](https://github.com/ROCm/mori/tree/a612473e56f2c509a06cccabef99aed0fff94a6b)
核对接口。历史 SGLang 实验的版本和启动方式见 [SGLang demo](sglang-demo.zh-CN.md)。

## 1. 选择本机运行方式

| 方式 | 存储由谁持有 | 适用情况 | 配置入口 |
|---|---|---|---|
| 进程内 embedded | 当前进程 | API 学习、单进程应用、小容量检查 | `UMBPClient(config)` |
| 独立 standalone 进程 | 一个服务进程，多 worker 共用 | TP 多 rank、集中管理本机缓存容量 | `config.standalone_process`，同一个 Unix socket |

当前默认 embedded 部署提供 DRAM 池。显式选择 SSD 可以提供本机 SSD 存储；
需要多层策略时使用 `backend_policy_path` 定义 backend 与 tier。
**单独设置 `ssd.enabled=True` 不会给默认 DRAM 池自动添加 SSD 层。**

当前 embedded 使用的是无 master 的 `DistributedClient` 实现，因而
`get_deployment_mode()` 返回 `Distributed`，`is_distributed()` 也返回 `True`；
standalone forwarding client 的 `is_distributed()` 返回 `False`。
这些值在当前实现中不能用于判断是否连接 master；应检查实际部署配置中的
`master_config.master_address` 与服务日志。

## 2. 安装与检查

使用 Python 3.10+ 和具有兼容 ROCm/HIP 开发环境的 Linux 主机。
从源码构建前完成 [Mori 安装要求](../../README.md#installation)，并安装 UMBP 的 C++ gRPC/Protobuf 依赖。
Debian/Ubuntu 示例：

```bash
sudo apt-get install libgrpc-dev libgrpc++-dev libprotobuf-dev \
  protobuf-compiler protobuf-compiler-grpc

# 在 Mori 仓库根目录、准备好的 Python 环境中执行。
python3 -m pip install -r requirements-build.txt
BUILD_UMBP=ON BUILD_UMBP_SPDK=OFF \
  python3 -m pip install . --no-build-isolation

python3 -c 'from mori.umbp import UMBPClient, UMBPConfig; print("UMBP import OK")'
```

`setup.py` 在缺少 gRPC/Protobuf 时可能禁用 UMBP，即使整个安装成功。
必须确认构建日志以及以上 import；普通 Mori 模块能加载不代表 UMBP 已编入。
CMake 单独构建时选项为 `-DBUILD_UMBP=ON`；禁用 SPDK 使用 `-DUSE_SPDK=OFF`。

MI455 的 gather kernel 需要匹配运行时报告的 GPU 架构。若设备报告 `gfx1250-strict`，
在构建时将该目标加入 `MORI_GPU_ARCHS`，见 Mori 根 README 的架构说明。
上面的安装命令用于当前源码环境，不用于替换历史回放镜像里的 Mori。

## 3. 最小 DRAM 用法

以下使用 4 MiB 池，写入 256 字节并逐字节校验：

```python
import ctypes
import uuid
from mori.umbp import UMBPClient, UMBPConfig

config = UMBPConfig()
config.dram.capacity_bytes = 4 * 1024 * 1024
config.dram.use_hugepages = False
config.ssd.enabled = False
client = UMBPClient(config)
try:
    key = f"demo/{uuid.uuid4().hex}"
    src = (ctypes.c_ubyte * 256)(*range(256))
    dst = (ctypes.c_ubyte * 256)()
    if not client.put_from_ptr(key, ctypes.addressof(src), len(src)):
        raise RuntimeError("PUT failed")
    if not client.get_into_ptr(key, ctypes.addressof(dst), len(dst)):
        raise RuntimeError("GET failed")
    if bytes(src) != bytes(dst):
        raise RuntimeError("Byte mismatch")
finally:
    del client
```

完整示例还覆盖批量和范围 I/O。在仓库根目录执行：

```bash
python3 examples/umbp/local_kv_demo.py
```

成功输出 `"ok": true`，校验 4 个对象，并检查单对象、batch、ranges 和连续命中前缀。
每次使用独立 key namespace；不会调用共享池上的全量 `clear()`。

## 4. 本机 SSD 用法

显式选择 SSD medium。这里的 `UMBPDistributedConfig` 也用于配置本机存储，
不填写 master，`io_engine.host` 留空，`peer_service_port=0`。

```python
from mori.umbp import UMBPConfig, UMBPDistributedConfig, UMBPIoBackend, UMBPMedium

config = UMBPConfig()
config.dram.capacity_bytes = 4 * 1024 * 1024
config.ssd.enabled = True
config.ssd.storage_dir = "/mnt/nvme/umbp-demo-new-run"  # 本次运行专用目录
config.ssd.capacity_bytes = 64 * 1024 * 1024
config.ssd.segment_size_bytes = 8 * 1024 * 1024
config.ssd.io.backend = UMBPIoBackend.Posix

storage = UMBPDistributedConfig()
storage.medium = UMBPMedium.SSD
storage.io_engine.host = ""
storage.peer_service_port = 0
storage.dram_page_size = 64 * 1024
storage.ranged_scratch_size = 0
storage.ssd_staging_buffer_slots = 8
config.distributed = storage
# 后续与 DRAM 使用同一套 UMBPClient 指针 API。
```

可执行示例自动创建并删除专用临时目录：

```bash
python3 examples/umbp/local_kv_demo.py --medium ssd
# 指定已有 NVMe 挂载目录作为临时目录的父目录：
python3 examples/umbp/local_kv_demo.py --medium ssd --ssd-parent /mnt/nvme
```

默认 `/tmp` 适合功能检查；SSD 性能测试应使用实际设备挂载点。
`direct_io=True` 时会探测 `O_DIRECT`，不支持的文件系统可能退回 buffered I/O 并记录警告。
容量是缓存预算，并非启动时写满整个磁盘。SSD 使用分段日志，空间按记录和 segment 回收；
其容量、padding、GC 和 staging 占用需一起观察。

UMBP SSD 用于可重建缓存，不把服务重启后的缓存留存作为应用持久化承诺。
当前实现存在启动时清理旧缓存 segment 的路径，测试和服务都应使用专用目录。

## 5. 多进程共用一个 standalone

先定位当前安装附带的服务程序：

```bash
export UMBP_STANDALONE_BIN=$(python3 -c \
  'import mori.umbp, pathlib; print(pathlib.Path(mori.umbp.__file__).resolve().parents[1] / "umbp_standalone_server")')
test -x "$UMBP_STANDALONE_BIN"
export UMBP_DEMO_DIR=$(mktemp -d /tmp/umbp-demo.XXXXXX)
export UMBP_DEMO_ADDRESS="unix://$UMBP_DEMO_DIR/server.sock"
```

在干净终端中启动本机 DRAM 服务。不要继承其他部署的 `UMBP_MASTER_ADDRESS`、
`UMBP_NODE_*`、`UMBP_IO_ENGINE_HOST`、`UMBP_BACKEND_POLICY` 或 SSD medium 设置。
服务程序使用第一个位置参数作为地址，支持 `UMBP_STANDALONE_ADDRESS` 环境变量：

```bash
UMBP_DRAM_CAPACITY=67108864 UMBP_DRAM_USE_HUGEPAGES=0 \
UMBP_SSD_ENABLED=0 UMBP_STANDALONE_METRICS_PORT=0 \
UMBP_HBM_GATHER_PREWARM_STREAMS=0 MORI_UMBP_LOG_LEVEL=INFO \
  "$UMBP_STANDALONE_BIN" "$UMBP_DEMO_ADDRESS" \
  > "$UMBP_DEMO_DIR/standalone.log" 2>&1 &
UMBP_DEMO_PID=$!
```

观察日志确认服务启动，再让客户端实际连接；socket 文件存在本身不是 ready 验证：

```bash
python3 examples/umbp/local_kv_demo.py \
  --mode standalone --address "$UMBP_DEMO_ADDRESS"
```

上例关闭 gather stream 预热以减少功能检查的启动开销；正常 serving 的性能配置另行测量。
多 worker 需要相同地址和 `auto_start=False`：

```python
from mori.umbp import UMBPClient, UMBPConfig, UMBPStandaloneProcessConfig

config = UMBPConfig()
process = UMBPStandaloneProcessConfig()
process.address = "unix:///tmp/your-run/server.sock"
process.auto_start = False
config.standalone_process = process
client = UMBPClient(config)
```

**池容量由 server 的配置决定。** worker 中设置 `config.dram.capacity_bytes` 不会重设
已经启动的 server。服务拥有对象；一个 worker 关闭自己的 client 不等于清空整个共享池。

CPU 指针不能直接跨进程使用。共享 host buffer 的正确顺序是：

```python
from mori.umbp import UMBPHostMemAllocator, UMBPHostBufferBacking

allocator = UMBPHostMemAllocator()
buffer = allocator.alloc(4096, UMBPHostBufferBacking.AnonymousShm)
if not buffer or not client.register_memory(buffer.ptr, buffer.requested_size):
    raise RuntimeError("Shared buffer registration failed")
# 在该区域内执行 put/get；保留 buffer 和 allocator，直到操作完成。
# 以下清理在全部读写成功完成后执行。
client.deregister_memory(buffer.ptr)
del client
allocator.free(buffer)
```

UMBP 通过 fd handoff 建立共享映射；请求描述区域和字节偏移。
普通 `ctypes` 或常规 Python heap buffer 适用于 embedded，不能替代此处的共享分配。
GPU 缓冲区使用客户端支持的内存注册接口，建立同机 GPU IPC 映射；应保留原分配、设备身份和注册生命周期。

结束自己的 demo 服务后清理本次临时目录：

```bash
kill -TERM "$UMBP_DEMO_PID"
wait "$UMBP_DEMO_PID"
rm -r -- "$UMBP_DEMO_DIR"
```

本机 SSD standalone 可在新服务中设置 `UMBP_DISTRIBUTED_MEDIUM=SSD`、`UMBP_SSD_ENABLED=1`、
`UMBP_SSD_DIR` 和 `UMBP_SSD_CAPACITY`；客户端仍连接同一个 socket。
standalone 仅设置 `UMBP_SSD_ENABLED=1` 而未选择 medium 时会报配置错误。

## 6. API 速查与语义

所有 size 和 offset 都是**字节数**。返回值必须检查；batch 返回每个 key 的成功状态。
以下参数名与当前 Python binding 一致：

| API | 作用与返回值 |
|---|---|
| `put_from_ptr(key, src, size)` | 写入一个对象，返回 `bool` |
| `get_into_ptr(key, dst, size)` | 恢复对象到目标，返回 `bool` |
| `exists(key)` / `batch_exists(keys)` | 单个存在性 / 每 key 存在性列表 |
| `batch_put_from_ptr(keys, ptrs, sizes)` | 批量写入，返回 `list[bool]` |
| `batch_get_into_ptr(keys, ptrs, sizes)` | 批量恢复，返回 `list[bool]` |
| `batch_put_from_ptr_with_depth(keys, ptrs, sizes, depths)` | 附带 radix 深度的批量写入 |
| `batch_exists_consecutive(keys)` | 从索引 0 起，至第一个 miss 前的连续 key 个数 |
| `batch_put_ranges_from_ptr(keys, object_sizes, ptrs, sizes, dst_offsets)` | 多缓冲区组装对象，按 key 发布 |
| `batch_get_ranges_into_ptr(keys, ptrs, sizes, src_offsets)` | 将对象的多个字节范围恢复到多个目标缓冲区 |
| `supports_ranged_io()` | 检查当前 backend 是否实现 range I/O |
| `register_memory(ptr, size, loc=CPU, device=-1)` | 注册调用者缓冲区；GPU 应提供正确位置和 device |
| `deregister_memory(ptr)` | 在读写完成后解除注册 |
| `flush()` | 同步待处理工作；不能替代 GPU producer 同步 |
| `clear()` | 清空存储；共享池操作需由服务使用者协调 |

`ptrs`、`sizes` 和 offsets 的 range 参数是二维列表，外层对应 key，内层对应该对象的各个范围。
PUT 范围必须恰好覆盖 `[0, object_sizes[i])`，不得重叠或留下空洞。GET 可以选取对象中的部分范围。
shape 错误会让整批失败；单个对象的数据错误按 key 返回失败。

例如，三个分散的 4 字节 buffer 组装 `AAAABBBBCCCC`：

```python
# p_a / p_b / p_c 分别指向 4 字节的 A/B/C 数据。
ok = client.batch_put_ranges_from_ptr(
    ["object"], [12], [[p_b, p_c, p_a]], [[4, 4, 4]], [[4, 8, 0]]
)
# 读取 A 和 C 到两个目标地址。
ok = client.batch_get_ranges_into_ptr(
    ["object"], [[dst_a, dst_c]], [[4, 4]], [[0, 8]]
)
```

key 按不可变对象使用：相同 key 再次 PUT 不用于更新旧值。
使用模型/布局/dtype/页面标识或内容 hash 组织 key，并为不同运行或不兼容布局划分 namespace。
`exists()` 之后对象仍可能被回收，GET 失败必须能退回重算。

该主线版本的 Python binding 没有公开 `close()`。读写完成后解除 buffer 注册、
释放最后一个 client 引用，C++ 析构函数会调用 `Close()` 清理资源。
示例中的 `del client` 适用于没有其他引用的客户端；框架集成需统一管理引用和关闭时机。

## 7. 配置速查

`UMBPConfig()` 与 `UMBPConfig.from_environment()` 在构造时读取环境默认值；
随后显式赋字段覆盖该配置对象。不要期望修改环境变量能更改已经运行的 pool 或 server。
部分 runtime knob 在首次使用时缓存，调整后应启动新进程。

| 配置项 | 环境变量 | 单位或说明 |
|---|---|---|
| `dram.capacity_bytes` | `UMBP_DRAM_CAPACITY` | 字节；当前默认 4 GiB |
| `dram.use_hugepages` | `UMBP_DRAM_USE_HUGEPAGES` | 0/1；默认关闭，查看实际 backing |
| `dram.numa_nodes` | `UMBP_DRAM_NUMA_NODE` | 如 `0` 或 `0,1`；留空为不绑定 |
| `dram.prefault` / `prefault_threads` | `UMBP_DRAM_PREFAULT` / `UMBP_DRAM_PREFAULT_THREADS` | 启动时触页、线程数；范围 0..16 |
| `dram.high_watermark` / `low_watermark` | `UMBP_DRAM_HIGH_WM` / `UMBP_DRAM_LOW_WM` | 当前默认 0.95 / 0.90 |
| `ssd.enabled` | `UMBP_SSD_ENABLED` | 默认配置为 True；仍需显式选择 SSD 或 policy |
| `ssd.storage_dir` / `capacity_bytes` | `UMBP_SSD_DIR` / `UMBP_SSD_CAPACITY` | 路径 / 字节；多目录时容量是合计值 |
| `ssd.direct_io` / `verify_crc` | `UMBP_SSD_DIRECT_IO` / `UMBP_SSD_VERIFY_CRC` | 当前默认均开启 |
| `ssd.durability.mode` | `UMBP_SSD_DURABILITY` | `strict`（默认）或 `relaxed` |
| `standalone_process.address` | `UMBP_STANDALONE_ADDRESS` | 地址字段供 API 使用；环境变量供 server / launcher 使用 |
| 服务程序路径 | `UMBP_STANDALONE_BIN` | `mori.umbp` import 会发现安装包附带的二进制 |
| 日志等级 | `MORI_UMBP_LOG_LEVEL` | 例如 `INFO` |

具体 Python 字段见 [配置 binding](../../src/pybind/pybind_umbp.cpp)，
默认值及环境读取见 [config.h](../../src/umbp/include/umbp/common/config.h)。
完整环境参数见 [runtime-env-vars.md](../../src/umbp/doc/runtime-env-vars.md)。

多层存储使用 `config.backend_policy_path`，显式定义 entry tier、backend 容量、
放置权重、offload 和 promotion 策略。见 [策略示例](../../examples/umbp/multi_backend_policy.json)
与 [tier 管理说明](../UMBP-TIER-MANAGEMENT-BENCHMARK.md)。策略文件的容量会优先于 legacy medium 的容量字段；
采用策略后应从实际日志或 metrics 核对分配，而不是根据环境变量估算。

## 8. 生命周期与性能

本机的 pointer/range 方法是阻塞调用，但不会自动等待框架之前在任意 GPU stream 上提交的 producer。
调用者需确保源 KV 已完成写入，GET 完成后再调度 consumer，不能提前释放或复用相关页面。
保留 tensor 的底层分配和注册，buffer 大小至少覆盖全部传输范围。

GPU 传输或 RPC 异常后，失败返回值不总能证明另一进程已经停止访问缓冲区。
遇到完成状态不明的异常，应保留相关映射和对象，停止该 worker 并重新建立服务连接；
不要复用该内存继续推理。正常清理顺序为停止新请求、完成已有工作、解除注册、
释放最后一个 client 引用、释放 buffer。

优化时按瓶颈分阶段测量：

1. **批量元数据与数据操作。** 用 batch exists 与 batch range I/O，减少 Python/C++ 切换和每 key 请求开销；
   连续前缀匹配使用 `batch_exists_consecutive`，不能把任意命中数量当成可恢复前缀长度。
2. **页面容量。** DRAM 按页分配；粗略占用为 `ceil(value_bytes / page_bytes) * page_bytes`。
   KV 的 token page size 与 UMBP 的 byte page size 不同，要按实际模型池布局换算。
3. **共享容量。** 多 rank 共用一个 standalone，注册长寿命缓冲区，避免每轮分配和注册。
   共享 128 GiB payload 与每 rank 128 GiB 是不同预算；RSS 还包含元数据、映射和临时工作区。
4. **NUMA 与启动。** 先检查 GPU/CPU/内存拓扑，安排服务 CPU 与 host buffer 的 NUMA placement。
   prefault 与 hugepages 的启动开销、实际 backing、恢复吞吐分别记录，避免只看请求耗时。
5. **SSD。** 在真实设备上查看读写字节、带宽、CRC 开销、I/O ring 并发和 staging 压力。
   `relaxed` durability 适用于允许重建的缓存，但要明确恢复语义；先保留 CRC 做正确性基线。
6. **框架收益。** 同时报告 miss/fill、host hit/replay、端到端 latency 和 token/s。
   固定模型、dtype、GPU KV 容量、host payload、并发和执行模式，重复多轮再比较。

## 9. 常见问题

| 现象 | 检查与处理 |
|---|---|
| `mori was built without UMBP` | 检查构建开关、gRPC/Protobuf 依赖以及 `libmori_umbp_pybinds.so` |
| standalone 无法连接 | 确认地址、socket 所在目录权限、server 日志；用真实 client 探测 ready |
| `requires an AnonymousShm-backed host buffer` | 用 `UMBPHostMemAllocator` 的 `AnonymousShm` 分配并注册 |
| SSD 开启却仍只有 DRAM | 显式选择 SSD medium，或提供 backend policy；核对 server 的实际配置 |
| 小池 PUT 失败 | 检查 page size 与容量、对象字节数、水位；逐个检查 batch 结果 |
| 重复 PUT 没有更新 | key 按不可变对象使用；更换 key/namespace |
| GPU 数据恢复后异常 | 检查 dtype/layout/范围、device、producer 就绪与目标内存生命周期 |
| HTTP 正常但没有 host hit | 两次小请求通常可从 GPU 命中；用固定 fill/replay 工作负载，并检查 Linker 日志与 cache details |

基础接口验证使用 [local_kv_demo.py](../../examples/umbp/local_kv_demo.py)；
框架接入验证继续 [SGLang demo](sglang-demo.zh-CN.md)。
