# UMBP SGLang KV Event Bridge API 设计

## 1. 目标

`UMBPSglangKvEventBridge` 用于订阅 SGLang 的 ZMQ KV events，并把事件转换成 UMBP master 可识别的 External KV metadata 上报。

核心链路：

```text
SGLang ZMQ KVEventBatch
  -> decode BlockStored / BlockRemoved / AllBlocksCleared
  -> map StorageMedium / block_hash
  -> call UMBP master External KV API
```

这个 API 只同步 metadata，不搬运 KV bytes。

## 2. API 定位

Bridge 应作为独立 sidecar / adapter API，而不是放进 `UMBPClient` 数据面。

| 对象 | 职责 |
| --- | --- |
| `UMBPSglangKvEventBridgeConfig` | 描述 SGLang event source、UMBP master target、hash / tier 映射和运行参数 |
| `UMBPSglangKvEventBridge` | 订阅 ZMQ event，解码并转发为 UMBP External KV report / revoke 操作 |

## 3. 使用示例

```python
from mori.umbp import (
    UMBPSglangKvEventBridge,
    UMBPSglangKvEventBridgeConfig,
    UMBPTierType,
)

cfg = UMBPSglangKvEventBridgeConfig(
    master_address="127.0.0.1:15558",
    node_id="worker-0",

    # 推荐：从 SGLang /server_info 自动发现 endpoint / topic / dp_size / block_size。
    sglang_server_info_url="http://127.0.0.1:30000/server_info",

    # 可选：手动覆盖 SGLang event source。
    zmq_endpoint=None,
    zmq_topic=None,
    dp_size=None,

    medium_to_tier={
        "GPU": UMBPTierType.HBM,
        "CPU_PINNED": UMBPTierType.DRAM,
        "DISK": UMBPTierType.SSD,
    },
    hash_format="decimal",
    ignore_unknown_medium=True,
)

bridge = UMBPSglangKvEventBridge(cfg)
bridge.start()

# ...

bridge.stop()
```

多 worker 场景建议使用 `sources[]`，每个 source 独立配置 `node_id` 和 `server_info_url`：

```python
cfg = UMBPSglangKvEventBridgeConfig(
    master_address="127.0.0.1:15558",
    sources=[
        UMBPSglangKvEventSource(
            node_id="worker-0",
            sglang_server_info_url="http://worker-0:30000/server_info",
        ),
        UMBPSglangKvEventSource(
            node_id="worker-1",
            sglang_server_info_url="http://worker-1:30000/server_info",
        ),
    ],
)
```

## 4. Config 字段

| 字段 | 说明 |
| --- | --- |
| `master_address` | UMBP master 地址 |
| `node_id` | 单 source 模式下，表示当前订阅的 SGLang worker / UMBP node |
| `sources` | 多 worker 模式下的 source 列表；每个 source 独立配置 `node_id` 和 `sglang_server_info_url` |
| `sglang_server_info_url` | 推荐入口，通过 SGLang `/server_info` 自动发现 KV event publisher |
| `zmq_endpoint` | 可选，手动指定 ZMQ endpoint；设置后覆盖 `server_info` |
| `zmq_topic` | 可选，ZMQ topic；默认从 `server_info.kv_events.topic` 读取 |
| `dp_size` | 可选，DP rank 数；默认从 `server_info.kv_events.dp_size` 读取 |
| `medium_to_tier` | SGLang `StorageMedium` 到 UMBP tier 的映射 |
| `hash_format` | block hash 编码格式，默认 `decimal` |
| `ignore_unknown_medium` | 遇到未知 medium 时跳过还是报错 |
| `batch_report_size` | 按 action 和 tier 聚合的 block hash 数量阈值，达到后批量 report / revoke |
| `poll_timeout_ms` | `poll_once()` 的 ZMQ poll timeout |

## 5. Runtime API

| API | 作用 |
| --- | --- |
| `start()` | 启动后台线程，订阅 SGLang ZMQ events 并上报 UMBP |
| `stop()` | 停止后台线程，关闭 ZMQ socket 和 master client |
| `poll_once(timeout_ms=None)` | 手动处理一批事件，便于测试或同步集成 |
| `flush()` | 尽量处理完 pending events |
| `stats()` | 返回处理、上报、撤销、跳过、失败计数 |
| `last_sequence(dp_rank)` | 返回某个 DP rank 最近处理到的 sequence |

## 6. SGLang Event 格式

SGLang 的 ZMQ publisher 发送 multipart message：

```text
[topic, seq_bytes, payload]
```

| Frame | 内容 |
| --- | --- |
| `topic` | ZMQ topic，例如 `kv-events` |
| `seq_bytes` | 8-byte big-endian sequence number |
| `payload` | `msgspec.msgpack` 编码后的 `KVEventBatch` |

Bridge 解码时应使用：

```python
from msgspec.msgpack import Decoder
from sglang.srt.disaggregation.kv_events import KVEventBatch

decoder = Decoder(type=KVEventBatch)
batch = decoder.decode(payload)
```

`KVEventBatch` 中可能包含：

| Event | 含义 |
| --- | --- |
| `BlockStored` | 某个 KV block 被写入某个 medium |
| `BlockRemoved` | 某个 KV block 从某个 medium 移除 |
| `AllBlocksCleared` | 当前 cache 被整体清空 |

## 7. Event 到 UMBP 的映射

| SGLang event | UMBP 动作 |
| --- | --- |
| `BlockStored(block_hashes, medium="GPU")` | `report_external_kv_blocks(node_id, hashes, HBM)` |
| `BlockStored(block_hashes, medium="CPU_PINNED")` | `report_external_kv_blocks(node_id, hashes, DRAM)` |
| `BlockStored(block_hashes, medium="DISK")` | `report_external_kv_blocks(node_id, hashes, SSD)` |
| `BlockRemoved(block_hashes, medium=...)` | `revoke_external_kv_blocks(node_id, hashes, mapped_tier)` |
| `AllBlocksCleared()` | 对该 `node_id` 的所有 mapped tier 执行 revoke-all |

推荐默认 medium 映射：

| SGLang `StorageMedium` | UMBP tier |
| --- | --- |
| `GPU` | `UMBPTierType.HBM` |
| `CPU_PINNED` | `UMBPTierType.DRAM` |
| `DISK` | `UMBPTierType.SSD` |
| `EXTERNAL` | 首期跳过，后续可扩展 |

## 8. DP Rank 处理

SGLang 在 DP attention 下每个 DP rank 一个 publisher。端口规则是：

```text
port = endpoint_port_base + dp_rank
```

例如：

| DP rank | Endpoint |
| --- | --- |
| `0` | `tcp://host:5557` |
| `1` | `tcp://host:5558` |
| `2` | `tcp://host:5559` |

Bridge 应为每个 DP rank 建立一个 SUB socket，并分别维护：

| 状态 | 说明 |
| --- | --- |
| `last_seq` | 最近处理的 sequence |
| `dropped_seq_count` | 检测到的 sequence gap |
| `last_event_ts` | 最近 event batch timestamp |

`KVEventBatch.attn_dp_rank` 可以用于校验事件来源是否与当前 socket 对应。

## 9. Hash 编码

SGLang `block_hashes` 是 `int`。UMBP External KV metadata 建议统一使用 string key。

默认编码：

```python
def encode_hash(h: int) -> str:
    return str(h)
```

可选格式：

| `hash_format` | 示例 |
| --- | --- |
| `decimal` | `"123456789"` |
| `hex` | `"0x75bcd15"` |
| `sglang:int64` | `"sglang:int64:123456789"` |

首期建议使用 `decimal`，简单且稳定。

## 10. 错误处理

| 场景 | 建议行为 |
| --- | --- |
| ZMQ 暂时无消息 | `poll_once()` 返回 `0` |
| msgpack decode 失败 | 计入 `decode_errors`，跳过该 batch |
| sequence gap | 计入 `dropped_batches`，用于观测是否存在事件丢失 |
| unknown medium | 默认跳过，计入 `skipped_unknown_medium` |
| UMBP master RPC 失败 | 有限重试；失败后计入 `report_errors` / `revoke_errors` |
| `AllBlocksCleared` | 对所有 mapped tier 执行 revoke-all，保证 metadata 收敛 |

## 11. UMBP 侧依赖

Bridge 应调用 UMBP External KV metadata API，而不是 UMBP-owned KV API。

需要的 master API：

```python
report_external_kv_blocks(node_id, hashes, tier)
revoke_external_kv_blocks(node_id, hashes, tier)
revoke_all_external_kv_blocks_at_tier(node_id, tier)
```

原因：

| 路径 | 是否适合 |
| --- | --- |
| External KV metadata API | 适合。SGLang / HiCache 持有 KV bytes，UMBP 只记录 placement metadata |
| UMBP-owned `PublishLocalBlock` / `Put` path | 不适合。该路径表示 KV bytes 由 UMBP 管理或可通过 UMBP 数据面读取 |

## 12. 自动发现

如果设置 `sglang_server_info_url`，Bridge 可从 SGLang `/server_info` 读取：

| 字段 | 用途 |
| --- | --- |
| `kv_events.publisher` | 判断是否为 `zmq` |
| `kv_events.endpoint_host` | ZMQ host；若为 `*` / `0.0.0.0`，订阅端应替换为 worker host |
| `kv_events.endpoint_port_base` | DP rank 0 的 base port |
| `kv_events.topic` | ZMQ subscribe topic |
| `kv_events.block_size` | 与 SGLang page size 对齐，供后续 hash 校验或 prompt-side index 使用 |
| `kv_events.dp_size` | 需要订阅的 DP rank 数 |

手动配置 `zmq_endpoint` 时，Bridge 可跳过 `/server_info` 自动发现。

## 13. 推荐实现边界

首期建议只实现：

| 能力 | 是否首期实现 |
| --- | --- |
| ZMQ SUB 订阅 | 是 |
| `KVEventBatch` 解码 | 是 |
| DP rank 多端口订阅 | 是 |
| `BlockStored` report | 是 |
| `BlockRemoved` revoke | 是 |
| `AllBlocksCleared` revoke-all | 是 |
| sequence gap 统计 | 是 |
| bytes put/get | 不做，继续只走 metadata |

## 14. 总结

`UMBPSglangKvEventBridge` 是一个 SGLang KV event 到 UMBP External KV metadata 的桥接 API。

它把 SGLang 的 `BlockStored`、`BlockRemoved`、`AllBlocksCleared` 转换成 UMBP master 的 `report` / `revoke` / `revoke-all` 操作，用于让 UMBP 发现 SGLang / HiCache 持有的 KV cache placement，从而支持跨 worker KV 查询和调度。
