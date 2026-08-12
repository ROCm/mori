# InterNodeV1 Token 编码逻辑分析

本文梳理 `src/ops/dispatch_combine/internode_v1.cpp` 中跨节点 dispatch/combine 的
token 索引体系。所有行号对应当前 `internode_v1.cpp`(1470 行)。

阅读顺序建议:先看「五个编号空间」建立词汇表,再看「端到端实例」把词汇串起来,
最后看「#475」了解编号串台的后果。

---

## 1. 两套编解码函数

`src/ops/dispatch_combine/common.hpp` 提供两组 encode/decode helper。两者形式相同
(`a * stride + b`),但含义和步长不同,是最容易混淆的地方。

```cpp
// stride = MaxNumTokensToSend() = worldSize * S
FlatTokenIndex(config, pe, localTokId)          // (PE, 该 PE 上的行号) → flat
PeFromFlatTokenIndex(config, flatIdx)
LocalTokIdFromFlatTokenIndex(config, flatIdx)
NullFlatTokenIndex(config)                      // = worldSize * MaxNumTokensToSend()

// stride = MaxNumTokensToSendPerRank() = S = maxNumInpTokenPerRank
SendBufSlotOffset(config, pe, slotId)           // (区号, 槽号) → flat
PeFromSendBufSlotOffset(config, flatIdx)
SlotIdFromSendBufSlotOffset(config, flatIdx)
NullSendBufSlotOffset(config)                   // = worldSize * S
```

| | 编码内容 | 步长 | 用途 |
|---|---|---|---|
| `FlatTokenIndex` | (PE, 该 PE 上的行号) | `worldSize * S` | 映射表里记录"贡献在谁那儿" |
| `SendBufSlotOffset` | (区号, 槽号) | `S` | 定位 buffer 里的槽位 |

**Null 哨兵靠越界解码被顺手过滤。** 两个 Null 值都等于 `worldSize * 步长`,解码出的
PE 号恰为 `worldSize`,于是 `destNode = worldSize / gpuPerNode` 必然不等于任何真实
节点号。这就是为什么代码里只写 `if (destNode == myNode)` 就同时完成了「是否本节点」
和「是否 Null」两重判断,不需要单独判空。

---

## 2. 五个编号空间

### ① 源 tokenId

本 rank 的输入下标,范围 `[0, curRankNumToken)`。索引 `tokenIndices`、`combineOut`、
`dispatchStaging`、以及 `staging` 后半区中属于本节点的那一格。

### ② destTokId — 发往某节点的压缩排队号

发送端按 `shouldSend` 掩码压缩得到,与 ① 不连续:

```203:207:src/ops/dispatch_combine/internode_v1.cpp
        index_t destTokIdOffset = flagSlotId * warpSize;

        uint64_t warpOffset = 0;
        if (laneId > 0) warpOffset = __popcll(mask << (warpSize - laneId));
        index_t destTokId = destTokIdOffset + warpOffset;
```

压缩保序,因此源侧连续的发送者对应目标侧连续的槽,`count` 逻辑得以把一段 run 合并
成单次 RDMA。① → ② 的唯一翻译途径是 `interNodeDispSendMap[nNodes * tokenId + node]`,
该表存放在**源 PE**。

### ③ tokIdx — 接收槽的扁平号

`tokIdx = SendBufSlotOffset(config, srcNode, ②)`。收发两端各自独立计算却必然相等
(见 §3.2),这是整个协议的对齐点。它同时索引 `dispatchInp` 的 payload、
`interNodeDispDestTokIdMap`、以及 `staging` 前半区的槽。

### ④ 专家卡上的行号

转发时在目标卡的计数器上原子抢占得到,与 (PE 号) 一起打包存入映射表:

```443:448:src/ops/dispatch_combine/internode_v1.cpp
            destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
            assert(destTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
            args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + e] =
                FlatTokenIndex(config, destPe, destTokId);
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] = srcTokId;
```

索引 `dispatchOut` / `combineInp`。

### ⑤ staging 区号

前半 `[0, nNodes)` 按**目的**节点编号(代理身份,替别人攒 partial),后半
`[nNodes, 2*nNodes)` 按**来源**节点编号(收别人回传的 partial)。

---

## 3. 端到端实例

### 3.0 配置

```
nNodes=2, gpuPerNode=3, worldSize=6      node0: PE0 PE1 PE2 | node1: PE3 PE4 PE5
maxNumInpTokenPerRank = 128   →  S = MaxNumTokensToSendPerRank() = 128
MaxNumTokensToSend() = 6*128 = 768       NullFlatTokenIndex = 6*768 = 4608
numExpertPerRank = 2, topk = 3, warpSize = 64
专家号 → PE:  0,1→PE0   2,3→PE1   4,5→PE2   6,7→PE3   8,9→PE4   10,11→PE5
```

追踪 node0.PE0 上的 **tokenId = 5**,路由 `tokenIndices[15..17] = {1, 9, 10}`:

- 专家 1 → PE0(本卡)
- 专家 9 → node1.PE4
- 专家 10 → node1.PE5

为使例子干净,假设同一 warp 内只有 token 4、5、6 需要发往 node1(连续三个)。

### 3.1 发送端:① → ②

代理按**卡内序号**配对:`proxyPe = 1*3 + (0 % 3) = PE3`。node0.PE0 只与 node1.PE0
(全局 PE3)通信。

`mask` 的 bit 4/5/6 置位,`flagSlotId = 0`,lane5 前面只有 lane4,故 `warpOffset = 1`:

| lane / tokenId | 4 | **5** | 6 |
|---|---|---|---|
| destTokId | 0 | **1** | 2 |

```222:233:src/ops/dispatch_combine/internode_v1.cpp
          size_t remoteIdx = SendBufSlotOffset(config, myNode, destTokId);
          if (count > 0) {
            size_t stagingTokOffset = tokenId * xferBytes;
            int qpId = (tokenId / warpSize) % config.numQpPerPe;
            shmem::ShmemPutMemNbiSignalThread(
                args.interNodeV1TokBufs.dispatchInp, remoteIdx * xferBytes,
                args.interNodeV1TokBufs.dispatchStaging, stagingTokOffset, count * xferBytes,
                args.interNodeChunkFlagMemObj,
                (myNode * maxChunkNum + flagSlotId) * sizeof(uint64_t), flag,
                core::atomicType::AMO_ADD, proxyPe, qpId);
          }
          if (!args.replayMode) args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
```

- 源地址用 `tokenId`(①),目标地址用 `remoteIdx = SendBufSlotOffset(0, 1) = 1`
  (③,区号 = **自己的节点号 0**)。
- `interNodeDispSendMap[2*5 + 1] = 1`,存在 node0.PE0,§3.6 要用。
- `dispDestTokIdMap[5*3+1]`、`[5*3+2]` 置 Null,告知本节点 intra-node 路径这两个专家
  不归它管。

### 3.2 代理 PE3 收货:③ 的对齐

```406:409:src/ops/dispatch_combine/internode_v1.cpp
    for (int j = startTokenIdx + (blockId % numRecvBlock) * warpNum + warpId; j < endTokenIdx;
         j += numRecvBlock * warpNum) {
      int tokIdx = SendBufSlotOffset(config, node, j);
      index_t* indices = reinterpret_cast<index_t*>(stagingPtr + tokIdx * xferBytes + hiddenBytes);
```

`node = (myNode + 1 + i) % nNodes = 0`,`chunkFlag` 表明本 chunk 有 3 个 token,`j` 扫
0/1/2。目标 token 位于 `j = 1` → `tokIdx = SendBufSlotOffset(0, 1) = 1`。

**与发送端的 `remoteIdx` 恒等**:发送端的 `myNode` 即接收端看到的 `node`,发送端的
`destTokId` 即接收端的 `j`。两边各自计算、从不传递。

PE3 从 payload 读出 `indices = {1, 9, 10}`,逐专家判归属:

| e | 专家 | destPe | 判定 | `interNodeDispDestTokIdMap[1*3+e]` |
|---|---|---|---|---|
| 0 | 1 | PE0 | destNode=0 ≠ 1,跳过 | `4608` (Null) |
| 1 | 9 | PE4 | 本节点,抢到行 **17** | `FlatTokenIndex(4,17) = 3089` |
| 2 | 10 | PE5 | 本节点,抢到行 **8** | `FlatTokenIndex(5,8) = 3848` |

随后用 XGMI `WarpCopy` 把 hidden/indices/weights 写入 PE4 的 `dispatchOut[17]` 和
PE5 的 `dispatchOut[8]`。

### 3.3 专家卡的视野

PE4 只看到自己 `dispatchOut` 第 17 行,算完写回自己 `combineInp` 第 17 行。它手上唯一
的溯源信息是 `dispTokIdToSrcTokId[17] = 5`;destTokId=1 与 tokIdx=1 它从未见过。

**因此回传不可能由专家卡发起,只能由代理 PE 完成。**

### 3.4 代理 PE3 做 combine:③ 被重算

`tokIdx` 不做保存,而是用同样的 bid 分解 + `SendBufSlotOffset` 重算,再次得到 1:

```930:947:src/ops/dispatch_combine/internode_v1.cpp
              int tokIdx = SendBufSlotOffset(config, node, j);

              if (laneId < config.numExpertPerToken) {
                srcPtrs[laneId] = nullptr;
                srcWeightsPtr[laneId] = nullptr;
                index_t destTokId =
                    args.interNodeDispDestTokIdMap[tokIdx * config.numExpertPerToken + laneId];
                index_t destPe = PeFromFlatTokenIndex(config, destTokId);
                index_t destNode = destPe / config.gpuPerNode;
                if (destNode == myNode) {
                  index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
                  srcPtrs[laneId] =
                      args.interNodeV1TokBufs.combineInp->template GetAs<TokT*>(destPe) +
                      destLocalTokId * hiddenDim;
```

- lane0 读到 4608 → `4608/768 = 6` → `destNode = 2 ≠ 1` → 指针留 null(Null 过滤)。
- lane1 读到 3089 → PE4 行 17 → 指向 PE4 的 `combineInp[17]`,跨卡 XGMI 读。
- lane2 读到 3848 → PE5 行 8。

`WarpAccum` 累加后写入 `staging + tokIdx * tokCombXferBytes`,即**区 0 槽 1**。

### 3.5 回传:⑤ 的区号约定

```981:986:src/ops/dispatch_combine/internode_v1.cpp
              shmem::ShmemPutTypeNbiWarp<uint8_t>(
                  args.interNodeV1TokBufs.staging,
                  SendBufSlotOffset(config, myNode + nNodes, startTokenIdx) * tokCombXferBytes,
                  args.interNodeV1TokBufs.staging,
                  SendBufSlotOffset(config, node, startTokenIdx) * tokCombXferBytes,
                  thisChunkTokenNum * tokCombXferBytes, proxyPe, qpId);
```

源 = 区 `node = 0`(替 node0 攒的);目的 = 区 `myNode + nNodes = 3`,在对方视角即
"来自 node1"。`proxyPe = 0*3 + (3 % 3) = PE0`,回到出发的那张卡。**槽号 1 全程不变。**

node0.PE0 上 staging 的最终形态:

```
staging on node0.PE0        每区 S=128 槽
┌──────────┬──────────┬───────────────────┬───────────────────┐
│  区 0    │  区 1    │  区 2 = nNodes+0  │  区 3 = nNodes+1  │
│ 发往node0│ 发往node1│  本节点的 partial │  node1 回传的     │
│ (代理用) │ (代理用) │  槽号 = tokenId   │  槽号 = destTokId │
└──────────┴──────────┴───────────────────┴───────────────────┘
                            槽 5 ← 专家1        槽 1 ← 专家9+10
                            (flat 261)          (flat 385)
```

后两区槽号编法不同,因为写它们的人不同:区 2 由本卡 intra-node 路径写,手里是原始
`tokenId`;区 3 由远端代理写,手里只有压缩后的 `destTokId`。

### 3.6 EpCombineAll:② → ①,回到原点

```1333:1334:src/ops/dispatch_combine/internode_v1.cpp
  uint8_t* stagingPtr = args.interNodeV1TokBufs.staging->template GetAs<uint8_t*>() +
                        SendBufSlotOffset(config, nNodes, 0) * combXferBytes;
```

先跳过前 `nNodes` 个区(偏移 `2*128 = 256`),此后 `SendBufSlotOffset(n, ...)` 落在区
`nNodes + n`。

```1357:1364:src/ops/dispatch_combine/internode_v1.cpp
    for (int n = 0; n < nNodes; n++) {
      if (__any(laneNode == n) && (laneId == 0)) {
        int mappedId = (n == myNode) ? tokenId : args.interNodeDispSendMap[nNodes * tokenId + n];
        uint8_t* base = stagingPtr + SendBufSlotOffset(config, n, mappedId) * combXferBytes;
        srcPtrs[n] = reinterpret_cast<T*>(base) + hiddenDimOffset;
        srcWeightsPtrs[n] = reinterpret_cast<float*>(base + hiddenBytes);
      }
    }
```

`laneNode` 由 `args.tokenIndices[15..17] = {1,9,10}` 推得 `{0,1,1}`:

| n | `__any(laneNode==n)` | mappedId | flat 槽 | 命中 |
|---|---|---|---|---|
| 0 (=myNode) | 真(专家 1) | `tokenId = 5` | 256+5 = **261** | 区 2 槽 5 |
| 1 | 真(专家 9,10) | `sendMap[2*5+1] = 1` | 256+129 = **385** | 区 3 槽 1 |

两个指针都对上 §3.5 的布局,`WarpAccum` 累加进 `combineOut[5]`,token 回到出发编号。

`n == myNode` 用 ① 而其余用 ②,这个三目运算符是整套编码体系的收口:本节点 partial
由自己按原始 tokenId 写下,远端 partial 落在压缩槽位,必须查表翻译。

### 3.7 全链路

```
node0.PE0                                    node1.PE3(代理)        node1.PE4/PE5
tokenId 5  ──压缩──▶ destTokId 1
  ①                    ②
                        │ SendBufSlotOffset(0, 1)
                        └──── RDMA ────▶ dispatchInp 槽 1 = tokIdx 1
                                             ③   │
                                                 │ atomicAdd 抢行
                                                 └──▶ dispatchOut 行 17 / 行 8  ④
                                                              │ 专家计算
                                             tokIdx 1 ◀── XGMI 读 combineInp 行 17 / 行 8
                                                 │ (查 interNodeDispDestTokIdMap)
staging 区3 槽1 ◀──── RDMA ──────────────────────┘
  ⑤    │
       │ sendMap[2*5+1] = 1 反查
combineOut[5]
  ①
```

---

## 4. Buffer 与编号空间对照

```
 dispatchStaging   ①            我的 token, 打包待发
 dispatchInp       ③            收到的槽, 区号 = 来源节点
 dispatchOut       ④            交付给专家的行
 combineInp        ④            专家算完的结果(同一行号)
 staging 前半      ③            替来源节点攒的 partial(槽号沿用 ②)
 staging 后半      ① 或 ②       本节点那格用 ①, 远端那格用 ②
 combineOut        ①            最终输出
```

两张映射表的存放位置决定了各自的职责边界:

- `interNodeDispSendMap` 在**源 PE**,负责 ① ⇄ ②;
- `interNodeDispDestTokIdMap` 在**代理 PE**,负责 ③ → ④。

专家卡两头都摸不到,这从编码层面解释了为什么回传必须经代理 PE 汇聚。

---

## 5. #475:编号空间串台

`EpCombineAll` 中的 `args.tokenIndices` 必须是按 ① 排列的**原始路由表**,因为它的下标
是 `tokenId`(`internode_v1.cpp:1291` 与 `:1345`)。但调用方传入的是 dispatch 输出的
indices —— 那张表按 ④ 排列,形状为 `[max_recv, topk]`。

由于两者行号都是小整数且不越界,下标 `15..17` 依然能读到三个合法专家号,只不过属于
**另一个 token**。于是 `__any(laneNode == n)` 这道门开错:

- 误判为"没有专家在 node1":`srcPtrs[1]` 留 null,区 3 槽 1 的远端结果被整块丢弃,
  表现为静默少加。
- 误判为"有专家在 node0"而实际没有:区 2 槽 5 的陈旧内容被加入,表现为静默多加。

复现扫描测得的 0.387% 丢失率,恰等于随机 token 的全部专家落在同一节点的概率,即
"误判正好导致整块丢失"的那一档。详见 `report_475_reproduction.md`。
