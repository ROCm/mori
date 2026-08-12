# ROCm/mori#475：InterNodeV1 跨节点 combine 静默损坏 —— 根因与修复

[ROCm/mori#475](https://github.com/ROCm/mori/issues/475) 的根因定位与修复。

基线 `0d05a4d2`（main），2 × (8× MI355X gfx950)，Pensando Pollara / `ionic` RoCE。
**根因不在 kernel，在 `combine()` 的入参约定**；修复只改 Python 绑定层，kernel 源码
未改动（`internode_v1.cpp` md5 `f589acc7…`）。

---

## 一、根因

`EpCombineAll` 用 `args.tokenIndices[tokenId * topk + lane]` 判断"本 rank 的第
`tokenId` 个 token 有没有专家落在节点 n"，据此决定要不要去读对端返回的 partial
（`internode_v1.cpp:1345`，fp8 变体在 `:1291`）。这里的 `tokenId` 是**本 rank 自己的**
token 序号，所以 `tokenIndices` 必须是本 rank 交给 `dispatch()` 的那份
`[num_token, topk]` 路由。

而 `combine()` 把调用方传入的 `indices` 直接绑到了 `args.tokenIndices`。调用方普遍
传的却是 **`dispatch()` 的返回值 `out_idx`**——那是"本 rank *收到的* token"的路由，
形状 `[max_recv, topk]`，与本 rank 自己的 token 毫无对应关系。于是那个判据是在拿
**另一个不相干 token 的路由**回答"我这个 token 要不要读对端"。

判错的两个方向正好对应现场观测到的两种损坏形态：

- 误判为**需要读**对端 → 凭空加进一份外来 payload（污染）
- 误判为**不需要读**对端 → 远端副本全丢（丢失）

比例可以精确对上。`uniform` 路由下一个 token 的 8 个专家恰好全落在同一节点的概率
约 `(1/2)^8 = 0.39%`，而判据读的是随机另一行，所以：

| token 形态 | 后果 | 预期 | 实测 |
|---|---|---|---|
| 专家全在本节点（local-only） | 几乎必然误判为要读对端 → 污染 | ≈ 99.6% | **42 / 42** |
| 专家全在对端（peer-only） | 同样误判，但本节点区在无本地专家时被写零，加零不可见 | 不可见 | 0 / 65 |
| 跨两节点（spanning） | 0.39% 概率误判为不用读 → 远端副本全丢 | ≈ 0.39% | 63 / 16277 = **0.387%** |

"丢失按 token 全有全无"、"污染源永远是同 rank 的邻近 token"、"只在批次含单节点局限
token 时才出错"、乃至"`remote` 单独成批时输出全零"，都是这一条的直接推论
（`remote` 批次全是 peer-only token，对端 partial 被系统性跳过）。

**判决实验**：只把传给 `combine()` 的张量从 `out_idx` 换成原始 `idx`，其余一概不动
——报告人原探针 T=16…512 由**每档 FAIL 变为全部 PASS**。

**为什么长期没暴露**：combine 侧只有 InterNodeV1 的 `EpCombineAll` 会读
`tokenIndices`，且只在 `nNodes > 1` 时才真正用到这个判据。IntraNode / LowLatency 的
combine 根本不碰它，传什么都对，所以单节点永远 clean。而 mori 唯一传对的调用点是
正确性测试 `dispatch_combine_test_utils.py:743`（传 `all_rank_indices[rank]`），它又跑
不到真正的两节点配置——见 §四。

---

## 二、修复

`python/mori/ops/dispatch_combine.py`：`dispatch()` 记住它实际散射用的 indices
（`:681`），`combine()` 优先用这一份而不是调用方传进来的（`:973`）；standard-MoE 的
那对入口同样处理（`:1368` / `:1469`）。

这样既修好了所有现存的误用，也不改变本来就传对的调用方的行为——那种情况下两者
是同一个张量。另外补了 `combine()` 的 docstring 说明这条约定，并给 `combine_indices`
加了一个 token 数下界校验，避免越界读。

顺带把 `tests/python/ops/bench_dispatch_combine.py` 的三处调用改成传原始 indices。
修复后它功能上已无影响（该入参被忽略），但作为示例仍具误导性。

---

## 三、验证

`MORI_SRC=1`，两节点 world=16，**调用方仍按原来的错误写法传 `out_idx`**：

| 验证 | 结果 |
|---|---|
| 报告人原探针 T=16/32/128/256/512 | 全部 clean，`RESULT: PASS`（修前每档 FAIL，T=512 为 107） |
| 解码探针 `uniform` / `mixed,remote` / `local,mixed` / `remote` / 四类混合 | 五种批次均 `0 / 16384` |
| 既有单节点测试（intranode + routing handle） | 342 passed, 640 skipped |

§一 那张按 token 形态分类的表出自一个解码探针（one-hot payload，使每个非零位置直接
指名是哪个 `(rank, token)` 的 partial）。它已不在仓库里，表中数据为当时实测留档。

留下的复现脚本是 `repro_475_combine_probe.py`（报告人原始探针，给 PASS/FAIL）和
`repro_475_launch_2node.sh`（两节点启动器，默认值已按 mi355-gpu-49 + mi355-gpu-51
配好，直接跑即可）：

```bash
./repro_475_launch_2node.sh                    # T=16..512，预期 PASS
COMBINE_IDX=orig ./repro_475_launch_2node.sh   # 判决实验：改传原始 idx
                                               # （在未打本修复的 mori 上才有区分度）
MORI_SRC=1 ./repro_475_launch_2node.sh         # 跑仓库内构建而非 site-packages 的 wheel
```

---

## 四、遗留

CI 覆盖缺口是这个 bug 能长期存活的直接原因，修复本身没有解决它：
`test_dispatch_combine_internode_v1.py` 只参数化 `world_size=8, gpu_per_node=8`
（走 RDMA 的 `gpu_per_node=4` 被注释掉），且 `max_num_inp_token_per_rank ≤ 128`——
跨节点 combine 路径从未被真正跑到。要防回归需要补一个真两节点的 combine 正确性用例。
