# MORI JIT v2

`python/mori/ops/dispatch_combine_v2/` 的 HIP kernel 后端：kernel 在 C++ 侧生成、编译、发射，
基于 cco-LSA，不依赖 mori-shmem。与既有的 FlyDSL 后端并列，同一个 op 类下可切换。

## 1. 要解决什么

debug-aa 的 kernel 有 69 个环境变量 gate、323 处 `#if`，缓存 key 在 `cache.py` 里人工拼接。

| 伤 | 表现 |
|---|---|
| 配置在带外 | gate 经 `-D` 进编译，不进 key |
| key 靠人工复述 | `NOQUANT` 漏进 key → A/B 复用满配二进制 → 读出"省 0us" |
| 常量有四份 | Python `_combine_shared_mem` / `_tunable_defines` / kernel `#if` / FlyDSL `_detect_wave_size` |
| 名字有两份 | `.hip` 宏拼符号名 + Python f-string 重建，无校验 |

这套设计消掉的是**这类 bug 的可能性**，不是它的实例：Cfg 值就是特化身份，渲染出的文本就是源码，
源码文本就是缓存 key。配置想进编译器，只能经过这段文本，而这段文本进了 sha256。

## 2. 分层

```
 Python                              │  C++
 ────────────────────────────────────┼──────────────────────────────────
 EpDispatchCombineOp（父类）         │
   ├─ …OpFlyDSL   backend="flydsl"   │
   └─ …OpHip      backend="hip"  ────┼─► Spec: ep_dispatch / ep_combine
   arena · scratch · 变体表 · _pick   │        │
                                     │        ▼
        ctypes：名值对 / 裸指针       │   Cfg 值 = 特化身份 = 缓存 key
        ──────────────────────────►  │        │ Render(cfg) → 源码文本
                                     │        ▼
                                     │   hipcc --genco → 内容寻址缓存
                                     │        │ .hsaco
                                     │        ▼
                                     │   GetFunction(Spec::EntryName(cfg))
                                     │   hipModuleLaunchKernel
```

Python 拥有 op 的**状态**（arena、scratch、变体表、routing handle）和调度。
C++ 拥有**一个 kernel 长什么样**（Cfg → 源码 → 编译 → 几何）。
Python 不算发射几何、不拼 kernel 名、不决定一个 Cfg 编成什么；`plan.info` 只能读回 C++ 的决定。

借鉴：**CK ck_tile** 的声明式实例模型、**DeepEP** 的内容寻址缓存与目录级原子发布、
**aiter opus_gemm** 的「绑好的 callable + 行为标志」选择器（§5）。
mori 特有的简化：走 `.hsaco` + `hipModuleGetFunction` 而非链接，所以入口名恒定，
不需要 DeepEP 的符号枚举，也不需要 CK 的链接期解析。

## 3. 一个 kernel 的三件套

| | 文件 | 内容 | 谁编译 |
|---|---|---|---|
| ① Cfg | `ep_cfg.hpp` | Cfg + Args + `VisitFields` + 共享几何 constexpr | **两侧**（HIP-free） |
| ② body | `ep_intranode_kernel.hpp`（portable）<br>`ep_intranode_1250x.hpp`（gfx125x TDM） | `template <EpCfg kCfg, typename T> __device__ void Body(EpArgs)` | 只有生成的 TU |
| ③ Spec | `ep_spec.hpp/.cpp` | Request + `RenderSource` + `Geometry` + 注册宏 | host |

新增一个 kernel 就是写这三样。**Python 侧零行**——Plan 类由 C++ 发布的 schema 生成。

### 3.1 Cfg 作为单一模板参数（C++20 structural NTTP）

```cpp
// debug-aa：位置绑定，9 个参数，调换两个照样编过、静默出错
template <typename T, bool UseP2PRead, bool EnableStdMoE, ...>
__device__ void EpCombineIntraNodeKernel_body(EpDispatchCombineArgs<T> args);

// v2：一个具名 struct 值
template <EpCfg kCfg, typename T>
__device__ void EpCombineBody(EpArgs args);
```

kernel 内所有 `#if MORI_COMB_XXX` 变成 `if constexpr (kCfg.xxx)`。差别不是语法糖：`#if` 的条件
来自命令行 `-D`（带外、不进 key、323 处散落），`if constexpr` 的条件来自 Cfg 字段（进 key、有守护、
IDE 能跳转）。DeepEP 和 CK 都用全 `int` 的位置展开，具名字段没有那个隐患。

### 3.2 字段列表的完整性是承重的

渲染器漏一个字段，生成的源码里就没有它，kernel 拿到默认值 → **静默跑错的 kernel**，比缓存失效严重。

```cpp
struct EpCfg {
  int worldSize = 8;  int hiddenDim = 7168;  int maxTokPerRank = 128;
  int numExpertPerRank = 8;  int numExpertPerToken = 8;
  int maxRecv = 0;              // 0 = worldSize*maxTokPerRank；同时是 flat index 的 stride
  EpDType dtype = EpDType::Bf16;
  int blockNum = 64;  int warpPerBlock = 16;  int waveSize = 64;   // 几何，host 算（§3.5）
  bool useWeights = true;
};

template <typename Self, typename Visit>
inline void VisitFields(Self& c, const EpCfg& d, Visit&& v) {
#define MORI_FIELD(x) v(#x, c.x, d.x)
  MORI_FIELD(worldSize); ... MORI_FIELD(useWeights);
#undef MORI_FIELD
}
MORI_JIT_ASSERT_FIELD_COUNT(EpCfg, 11, "加了字段就要同步 VisitFields");
```

加字段 = 三处编辑（struct、`MORI_FIELD`、计数），漏掉任何一处都是**编译期红字**。
`FieldCount<T>()` 用 `requires` 递归探测 aggregate 成员数，嵌套 struct 计为 1。

**一次遍历，四个消费者**：`Render`（缓存 key）、`Describe`（`info`）、request schema（发给 Python）、
`EpApplyFields`（请求 → struct）。所以加字段 Python 自动看得见。

`EpArgs` 的 wire schema 同理由一份 `MORI_EP_ARGS_FIELDS(X)` 生成，并按该顺序取 `offsetof` 断言递增。
只对 `sizeof` 设防是不够的：22 个字段里 8 个是裸指针，调换两个同类型字段所有尺寸校验都过，
kernel 安静地读错 buffer。

**只发非默认字段**：加一个默认不改行为的字段，已有实例的文本不变 → 缓存不失效。若新字段在默认值下
也改了 kernel 代码，文本虽没变但 include 哈希会变，照样重编。两级哈希各管一头。

### 3.3 生成的源码

```cpp
// mori jit v2 — generated, do not edit.
#include "src/ops/dispatch_combine_v2/ep_intranode_kernel.hpp"
using namespace mori::ops::v2;
constexpr EpCfg kCfg = EpCfg{.hiddenDim=2048, .maxTokPerRank=512, .maxRecv=4096};
using TokT = hip_bfloat16;
extern "C" __global__ void __launch_bounds__(EpBlockThreads(kCfg))
mori_ep_dispatch_bf16_ws8_h2048_k8_64x16(EpArgs args) { EpDispatchBody<kCfg, TokT>(args); }
```

- **入口名由 `Spec::EntryName(cfg)` 生成**，描述哪个 kernel、哪个 body、什么 dtype、形状和几何——
  否则 profile 里 dispatch、combine 和调优表为每个 token 档选出的每组几何全是同一行。
  要守的不变量不是「名字是常量」而是「**Python 不拼 kernel 名**」：名字在 C++ 里只写一处，
  渲染器把它写进源码、`Prepare` 把**同一个字符串**交给 `hipModuleGetFunction`，改名不可能漏改查找。
  名字里的字段都已在 Cfg 里，不给缓存 key 增加新维度。
- **文本即 key** → 配置不可能不进 key。
- **`__launch_bounds__` 是表达式不是数字** → 由 host/device 共用的同一个 constexpr 算出。
- **arch 路由在 host 侧**：include 哪个 body、调哪个函数，由 `RenderSource` 按
  `GetToolchain().arch` 决定——gfx125x 渲染 `ep_intranode_1250x.hpp` 的 TDM 版本，其余渲染 portable
  版。不是 device `#if`：渲染时看到的 arch 就是 `--offload-arch` 的 arch，两侧不可能不一致；而这个
  选择写在源码文本里，所以自动进 key，两种 body 各有自己的缓存条目。

**为什么生成不能放 Python**：Cfg 同时是 NTTP 类型，Python 渲染就要自己维护一份字段列表和默认值
→ 两张清单的问题原样回来。几何算术同理——device 侧也要调它。

### 3.4 缓存

```
~/.mori/jit/<arch>_<nic>/kernel.<name>.<hash>/{kernel.hip, kernel.hsaco}
hash = sha256(name $$ hipcc签名 $$ nic $$ flags $$ include哈希 $$ 源码文本)
```

发布无锁（抄 DeepEP）：编到 `tmp/<uuid>/` → 递归 fsync → 目录级 `rename`。抢输的删自己的、用赢家的。

`<arch>_<nic>` 只为人读，正确性不依赖它。**NIC 必须进 key**：intranode LSA 不链 NIC 相关的东西，
但设备侧 GDA 会——`libmori_cco_device.bc` 本身就按 arch+NIC 现编。NIC 是进程级事实，读
`MORI_DEVICE_NIC`（与 CMake、`python/mori/jit/config.py` 同一个权威），不在 JIT 层另做探测。

> **纪律：凡是以二进制形式进入编译的东西，哈希它的字节，不是它的路径。**
> `-mlink-builtin-bitcode=.../libmori_cco_device.bc` 的路径会随 flags 进 key，但那份 `.bc` 的
> 内容不会——而它自己就是 JIT 产物。GDA 落地时必须按 include 哈希同样的办法处理。

### 3.5 host/device 常量只有一份，且 host 不需要 hipcc

| 层 | 内容 | 谁编译 |
|---|---|---|
| Cfg + 共享算术 | `ep_cfg.hpp` | **两侧**：host `g++`，device `hipcc -std=c++20` |
| JIT 运行时 | `compiler`/`toolchain`/`spec`/`plan_api` | 普通 C++ 编译器（只用 `hip_runtime_api.h`） |
| kernel body | HIP intrinsic、`__global__` | 只有**生成的 TU** |

三条约束，违反了照样编过、只是悄悄把 host 拖进 hipcc，所以有 CI 守护
（`tools/jit_v2/check_host_device_split.sh`，已接入 ctest）：

1. Cfg 头**不得 include 任何 HIP 头**。dtype 用 `enum class EpDType` 标签，渲染时才展开成真实类型名。
2. 共享算术用**无属性 `constexpr`**，不写 `__host__ __device__`——后者会强制 host TU 走 hipcc。
   DeepEP 的 `TokenLayout` 是 `__device__ __host__` 的（它 host 侧本来就过 nvcc），这点不能照抄。
3. host 侧不用 C++20 指派初始化。

**C++20 的范围**：`FieldCount` 的探测必须用 requires-expression——C++17 下 gcc 和 clang 都把
"初始化器过多"当硬错误而非替换失败。所以持有 Cfg 的目标（`mori_jit`/`mori_ops_v2`/测试）按 C++20 编，
生成的设备 TU 无条件 C++20，mori 其余部分保持 C++17。**"host 不需要 hipcc"不受影响**。

```cpp
// ep_cfg.hpp —— 两侧共用
constexpr int EpBlockThreads(const EpCfg& c) { return c.warpPerBlock * c.waveSize; }
constexpr int EpMaxRecv(const EpCfg& c);            // 也是 flat index 的 stride
constexpr bool EpCfgIsValid(const EpCfg& c);

// ep_spec.cpp —— host 侧，arch 默认集中在这里
EpCfg MakeEpCfg(const std::string& arch, const EpRequest& req, EpKernelKind kind) {
  c.waveSize = mori::jit::v2::WaveSizeForArch(arch);    // gfx12* = 32，其余 64
  c.blockNum     = 64;                              // dispatch 搬运，combine 归约
  c.warpPerBlock = isDispatch ? 16 : 8;
  ...                                               // env 覆盖，唯一读环境变量的地方
}
```

### 3.6 环境变量

`MakeEpCfg` 末尾是**唯一**读 `MORI_V2_EP_*` 的地方，而且只做覆盖不做决定。覆盖后的 Cfg 自动进 key，
所以 A/B 不可能复用错的二进制——`NOQUANT` 那个 bug 在结构上不可能发生。

### 3.7 dtype 是传输属性，不是算术属性

dispatch 只**搬运**它的载荷（gfx9xx 是 `WarpCopy`，gfx125x 是一发 TDM tile），从不对元素做算术。
combine 要**归约**。这个不对称决定了 dtype 怎么建模：

```cpp
enum class EpDType : int { Bf16 = 0, Fp32 = 1, Byte8 = 2 };
```

`Byte8` 是传输类型，渲染成 `unsigned char`：fp8 直接用它，fp4 也用它、由调用方把 hiddenDim 减半
（2 个 e2m1 挤一个字节）。所以支持 fp8/fp4 dispatch **没有新增 kernel**，只是打通 dtype 这条链。
`MakeEpCfg` 在 combine 那条腿上拒绝 `Byte8`——它要求和，字节类型在那里能编过但会静默把字节加起来。

两条腿本来就是两个独立的 Plan，各带各的 dtype 和元素数，所以「非对称」（fp8/fp4 进、bf16 出）不是
一个特例路径，就是两个普通 kernel。arena 按 `token_nbytes` / `combine_token_nbytes` 分别定尺寸——
这是 fp4 唯一正确的算法。mori 在这里**不做量化**：载荷进来时就已经是打包好的。

## 4. op 层

op 在 Python，因为它持有的全是状态。C++ 侧没有对应的类。

```python
class EpDispatchCombineOp:          # dispatch_combine_op.py —— 后端无关 + 入口
    def __new__(cls, cfg, comm)     # 按 cfg.kernel_backend 分发到子类
    def dispatch(self, input, weights, scales, indices, *, routing=None, return_routing=False)
    def combine(self, input, weights=None, indices=None, *, routing)
    def _pick(self, num_tokens)     # 在**已编好的**变体里挑
```

三条时序约束，顺序不能换：

```
comm → arena → 每变体一个 Plan（编译）→ launch
                       ↑  graph capture 必须在这之后
```

- **arena 在编译前**：kernel 要 window handle，它只有 arena 建好才存在。
- **编译在 graph 捕获前**：捕获期不能 fork hipcc。所以 `Prepare`/`Launch` 是两个接口，
  而不是 DeepEP 那样每次 launch 都 generate+build 靠内存 map 兜。
- **变体表在 `Pick` 前编全**：`Pick` 只查表，关键路径上绝不触发编译。与 opus_gemm 那条
  「启发式能返回的 kid 必须在编译子集里」是同一个不变量，只是靠 map 查而非 codegen 期 assert。

捕获已实测（`tests/python/ops/dispatch_combine_v2/test_graph_capture.py`）：dispatch → identity
expert → combine 整步捕成一张图，replay 反复复现同一结果。能成立还差一个条件，而它在 kernel 里：
**跨设备 barrier 的 epoch 由设备侧自增**（dispatch 结尾 `atomicAdd(xdbFlag, 1)`）。若由 host 算好写
下去，捕获会把一个定值冻进图里，第一次之后每次 replay 要么空转要么白过——这正是那个测试要抓的东西，
另一个是 op 里混进 host 同步（`.item()`/`.cpu()`）会直接让捕获失败。

## 5. 两个后端共用一个 op

`EpDispatchCombineOp(cfg, comm)` 按 `cfg.kernel_backend`（或 `MORI_V2_KERNEL_BACKEND`，默认
`flydsl`）选出子类；也可直接实例化 `EpDispatchCombineOpHip`，`isinstance` 对两者都成立。

**父类拥有 op 本身**——`dispatch()`/`combine()`/`_pick`/视图/生命周期各只有一份。子类填三样：

```python
_regions(cfg)               # 这个后端要哪些 arena region
_build_kernels(cfg, arena)  # 编好的变体 + KernelSet
_unsupported(cfg)           # 它做不到什么
```

**行为差异是数据，不是 override。** `KernelSet` 上挂 `stages_in_kernel`（combine 的暂存拷贝在
kernel 内还是 host 侧）、`self_resets_counters`、`capabilities`/`unsupported`。形状抄自 aiter 的
`MOEMetadata`——但不抄它事后拆 `functools.partial` 反查身份那一手，那是把身份丢掉又猜回来。

唯一没法变成数据的是**调用约定**，所以它被吸收进 `_build_kernels`：返回的 callable 已经是父类的
具名签名，FlyDSL 的 11 个位置参数和 ctypes plan 的关键字参数各自在 `_wrap_*` 里适配。

**特性面是真包含不是交集**：HIP 后端做 gather combine，dispatch 收 bf16/fp32/fp8/fp4、combine 出
bf16/fp32（§3.7），其余一概不接：scatter、量化（`quant_type`，与直接传已量化载荷是两回事）、
StdMoE、per-token scales、routing replay、`local_expert_count`。能力门在**构造期**跑、且在分配 arena
之前——问它要 scatter 会当场报错并列出缺什么，不是发射时给出错误数字。

`_regions` 归子类还消掉一个陷阱：FlyDSL 的 `off_out_tok` 对 dispatch 指 `disp_out`、对 combine 指
`out_tok`，任何共享的 offset 传递路径都会静默接错 buffer。

**选后端只 import 那一个**：选 `hip` 不会拉进 flydsl，所以它在没装 FlyDSL 的机器上能跑。

### 5.1 几何调优表：每个后端一份，dispatch 和 combine 再分开

调优表也归后端。`hip_tuning_configs.py` 不 import FlyDSL 的 `tuning_configs`——不同 kernel、不同最优点，
借过来只会得到一个看起来合理的错答案。

表内再拆成两张：`_DISPATCH_TABLE` 带 dtype 键，`_COMBINE_TABLE` 不带。因为 **combine 根本不依赖
dispatch 的 dtype**——它只归约 bf16/fp32 的暂存区，不管前面搬来的是什么。合成一张表就必须给 combine
挂一个假的 dtype 轴，然后维护三份相同的答案。`lookup()` 把两张表合成 op 要的那条 schedule，能处理两边
桶边界不对齐的情况，所以任一半都能单独重调。

键是 `(world_size, hidden_dim, topk, experts_per_rank)`。实测结论直接编码在键里：**topk 会移动桶边界**
（它决定 `_tpi = warpSize/topk`，也就是一个几何在多少 token 内能一轮盖完），**专家数不会**（64 vs 96
在所有几何/dtype/topk 上都在 2% 以内），所以后者用 `None` 通配——那个通配是个结论，不是省事。

取值规则：**性能接近时取更小的几何**，因为少占 CU 在与专家 GEMM 重叠时是真收益。这类 3% 以内的取舍是
**策略不是测量**（单次 bench 偶尔会偏 20%）；真正由数据定的是桶的**边界**，它们来自 10~40% 的差异。

## 6. Python 绑定

C ABI 是**十个符号，对所有 kernel 永久有效**：

```
mori_jit_plan_create / _launch / _destroy / _info
mori_jit_plan_args_schema / _args_size / _request_schema
mori_jit_precompile / mori_jit_registered_plans / mori_jit_last_error
```

| | 机制 |
|---|---|
| **请求** | 以 `(name, value)` 对穿过边界。**没有结构体要声明两次**；未知字段名报错（拼错的旋钮悄悄不生效，正是测量描述错二进制的成因），缺失字段取 C++ 默认 |
| **参数** | C++ 发布 schema（字段表 + `sizeof`），Python **据此构造** ctypes 结构并断言大小。只有一份声明 |

通用绑定 `mori.jit.v2.plan_api`(即 libmori_jit.so 的 `mori_jit_*` C ABI 的 Python 侧,与
`src/jit/v2/plan_api.cpp` 对应)里没有任何 kernel 名字:`make_plan(kernel)` 由 schema 生成 Plan 类。
EP 专用的只有薄 shim `ops/dispatch_combine_v2/ep_plans.py`——它 `load_library("libmori_ops_v2.so")`
触发注册,再暴露 `EpDispatchPlan`/`EpCombinePlan`。约定两条:标为 `e` 的字段接受 dtype 名或 torch
dtype;launch 参数里 `off<Region>` 是 arena region 偏移,传 `arena=` 就绑定一次。

**为什么是 ctypes**：边界只传指针和标量，没有类型转换可做。pybind11 的代价是编译期（aiter 实测
libtorch+pybind11 的设备 pass 解析 ~15s）。绑定层里**没有 `import torch`**，靠鸭子类型取 `.data_ptr()`。

**交叉编译是进程级模式**：工具链只解析一次 arch，`arch=` 在那之前经 `SetArchOverride()` 落到一个
带锁的进程变量（不是 `setenv`——解析侧用 `getenv` 读，两线程并发建 plan 就是 POSIX 数据竞争）；
`MORI_JIT_ARCH` 仍作为静态覆盖有效。与已解析的不一致会**报错**，不是静默地"按 A 渲染、按 B 编译"。

## 7. 几个有证据的决定

**arena offsets 和 rank 是运行时参数，不是 Cfg 字段。**

| | dispatch 128/512 tok |
|---|---|
| 运行时参数 | 48.0 / 131.3 µs |
| 编译期常量 | 49.5 / 130.6 µs |

噪声以内，而代价实打实：进 Cfg 就进 key，于是每套 arena 布局、每个 rank 各编一份二进制——一次 8 卡
run 新增 16 份而不是 2 份，AOT 也无从谈起。旁证：gfx942 上 8 条载入的微基准测出朴素常量化是**负收益**
（VGPR 9→22，编译器丢掉"基址均匀"，把地址重新物化成向量形式；VGPR 通常才是占用率瓶颈）。

**`EpDType` 的枚举值必须与绑定的 dtype 表数值一致。** `e` 标签的字段以裸整数过边界，而
`mori.jit.v2.plan_api.DTYPES` 是一张表管所有 kernel 的所有枚举字段。独立编号 = 静默给错 kernel。

**AOT 预编译暂时没有。** 发射几何来自 Python 的调度表，所以 C++ 侧的预编译表永远渲染不出活的 op
会渲染的 Cfg。要预热缓存，就是构建期把 op 构造一次。

## 8. 性能

8×MI355 gfx950，EP8，hidden=7168 topk=8，eager，µs / 128·512·4096 tok/rank：

| | hip | flydsl |
|---|---|---|
| dispatch | **47.5** / 128.8 / 857.4 | 76.0 / **127.8** / 857.5 |
| combine | **57.8** / **124.4** / **781.8** | 82.7 / 133.6 / 888.8 |

4096 上 combine 快 12%，差在暂存拷贝：FlyDSL 在 kernel 前单独发一发 311MB 的 torch copy，
HIP 在 kernel 内做、与 barrier 等待重叠——104µs 变 ~1µs。

4×gfx1250（MI450）EP4，hidden=7168 topk=8，对照 v1 `tools/ep_test.sh` 的出厂默认，µs：

| tok/rank | 64 | 512 | 2048 | 4096 | 16384 |
|---|---|---|---|---|---|
| dispatch v1 → v2 | 80.9 → **73.1** | 82.8 → **74.7** | 104.6 → **97.4** | 171.4 → **152.0** | 584.5 → **519.6** |
| combine v1 → v2 | 26.9 → **21.3** | 41.7 → **36.9** | 96.7 → **83.8** | 169.6 → **146.4** | 588.7 → **523.0** |

每档都快（dispatch 7~12%，combine 11~23%）。同几何下的那部分来自 Cfg 是编译期 NTTP：ISA 里整数除法
dispatch 41→3、combine 57→10（`/maxRecv`、`%numExpertPerRank` 变成移位），combine 的 load 数 225→120
（peer 地址是 window 算术，不是穿过 `SymmMemObj` 的三次相关联 load）。其余来自调优表。

窄 dtype 的收益更大：4096 档 dispatch bf16 156.6 → fp8 98.5 → fp4 80.0，16384 档 fp4 是 bf16 的约 3 倍快。

单实例编译 1.6s，对照 debug-aa 整文件 115s（90 个 kernel，gfx1250）。这个比值才是按需编译成立的前提。

## 9. 移植来源与已知缺口

kernel 从 `origin/main` 的 intranode MoE kernel 移植（bf16 gather 路径）。移植是机械的，因为 v1 的
**整个**对称内存面只有两样：`SymmMemObjPtr::GetAs<T*>(pe)`（30 处，两次相关联的 load 查表）和三个
不带任何 shmem 状态的自旋等待。于是

```
memObj->GetAs<T*>(pe)  →  ccoGetLsaPeerPtr(win, pe, args.offRegion)
```

13 个 `SymmMemObjPtr` 塌成一个 window handle + 8 个 offset。生成的 TU 不 include `mori/shmem`，
所以没有设备全局变量、不需要 per-module init。

gfx125x 的 TDM body 同样是机械移植（`intranode_1250x.hpp` → `ep_intranode_1250x.hpp`）：v1 的 cco 路径
本来就把每块对称 buffer 从同一个 LSA window 里 bump 分配，和 SymmArena 是同一个内存模型，源码里的
`shmem` 命名空间只是那层抽象加几个与后端无关的自旋。TDM 那套机器（`amd_gfx1250_TDM.h` builtin、
`_cusplit_*` 设备全局暂存池）原样保留。范围收到 hip 后端能服务的部分：非量化的 TDM dispatch +
PULL/QUAD gather combine。

**HIP 后端未实现**（由能力门在构造期拒绝）：scatter combine、`quant_type` 量化、StdMoE、per-token
scales 转发、routing replay、`local_expert_count`。这些在 FlyDSL 后端都有。注意 fp8/fp4 **dispatch**
已经支持（§3.7）——那是搬运已量化的载荷，和让 mori 自己量化的 `quant_type` 是两回事。

**cross-node 没有**：v2 只有 intranode 两个 body，op 里也没有 `kernel_type` 的概念。

**debug-aa 的 69 个 gate 尚未归置**。归置的形状是三分：调优参数升格为 Cfg 字段；诊断开关进一个默认
全 false 的 `Diag` 子结构（只发非默认字段，所以正常运行的源码里根本不会出现它，一开诊断自动是另一个
缓存条目）；已有定论的删掉代码、结论留在注释里。
