# EPv2 TUNING 常见问题
本文回答五个问题：

1. **调优方法和 V1 有什么不同？** V1 像擂台赛：每个候选测一次，谁快谁当擂主；V2 像守门：表里现有的配置一直守着，每个候选都要和它交替比多次、明显赢了，再单独确认一次才写进表（[第 1 节](#1-调优方法和-v1-有什么不同)）。
2. **运行时怎么选 kernel？** V1 默认不查表，打开 `AUTO` 后每次调用现查；V2 在建 op 时就把每一档的配置定好、编译好，每次调用只按 token 数挑一档（[第 2 节](#2-运行时怎么选-kernel和-v1-有什么不同)）。
3. **表里没有对应规则时会怎样？** V2 宁可用默认值，也不借别的形状、别的型号的规则；V1 会尽量找一条"差不多"的来用（[第 3 节](#3-表里没有对应规则时会怎样v1-又是怎样)）。
4. **默认值和接口有哪些变化？** `num_qp_per_pe` 默认从 2 改成 8。这个参数原来既管分配多少条 QP、也管实际用几条发，现在只管分配；AsyncLL 仍然用满所有分配的 QP，所以默认从 2 条悄悄变成了 8 条；新增 `active_qps` 等参数；`internode_tuning_configs.lookup()` 破坏了兼容（[第 4 节](#4-默认值和接口有哪些变化)）。
5. **PR 里夹带的 combine 修复是什么？** V2LL 的 combine 可能把某个节点的 token 当成"没有"跳过，输出悄悄出错。要同时满足满容量、跨节点、dispatch 和 combine 紧挨着跑才会触发。调优正好全都满足，而 CI 的每个用例都至少缺一个（[第 5 节](#5-pr-里夹带的-combine-修复)）。

## 先约定几个词

- **几何**：一次 kernel 启动的 `block_num`（开多少个 block）、`rdma_block_num`（其中多少个负责跨节点 RDMA）、`warp_per_block`，下文写成 80/48/8 这样的三元组，也简称 B/R/W。
- **档**：表里每一行有一个 token 上限 `num_tokens`。一次调用有 T 个 token，就用"上限 ≥ T 的第一行"；T 比最大的上限还大，就用最后一行。
- **active QP**：每对 GPU 之间分配了若干条 RDMA 队列（`num_qp_per_pe`，默认 8）；每一档实际用几条发数据（1、2 或 4），由表里的 `active_qps` 决定。
- **D+C**：一次 dispatch 加一次 combine 的总延迟。
- **容量**：`max_num_inp_token_per_rank`，一个 rank 一次最多发多少 token，buffer 按它分配。

调优相关：

- **rep**：候选和基准各跑一遍，算一次 rep。每遍是 20 轮 warmup 加 30 轮计时。
- **配对差**：同一次 rep 里，候选减基准。两边几乎同时测，网络整体变快变慢会被抵掉。
- **margin**：候选至少要快这么多才算赢，取 max(1.5 µs, 基准 × 2%)。
- **整表 A/B**：拿新表和现有表各建一个 op，交替跑几轮，逐档比 D+C。
- **逐档合并**：整表 A/B 之后，只把确实变快的档换成新规则，其他档保留原规则。

## 1. 调优方法和 V1 有什么不同

**一句话**：V1 像擂台赛，每个候选只打一局，谁比擂主快谁就当擂主；V2 像守门，表里现有的配置 A 一直是守门员，每个候选都和 A 打几局、看中位数，打赢的再和 A 加赛确认，才写进表。

先看 1.1 的例子，理解两种思路的区别；再看 1.3，分清 V2 的两套调法。其余小节是细节，按需查。

### 1.1 先看一个例子

调 128 token 这一档的 dispatch。A 是表里现有的配置，B、C、D 是新候选。下面的数字是编的；V1 比的是 dispatch 自己的延迟，V2 默认比 D+C（combine 不变），所以两边的数字量级不同，不能互相比。

**V1：基准是"目前见过最快的"，谁赢就换成谁。** V1 调优时不读表，A 能参加比较只是因为它恰好也在候选网格里。每个配置只跑一次，按扫描顺序比：

```text
A  131.6us   → 第一个，当前最好 = A
B  129.9us   → 比 A 快      → 当前最好 = B
C  130.5us   → 比 B 慢      → 不要
D  129.0us   → 比 B 快      → 当前最好 = D
扫完 → 存 D
```

每个只测一次。B 比 A 快的 1.7 µs，可能只是 B 运行时网络正好闲；一旦 B 靠运气当了基准，后面的 C、D 都在和一个靠运气得来的好成绩比。另外，dispatch 和 combine 在同一轮扫描里各挑各的最快，最后可能 dispatch 用 D 的配置、combine 用 B 的配置，而这个组合从来没一起跑过。

**V2：基准一直是 A，每个候选都和 A 交替比 3 次。** 这里按单档扫描（1.5）的规则演示，整表流程（1.4）的思路相同，只是候选换成整张表。每次 rep 先跑 A、再跑候选，相减得到一个配对差，取 3 个配对差的中位数。要快过 margin = max(1.5 µs, 2% × 327 µs) = 6.5 µs 才算赢：

```text
        rep1          rep2          rep3          配对差           中位数   判定
B vs A  319 vs 327    332 vs 341    322 vs 325    -8, -9, -3       -8      赢（-8 < -6.5）
C vs A  320 vs 326    333 vs 330    324 vs 328    -6, +3, -4       -4      不算（没过 6.5）
D vs A  317 vs 327    331 vs 340.5  314 vs 325    -10, -9.5, -11   -10     赢
```

1. B、C、D 都只和 A 比。B 赢了，基准也不会换成 B。
2. rep2 时网络整体变慢（A 从 327 变成 341），B 也跟着慢，两边一减还是 −9。配对相减把这种整体漂移抵掉了。
3. 扫完，赢家里 D 比 A 快得最多，选 D。B 和 D 从没直接比过，差 2 µs 可能只是噪声。
4. 只拿 D 和 A 再比 15 次，中位数还赢，才写进表。下次调这一档时，D 就是新的 A。

### 1.2 V1 怎么调

入口是 `tools/run_all_internode_tuning.sh`，它调用 `examples/ops/dispatch_combine/test_dispatch_combine_internode.py --cmd tuning`。

1. 每个 token 数单独跑一次。候选是一个网格：block ∈ {8, 16, 32, 64, …, CU 数}，warp ∈ {4, 6, 8, 12, 16}，rdma ∈ {block/4, block/2, 2·block/3}。在 MI308X（80 个 CU）上是 5 × 5 × 3 = 75 个候选。
2. 在同一个 op 上，把每个候选的几何当参数传进去，测一次：20 轮 warmup，30 轮计时，去掉第 1 轮，取平均延迟。
3. dispatch 和 combine 用同一个候选一起跑，但各自挑平均延迟最低的，所以存下来的两相规则可能来自不同候选。门槛默认是 0：只要比当前最好的快一点点，就换上去。表里原来的规则不参与比较。
4. 扫描开始前，用默认几何校验一次结果；之后每个候选只计时、不校验。存表之前要另外跑 `tools/verify_tuned_config.sh`。

### 1.3 V2 有两套调法，先分清

两套都是"守门"：基准一直是现有配置，候选必须稳定地赢它才算数。区别在一次调多大范围、在什么条件下测。

|                    | 单档扫描                                           | 整表流程                                       |
| ------------------ | -------------------------------------------------- | ---------------------------------------------- |
| 在哪               | 仓库里，`test_dispatch_combine_v2_internode.py`    | 仓库外的实验脚本，没有进仓库                   |
| 谁用               | 贡献者调新形状；本 PR 里少数 MI300X EP8 规则       | 本 PR 的 MI308X、MI355X 跨节点表               |
| 一次调什么         | 一个档的一相几何，或两相 QP 数                     | 一个形状的整张表                               |
| 候选长什么样       | 单个配置，用环境变量注入，op 只有一档              | 整张表，op 照常读表、选档                      |
| 和谁比             | 表里这一档现有的规则                               | 现有的整张表                                   |
| 怎么算赢           | 3 次配对差的中位数快过 margin                      | 3 轮合计快 1% 以上，并且每轮都更快             |
| 能看到跨档影响吗   | 不能（看不到 QP 收尾把其他档拖慢）                 | 能                                             |

**本 PR 的 MI308X、MI355X 跨节点表来自整表流程，没有用单档扫描。** 少数 MI300X EP8 规则（256/6、264/9、384/6 h5120）是用单档扫描调的，tuning README 的 Measurement conditions 一节有说明。单档扫描的测量条件和线上不同：默认 `scale_dim=32`，候选和基准两个 op 同时驻留，结果取平均。用它选出的候选，很多放到整表里对比都复现不了。所以单档扫描适合用来找候选，要上线还得用整表流程确认。

### 1.4 整表流程：本 PR 的表是怎么来的

以一个形状、一个 kernel 族为单位：

1. **几何**：每张候选表只在现有表的基础上改一处，例如把这个形状所有档的 dispatch 几何都换成 X。第一轮测全部候选；第二轮只留下在某一档上离这一档最好成绩不超过 3% 的候选。
2. **QP 和 load-first**：几何定下来以后，再选每一档两相的 QP 数，以及 combine 开不开 load-first。
   - 测一组整表变体（普通 V2 17 张，LL 13 张，见 1.9）。每张变体把所有规则的 QP 数都设成同一组 (dispatch, combine)，交替跑 2 轮。
   - 根据结果估计每一档在各种 QP 组合下的 D+C，逐档挑最好的组合。
   - 几种 op 共用同一条 combine 规则时要一起挑。比如 bf16、fp8、fp4 dispatch 共用 bf16 的 combine 规则，挑的时候要看三者合起来的效果。
   - 要换掉默认的 (2, 2)，必须快 1% 以上，并且每轮都更快。
3. **整表 A/B**：新表和现有表交替跑 3 轮。
4. **逐档合并**：只有新规则的 D+C 比原规则快 1% 以上，而且 3 轮里每一轮都更快，这一档才换；没有收益的改动回退。
5. **确认 A/B**：合并后的表再和原表交替跑 3 轮，表里写的指标取自这一次。
6. **其他措施**：每个测量点计时前都做恒等 expert 的正确性校验。某次运行的延迟超过这个测量点所有运行中位数的 2 倍，视为卡顿，剔除重测。均值/中位数、最大值/中位数、三遍之间的差异都有门限。机器上一出现别人的 GPU 作业就暂停，受影响的数据作废重测。

### 1.5 单档扫描：给贡献者调新形状

代码在 `test_dispatch_combine_v2_internode.py` 的 `_tune()`。它和 1.4 的整表流程是两套独立的工具；本 PR 的跨节点表没有用它。一次运行只调**一个 token 档**：要么调**一相**的几何，要么调两相 QP 数的搭配。

1. **生成候选。** block 取 {8, 16}、2 的幂和 CU 数，MI308X 上是 {8, 16, 32, 64, 80}；warp 在 quick 档取 {4, 8, 16}，full 档取 {4, 6, 8, 12, 16}；rdma 按 block 的比例取，quick 档 1/2 和 2/3，full 档 1/8 到 3/4 之间七个点。quick 档共 5 × 3 × 2 = 30 个，和现有规则相同的那个去掉。
2. **基准取现有的表。** 用 `lookup_internode()` 读当前规则。加了 `--tuning-save` 或 `--tuning-config-dir` 时，会先把 `MORI_EP_V2_TUNING_DIR` 指到将要写入的表目录，保证读的和写的是同一张 CSV。因此先调 dispatch 并存表，再调 combine 时，基准里的 dispatch 已经是新规则。
3. **每个候选单独建一个 op。** 几何通过 `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM`（`B,R,W[,active_qps]`）注入，这两个变量会让 `internode_schedule()` 完全绕过表，op 只有一档。只改正在调的那一相，另一相固定用现有规则，因为两相互相影响。
4. **交替配对测量。** 每次 rep 先跑一遍基准、再跑一遍候选（默认 3 次 rep）。每遍 20 轮 warmup、30 轮计时、去掉第 1 轮，得到所有 rank 的平均延迟和跨 rank 最差的一轮。默认比 D+C。
5. **判定。** 每次 rep 算一个配对差（候选 − 基准），取中位数：
   - margin = max(1.5 µs, 基准中位数 × 2%)，tail_room = 基准中位数 × 10%。
   - **最差轮**：每遍里最慢的那一轮，再对各次 rep 取中位数。用来挡住平时更快、但偶尔卡一下的候选。
   - **WIN**：中位配对差 < −margin，并且候选的最差轮比基准慢不超过 tail_room。
   - **TAIL**：中位配对差在 ±margin 以内，但候选的最差轮比基准快 tail_room 以上。
   - 用配对差而不是两边中位数相减，是为了抵掉测量环境的漂移，就是例子里 rep2 的情况。实测中同一个基准前后读到过 84.9 µs 和 126.0 µs。
6. **不做贪心。** 就是例子里的「基准一直是 A」。加 `--tuning-greedy` 可以改成 V1 那样赢了就换基准，但不推荐：同一组 29 个候选这样扫 5 次，选出了 5 个不同的赢家。扫完后，WIN 排在 TAIL 前面，各组内按快多少排序，取第一名。
7. **保存。** 加 `--tuning-save` 时，赢家单独建 op 再测一遍，表里的指标取自这一次；然后 `save_internode_result()` 用它替换选择键（CSV 前 8 列）相同的那一行。

几个容易错的参数：

- 测试脚本的 `--num-qp` 默认是 1（op 本身的 `num_qp_per_pe` 默认是 8）。不改的话，表里写的 QP 数会被截到 1，基准就和线上配置不一样。MI308X 的表是按 8 测的。
- `--scale-dim` 默认是 32，表是按 0 测的。
- 单档扫描本身**不校验结果**：`_verify_once` 只在 `--cmd bench` 里调用。1.4 整表流程里的"每个测量点都校验"是仓库外的脚本做的。
- 扫描时 combine 的 load-first 是关的，重新保存一条 combine 规则会把 `combine_load_first` 清掉。存完要打开 load-first 再单独 A/B，赢了才把它加回去。

#### 用例：调 MI308X EP16 上 DeepSeek-V3 的 LL ≤128 档

现在 `gfx942_mi308x_InterNodeV2LL_ep16.csv` 里这一档（256/8、bf16、hidden 7168）是：

| 相       | 几何     | active QP | 延迟     |
| -------- | -------- | --------- | -------- |
| dispatch | 64/32/16 | 1         | 148.1 µs |
| combine  | 16/10/16 | 2         | 178.8 µs |

**1. 扫 dispatch 几何。** 两台机器各起一个 torchrun 进程，脚本在每台机器上自己起 8 个 worker：

```bash
torchrun --nnodes=2 --node_rank=$R --nproc_per_node=1 --master_addr=$IP --master_port=29500 \
  tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py \
  --cmd tuning --kernel-type v2_ll --max-tokens 128 --num-qp 8 --scale-dim 0 \
  --tuning-phase dispatch --tuning-scope quick --tuning-reps 3
```

64/32/16 已经在 30 个候选里，去掉后剩 29 个。每个候选要测 3 × 2 遍。

**2. 读结果。** 下面的数字是编的，只用来演示判定规则：

```text
#   [17/29] (80, 53, 8, 1) med= 315.2us vs incumbent (64, 32, 16, 1) med= 327.0us  paired=  -7.8us [-9.1,-3.0] worst 360/352  WIN
#   [18/29] (80, 53, 16, 1) med= 323.0us vs incumbent (64, 32, 16, 1) med= 327.0us  paired=  -4.0us [-12.0,+2.0] worst 349/352  --
```

基准中位数 327 µs，所以 margin = 6.5 µs，tail_room = 32.7 µs。

- 第 17 个：中位配对差 −7.8 µs，超过了 margin（6.5 µs）；最差轮只慢 8 µs，在 32.7 µs 以内。判 WIN。
- 第 18 个：有一次快了 12 µs，但中位数只有 −4.0 µs，不到 margin。不算赢，一次好运不会被写进表。

**3. 加长验证并存表。** 只拿赢家和现有规则比，rep 加到 15 次。加了 `--tuning-save` 后，赢了就把赢家写进表；没赢就保留原来的几何，只更新指标列：

```bash
... --tuning-phase dispatch --tuning-candidate 80,53,8 --tuning-reps 15 --tuning-save
```

**4. 调 combine。** 用 `--tuning-phase combine` 把 1–3 步再走一遍。这时 combine 的基准里，dispatch 已经是第 3 步写进去的新规则。

**5. 成对选 QP 数。** 两相几何都已经在表里，固定不动；两相的 QP 数交叉成 9 对，按 D+C 比：

```bash
... --tuning-phase pair --tuning-qps 1,2,4 --tuning-reps 5 --tuning-save
```

**6. 要上线，还得按整表流程确认。** 单档扫描看不到跨档影响，测量条件也和线上不同（见 1.3）。它选出的结果，最好再用 1.4 的整表 A/B 确认一遍，再提交。

### 1.6 V1 和 V2 对比

|                | V1                   | V2（本 PR 的表，整表流程）                           |
| -------------- | -------------------- | ---------------------------------------------------- |
| 调哪些参数     | 几何                 | 几何、每档每相的 active QP、V2 combine 的 load-first |
| 一个候选测几次 | 1 次                 | 筛选 2 轮；选中的再做 A/B 3 轮、确认 3 轮            |
| 和谁比         | 本次扫描里最好的     | 现在的表                                             |
| 采用条件       | 更快就采用（门槛 0） | 3 轮合计快 1% 以上，并且每一轮都更快                 |
| 两相怎么选     | 各自挑最快的         | 看 D+C，两相 QP 数成对选                             |
| 正确性         | 扫描前查一次，候选不查，事后另跑 | 每个测量点计时前都查                     |

### 1.7 为什么 V2 要这么麻烦

1. **噪声大得多，挑最快的会挑到最幸运的。** 假设 75 个候选的真实速度其实都一样，每次测量有 ±3% 的随机误差。75 个里测得最快的那个，几乎一定比它的真实值快好几个百分点。这不是因为它真快，而是 75 次里总有一次运气好。存进表、上线以后，它回到真实水平，"优势"就没了。上面例子里 B 靠一次 1.7 µs 当上擂主，就是这种情况。
   - V1 走 shmem，同一配置前后测几次只差几个百分点，候选之间的真实差距往往比这大，所以挑最快的基本不会挑错。
   - V2 走 CCO/GDA，同一个几何前后两次测出过 86 µs 和 137 µs。一次扫描"找到"的 5% 优势（64/32/4 对比现有的 64/32/8），换成交替测四对之后就没了。
   - 所以 V2 要靠配对相减抵掉漂移、靠中位数和 margin 挡住一次好运，挑出赢家后还要**重新测一次**：一个候选如果只是那几次运气好，换一批新的测量就会露馅。
2. **两相互相影响。** dispatch 结束前，还要把上一次 combine 在其他 QP 上留下的完成事件处理掉。所以 dispatch 用 2 个 QP、combine 用 4 个时，dispatch 反而比两相都用 2 个或都用 4 个更慢。两相各挑各的最优，可能挑出一对不配套的。
3. **一档的选择会影响其他档。** dispatch 按整张表里最大的 QP 数收尾，某一档改成 4 个 QP，其他档的 dispatch 也要多付一点（MI308X 上 T ≤ 64 时约 1–2 µs）。只看单档会漏掉这种影响，所以要整表 A/B。单档扫描看不到这种影响：它用环境变量注入候选几何，注入后 op 只有一档（见 1.5 第 3 步）。
4. **kernel 是 JIT 编译的。** V2 的几何、QP 数、load-first 都是编译期常量，换一个候选就是一套新 kernel；V1 的 kernel 是预编译的，几何只是启动参数。所以 V2 每换一个候选都要重新建 op、重新编译；要测到上线后的真实效果，候选就得写成整张表，走真实的查表路径。

### 1.8 哪种更合理

看噪声和候选之间的真实差距哪个大：

| 场景                                          | 选哪个                                                   |
| --------------------------------------------- | -------------------------------------------------------- |
| 噪声小（几个百分点），候选之间差距大          | V1 够用，而且快                                          |
| 噪声和差距是同一量级，比如 V2 的 CCO/GDA 路径 | 必须用 V2 的做法                                         |
| 要写进仓库、上线的表                          | 用 V2 的做法，至少要有"挑出来以后重新测一次再确认"这一步 |

V2 的代价是慢，而且偏保守：真实只快 2% 以内的改进会被 margin 挡掉，表里继续用 A。换来的是不会把噪声写进表，对线上的表来说这个取舍是对的。

V1 想低成本改进的话，最值得补两点：用 `MORI_EP_TUNING_MARGIN` 设一个门槛，不再快一点就换；扫完把前几名和表里现有的规则交替重测几次，再决定存不存。

### 1.9 一共测了哪些组合

分三层：表要覆盖哪些形状和档，单档扫描在一档里扫哪些候选，整表流程测了哪些整表变体。

#### 表要覆盖的组合（MI308X EP16）

InterNodeV2 和 InterNodeV2LL 两张表覆盖的组合相同，每张 233 条规则：dispatch 142 条，combine 91 条。

| 模型                                   | 专家/topk | hidden | dispatch dtype                       | combine dtype | 档  |
| -------------------------------------- | --------- | ------ | ------------------------------------ | ------------- | --- |
| DeepSeek-V3/R1                         | 256/8     | 7168   | bf16、fp8、fp4（按打包宽度 3584 存） | bf16、fp8     | 11  |
| GLM-5                                  | 256/8     | 6144   | bf16、fp8                            | bf16、fp8     | 11  |
| DeepSeek-V3/R1，shared expert 融进路由 | 272/9     | 7168   | bf16、fp8                            | bf16          | 11  |
| DeepSeek-V4-Pro                        | 384/6     | 7168   | bf16、fp8                            | bf16          | 7   |
| DeepSeek-V4.1-Flash                    | 384/6     | 5120   | bf16、fp8                            | bf16          | 11  |
| Kimi K3                                | 896/16    | 3584   | bf16、fp8                            | bf16          | 11  |
| 无已知模型                             | 256/6     | 7168   | bf16                                 | bf16          | 7   |

- 11 档是 4、8、16、32、64、128、256、512、1024、2048、4096；7 档是 64 到 4096。
- fp8 指 `fp8_e4m3_fnuz`。fp8 和 fp4 的 dispatch 规则，是配着同一档的 bf16 combine 规则测的。
- fp8 combine 规则只给两相都用 fp8 的场景，没有写 active QP，用的是更早按 2 个 QP 测的数据。
- 每条规则都带自己的 active QP（1、2 或 4），fp8 combine 除外。load-first 只有普通 V2 的 combine 规则有。
- 测量条件：容量等于这一档的 token 数，均匀路由，`scale_dim=0`。272/9 用 replica 路由：8 个路由专家，加上本 rank 上的 shared expert 副本。

#### 每一档里扫哪些候选（单档扫描）

| 模式          | 参数                                                                                      | 候选数             | 变的                      | 不变的                         |
| ------------- | ----------------------------------------------------------------------------------------- | ------------------ | ------------------------- | ------------------------------ |
| 扫一相几何    | `--tuning-phase dispatch` 或 `combine`                                                    | quick 30，full 165 | 这一相的 B/R/W            | 这一相的 QP 数；另一相整条规则 |
| 几何 × QP     | 再加 `--tuning-qps 1,2,4`                                                                 | quick 90，full 495 | 这一相的 B/R/W 和 QP 数   | 另一相整条规则                 |
| 只扫一相的 QP | `--tuning-phase dispatch` 或 `combine`，加 `--tuning-geometry shipped --tuning-qps 1,2,4` | 3                  | 这一相的 QP 数            | 两相几何                       |
| 两相 QP 成对  | `--tuning-phase pair --tuning-qps 1,2,4`                                                  | 9                  | (dispatch QP, combine QP) | 两相几何                       |
| 验证一个几何  | `--tuning-candidate B,R,W[,QP]`                                                           | 1                  | 指定的那个                | 另一相整条规则                 |
| 验证几对 QP   | `--tuning-phase pair --tuning-candidate 1,4/2,2`                                          | 指定几对           | 指定的 QP 对              | 两相几何                       |

- full 是 165 而不是 5 × 7 × 5 = 175：block 为 8 和 16 时，7 个 rdma 比例取整后有重复，各只剩 6 个。
- 和现有规则完全相同的候选会被去掉，所以实际数量可能少 1。
- 每个候选测 `reps × 2` 遍（候选、现有规则各一遍），每遍 20 轮 warmup + 30 轮计时。quick 档扫一相、默认 3 次 rep，就是 30 × 3 × 2 = 180 遍，约 9000 次 dispatch+combine。

#### 整表流程测了哪些整表变体

流程见 1.4，这里只列 QP 和 load-first 那一步的变体，以一个形状、一个 kernel 族为单位：

- dispatch 的 QP 数取 1、2、4，combine 取 1、2、4、8，共 12 种组合，再加两相都用 8，共 13 张。
- 普通 V2 另有 4 张"两相用同一个 QP 数、combine 开 load-first"的变体，所以 V2 一共 17 张。
- 每个测量点都是 3 遍 ×（20 轮 warmup + 30 轮计时），每遍去掉第 1 个计时轮，计时前先做正确性校验。

## 2. 运行时怎么选 kernel，和 V1 有什么不同

**一句话**：V1 默认根本不查表，设成 `AUTO` 后每次调用现查一条，把几何当参数传给预编译 kernel；V2 在建 op 时就把每一档的配置定好、编译好，每次调用只是按 token 数挑一档。

### 2.1 V2 概览（HIP 后端）

1. **建 op 时（只做一次）**
   - 按本机型号读对应的 CSV，例如 `gfx942_mi308x_InterNodeV2_ep16.csv`。
   - 只用和这个 op 完全一致的行：总专家数、topk、hidden、dtype（两相各看各的）。FP4 dispatch 按打包后的宽度匹配；combine 还要求 `zero_copy=0`、`quant_type=none`。
   - 得到一串档，每档定好 dispatch 和 combine 的几何、active QP 和 load-first。
   - 默认的 `internode_kernel="auto"` 会同时读 V2 和 V2LL 两张表：T ≤ 512 的档用 LL，大于 512 的档用普通 V2（切换点是 `internode_auto_ll_max_tokens`）。指定 `"v2"` 或 `"v2_ll"` 就只用那一族。
   - 把每一档要用的 kernel 都编译好。rank 0 会以 INFO 级别打印整张调度表：读到了哪个文件、每档用什么。
2. **每次调用**：按这次的 token 数挑档，直接启动编好的 kernel，不查表，也不编译。`op.set_active_qps(n)` 可以临时改 QP 数，但只能改成建 op 时预编译过的值（`active_qp_counts`）。
3. **单机（EP8）**：先查 IntraNodeV2 的 CSV；某一相没有行，就用 `hip_tuning_configs.py` 里的 Python 表；还没有就用默认值。
4. **注意后端**：CSV 表只给 HIP 后端用。跨节点只有 HIP 后端，必须指定 `kernel_backend="hip"`（或 `MORI_V2_KERNEL_BACKEND=hip`）。单机在 gfx942 和 gfx950 上默认是 FlyDSL 后端，它有自己的表。

### 2.2 V2 建 op 的完整调用链

先给出从用户代码到每次调用的整条链，再分段讲。贯穿始终的是 `KernelConfig`（下文也叫 spec）：它是调度表每一档的内容，是编译好的 kernel 字典的键，也是每次调用 `_pick()` 返回的东西。

文件都在 `python/mori/ops/dispatch_combine_v2/` 下。

```text
① 构造 config：跨节点不读表
EpDispatchCombineConfig.__post_init__                      dispatch_combine_op.py
 ├─ _pinned_geometry = 调用方显式设过的几何字段
 └─ _resolve_geometry()
      跨节点：只填兜底默认值 96/64/8，return
      单机：没设几何字段 → 按后端查表；设了 → 其余字段填默认值

② 构造 op：选后端 → 读表 → 调度表 → 编译
EpDispatchCombineOp.__new__                                dispatch_combine_op.py
 └─ _resolve_backend(name) → EpDispatchCombineOpHip
EpDispatchCombineOpHip.__init__                            hip_backend.py
 ├─ self._schedule = resolve_schedule(cfg)                 hip_tuning_configs.py
 │    └─ internode_schedule(cfg)
 │         ├─ _geometry_from_env()          设了 MORI_EP_DISP_GEOM / COMB_GEOM 就直接返回一档
 │         ├─ internode_kernel_families()   auto → ("v2", "v2_ll")
 │         ├─ internode_buckets() × 每个族
 │         │    └─ _v2_rules() → read_v2_table()       拼文件路径、读 CSV、按路径缓存
 │         └─ 按边界拼两个族、补默认值、套用调用方设的字段、定 QP 数、合并相同档
 │    rows → ScheduleEntry(max_tokens, dispatch=KernelConfig, combine=KernelConfig)
 ├─ describe_schedule()                    rank 0 打印 INFO
 ├─ _dispatch_specs / _combine_specs       去重后的 KernelConfig
 └─ _build() → self._kernels = _build_kernels() → _build_internode_kernels()
      每个 spec 展开成几个 pass，逐个 JIT 编译；kernels[phase][spec] = run

③ 每次调用：只查表，不读文件，不编译
op.dispatch(x)  → _dispatch_impl: n = x.shape[0]
                  → _pick(n) → kernel_config(n) → pick_bucket(self._schedule, n) → spec
                  → kernels["dispatch"][spec](...)
op.combine(y)   → _combine_impl: ct = routing.cur_rank_num_token（对应 dispatch 的源 token 数）
                  → 同上，取 combine 的 spec，所以和它的 dispatch 一定在同一档
```

#### 选后端：`__init_subclass__` 注册表

```python
class EpDispatchCombineOp:
    _BACKENDS: dict[str, type] = {}
    def __init_subclass__(cls, backend=None, **kw):
        super().__init_subclass__(**kw)
        if backend:
            cls.backend_name = backend
            EpDispatchCombineOp._BACKENDS[backend] = cls

class EpDispatchCombineOpHip(EpDispatchCombineOp, backend="hip"): ...        # hip_backend.py
class EpDispatchCombineOpFlyDSL(EpDispatchCombineOp, backend="flydsl"): ...  # flydsl_backend.py
```

- 子类**定义时**，Python 自动调用基类的 `__init_subclass__`，把自己登记进 `_BACKENDS`。
- 用户写 `EpDispatchCombineOp(cfg, comm)` 时，`__new__` 按 `cfg.kernel_backend` > `MORI_V2_KERNEL_BACKEND` > 默认值（gfx125x 用 hip，其余用 flydsl）选出名字。名字还没登记，就按 `_BACKEND_MODULES` 导入对应模块，导入时触发登记。然后返回子类实例，Python 接着调用**子类的** `__init__`。
- 只有通过基类构造才会分派；直接写 `EpDispatchCombineOpHip(cfg, comm)` 会跳过这一步。
- FlyDSL 遇到跨节点 config 会直接报错，所以跨节点必须用 hip。

#### 构造 config：跨节点只填兜底值

`__post_init__` 先记下调用方设过哪些几何字段（`_pinned_geometry`），因为 `_resolve_geometry()` 会把空着的字段都填上数字，之后就分不清哪些是调用方设的。建调度表时，这些字段会在每一档覆盖表里的值。

`_resolve_geometry()` 对跨节点只填兜底默认值 96/64/8，不读表：跨节点每档要分别定两相的 rdma 和 QP 数，`cfg.schedule` 的格式（只有 block 和 warp）装不下；而且要同时读两个族的表。

单机才在这里查表：

- **HIP**（`hip_tuning_configs.lookup()`）：`quant_type="none"` 时先查 IntraNodeV2 的 CSV。只要有一相查到，查到的那相用 CSV，另一相用 Python 表，再没有就用默认值（64/16、64/8）。两相都没查到时，两相都用 Python 表；Python 表缺一相，就整个用默认值。
- **FlyDSL**（`tuning_configs.lookup()`）：按 `(world_size, hidden, topk)` 和 dtype 查。这个 dtype 查不到会退回 fp8 那套；MI308X 和 MI325X 上 hidden 4096、2048 直接复用 7168 的调度。所以 FlyDSL 会借别的规则，HIP 不会。

#### 读哪个文件

```python
path = (MORI_EP_V2_TUNING_DIR 或包里的 tuning_configs/) / f"{arch}_{model}_{kernel_type}_ep{world_size}.csv"
```

| 部分 | 来源 | 例子 |
|---|---|---|
| `arch` | `utils.arch_name()`：KFD sysfs 的 `gfx_target_version` | `gfx942` |
| `model` | `utils.detect_model()`：先按 PCI device id 查 `_DID_TO_MODEL`，查不到再按 arch 查 `_ARCH_TO_MODEL`（只有 gfx950 → mi355x） | `mi308x` |
| `kernel_type` | `v2` → `InterNodeV2`，`v2_ll` → `InterNodeV2LL`，单机 → `IntraNodeV2`；auto 两个族都读 | `InterNodeV2LL` |
| `ep` | `cfg.world_size` | `16` |

- 文件不存在就当空表，全部走默认值，不借别的型号的文件。
- 第一次读时严格校验：列名不对、一格解析不了、选择键重复，都直接报错，op 建不起来。
- 按路径缓存，整个进程只读一次；运行中改 CSV 不会生效，只有 `save_internode_result()` 会清缓存。

几种情况下实际读到什么：

| 场景 | 读哪个文件 | 结果 |
|---|---|---|
| MI308X 两机 EP16，`auto` | `gfx942_mi308x_InterNodeV2_ep16.csv`、`..._InterNodeV2LL_ep16.csv` | 两张表拼成一张调度表 |
| 同上，`internode_kernel="v2_ll"` | 只读 `..._InterNodeV2LL_ep16.csv` | LL 表的各档全用上 |
| MI308X 单机 EP8，HIP 后端 | `gfx942_mi308x_IntraNodeV2_ep8.csv` | 在 config 阶段查好 |
| MI350X 两机 EP16 | `gfx950_mi350x_...`，不存在 | 全部默认值，不借 MI355X 的表 |
| MI350X 单机，HIP 后端 | `gfx950_mi350x_IntraNodeV2_ep8.csv`，不存在 | 退到 Python 表，通过 `_MODEL_ALIAS` 用上 MI355X 的条目 |
| gfx942 上不认识的 device id | 识别不出型号 | 跨节点和单机都走默认值 |

MI350X 跨节点读不到 MI355X 的 CSV，单机 HIP 却能通过 Python 表的别名用上 MI355X 的条目，两条路径行为不一致。另外 gfx950 上不认识的 device id 会被当成 MI355X，而认识的 MI350X 反而没有表。

#### 优先级（跨节点，从高到低）

单机不适用：单机只要设了几何字段，就完全不查表。

1. `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM`：完全绕过表，只有一档。
2. 调用方在 config 里设过的几何字段：每一档都覆盖表里对应的字段。
3. `cfg.active_qps`：每一档、两相的 QP 数都用它。
4. 本型号 CSV 里匹配上的规则。
5. 兜底默认值 96/64/8，block 截到 CU 数，QP 数取 `min(2, num_qp_per_pe)`。

### 2.3 V2 选档的代码路径

下面以 MI308X 两机 EP16、DeepSeek-V3（256/8、hidden 7168、bf16）、`internode_kernel="auto"` 为例，跟着 `hip_tuning_configs.py` 里的函数走一遍。文中的输出都是直接调这些函数得到的真实结果（型号 mock 成 MI308X，不建 op，不用 GPU）。对应 2.2 调用链里 `resolve_schedule` 那一段和每次调用那一段。

#### 第 1 步：`internode_buckets()`，一张 CSV 变成一串档

```python
matched[phase] = [rule for rule in rules
    if rule["phase"] == phase
    and rule["num_experts"] == num_experts        # world_size × experts_per_rank = 256
    and rule["topk"] == topk
    and rule["dtype"] == dtype_str                # 两相各看各的 dtype
    and rule["hidden_dim"] == hidden_dims[phase]  # FP4 dispatch 用 hidden / 2
    and (phase == "dispatch"
         or (not rule["zero_copy"] and rule["quant_type"] == "none"))]
edges.update(rule["num_tokens"] for rule in matched[phase])
```

- 只做精确匹配，所以不会借别的形状的规则。整个形状都没有规则时返回 `None`，后面走默认值。
- 两相的档边界取并集。每个边界上，各相分别取"上限 ≥ 边界的第一条规则"，取不到就用最后一条。所以 dispatch 和 combine 的档边界不需要对齐。
- 几何经过 `fit_internode_geometry(..., clamp_to_cu=True)`：block 截到 CU 数，并保证 rdma < block。

V2LL 表得到 11 档。每档是（上限, dispatch B/R/W/QP, combine B/R/W/QP, load-first）：

```text
(4,    (64,42,8,2),  (32,16,16,1), False)
(8,    (32,21,8,2),  (32,16,4,1),  False)
...
(128,  (64,32,16,1), (16,10,16,2), False)
...
(4096, (80,53,12,1), (24,15,12,4), False)
```

#### 第 2 步：`internode_schedule()`，两个族拼成一张表

```python
if any(overrides):      # 设了 MORI_EP_DISP_GEOM / MORI_EP_COMB_GEOM：完全绕过表，只有一档
    return [(None, ...)]

tables = {family: internode_buckets(...) for family in ("v2", "v2_ll")}  # auto 两张都读
edges = 两张表所有上限的并集
edges.add(cfg.internode_auto_ll_max_tokens)   # 切换点 512 本身也是一个边界

for ceiling in [*sorted(edges), None]:
    family = internode_kernel_family(cfg, ceiling)   # ≤ 512 是 v2_ll，否则是 v2
    tuned = pick_bucket(tables[family], ceiling)     # 只从这个族的表里取
    # 这一相没有规则：用 config 默认值（96/64/8，截到 CU 数）
    # QP 数：resolve_active_qps()，cfg.active_qps 优先，否则用表里的值，
    #        再没有就是 2，最后截到 num_qp_per_pe
    # load_first 只在 family == "v2" 时生效
    # 和上一档完全相同就合并，上限取后一个
```

真实输出是 10 档，比两张表各自的 11 档还少：

```text
(4,    (64,42,8,2),  (32,16,16,1), False)   ┐
(8,    (32,21,8,2),  (32,16,4,1),  False)   │
(16,   (80,40,4,2),  (80,40,4,1),  False)   │ ≤ 512：全部来自 V2LL 表
(64,   (80,48,8,2),  (64,48,6,2),  False)   │ ← LL 表里 ≤32 和 ≤64 两档一模一样，合并了
(128,  (64,32,16,1), (16,10,16,2), False)   │
(256,  (72,48,12,2), (16,10,16,2), False)   │
(512,  (80,48,8,2),  (64,48,6,4),  False)   ┘
(1024, (80,48,8,4),  (64,48,6,4),  False)   ┐ > 512：全部来自 V2 表
(2048, (72,43,16,2), (32,19,8,4),  True)    │
(None, (80,48,8,1),  (16,9,16,4),  True)    ┘ ← ≤4096 和末尾的"其余"相同，合并成没有上限的最后一档
```

从这张表能看出三点：

- **auto 只用到每张表的一半。** V2LL 表里 ≤1024、≤2048、≤4096 三档在 auto 下永远用不到，只有指定 `"v2_ll"` 时才用；V2 表里 ≤512 的各档也一样，只有指定 `"v2"` 时才用。
- **load-first 跟着族走。** V2 表里 ≤512 的 combine 规则写了 load-first，但 auto 下这些档走 LL，所以全是 `False`。
- **最后一档没有上限。** T 超过 2048 都落到这里，所以 ≤4096 那条规则实际管的是所有 T > 2048 的调用。

#### 第 3 步：`resolve_schedule()`，每档标上 kernel 族

每一行变成 `ScheduleEntry(max_tokens, dispatch=KernelConfig(...), combine=KernelConfig(...))`，`KernelConfig` 里带着 `family`。第 2 步合并相同档时，两档可能正好跨过切换点，所以这一步会把 512 重新加回边界，保证 LL 和 V2 不会混进同一档。

`hip_backend.py` 建 op 时接着做三件事：

```python
self._schedule = tuning.resolve_schedule(cfg)
logger.info("%s", tuning.describe_schedule(cfg, self._schedule))   # 只在 rank 0
self._dispatch_specs = tuple(dict.fromkeys(row.dispatch for row in self._schedule))
self._combine_specs  = tuple(dict.fromkeys(row.combine  for row in self._schedule))
```

`KernelConfig` 是 NamedTuple，可以直接当字典的键。几档用同一个 `KernelConfig` 时去重，只编译一次。rank 0 打印的调度表：

```text
tokens     <=4: dispatch v2_ll B/R/W=64/42/8 qps=2, combine v2_ll B/R/W=32/16/16 qps=1
tokens     <=8: dispatch v2_ll B/R/W=32/21/8 qps=2, combine v2_ll B/R/W=32/16/4 qps=1
tokens    <=16: dispatch v2_ll B/R/W=80/40/4 qps=2, combine v2_ll B/R/W=80/40/4 qps=1
tokens    <=64: dispatch v2_ll B/R/W=80/48/8 qps=2, combine v2_ll B/R/W=64/48/6 qps=2
tokens   <=128: dispatch v2_ll B/R/W=64/32/16 qps=1, combine v2_ll B/R/W=16/10/16 qps=2
tokens   <=256: dispatch v2_ll B/R/W=72/48/12 qps=2, combine v2_ll B/R/W=16/10/16 qps=2
tokens   <=512: dispatch v2_ll B/R/W=80/48/8 qps=2, combine v2_ll B/R/W=64/48/6 qps=4
tokens  <=1024: dispatch v2 B/R/W=80/48/8 qps=4, combine v2 B/R/W=64/48/6 qps=4
tokens  <=2048: dispatch v2 B/R/W=72/43/16 qps=2, combine v2 B/R/W=32/19/8 qps=4 load-first
tokens    rest: dispatch v2 B/R/W=80/48/8 qps=1, combine v2 B/R/W=16/9/16 qps=4 load-first
```

#### 第 4 步：每次调用，`pick_bucket()`

```python
def pick_bucket(buckets, num_tokens):
    for row in buckets:
        if row[0] is None or num_tokens <= row[0]:
            return row
    return buckets[-1]
```

- **dispatch**：`n = input.shape[0]`，超过容量直接报错，然后 `self._pick(n)`。
- **combine**：用 `routing.cur_rank_num_token`，也就是对应那次 dispatch 的源 token 数。所以 combine 一定和它的 dispatch 落在同一档，不会按收到的 token 数另选。
- `set_active_qps(n)` 只是在选出的档上把两相的 QP 数换掉，不重新选档。

调用时只是在不到 10 档里线性扫一遍，不读文件，也不编译。实际选档结果：

| T        | 落在哪档 | 族  | dispatch          | combine                      |
| -------- | -------- | --- | ----------------- | ---------------------------- |
| 1        | ≤4       | LL  | 64/42/8，2 个 QP  | 32/16/16，1 个 QP            |
| 100、128 | ≤128     | LL  | 64/32/16，1 个 QP | 16/10/16，2 个 QP            |
| 129      | ≤256     | LL  | 72/48/12，2 个 QP | 16/10/16，2 个 QP            |
| 300、512 | ≤512     | LL  | 80/48/8，2 个 QP  | 64/48/6，4 个 QP             |
| 513、700 | ≤1024    | V2  | 80/48/8，4 个 QP  | 64/48/6，4 个 QP             |
| 3000     | 最后一档 | V2  | 80/48/8，1 个 QP  | 16/9/16，4 个 QP，load-first |

### 2.4 spec 和 pass：调度表怎么变成 kernel

**spec**（`KernelConfig`）是一相 kernel 的完整编译身份：

```python
class KernelConfig(NamedTuple):
    family: str            # "v2" 或 "v2_ll"
    block_num: int         # B
    warp_per_block: int    # W
    rdma_block_num: int    # R
    load_first: bool       # 只有普通 V2 的 combine 用
    active_qps: int        # 这个 kernel 在几条 QP 上发
```

这几个值都是编译期常量，任何一个不同就是另一套 kernel。调度表里各档的 spec 先去重（`dict.fromkeys`），上面的例子里 10 档只有 9 个不同的 dispatch spec 和 9 个不同的 combine spec。

**pass** 是一次真正的 GPU kernel launch。一个 spec 按 `(phase, family)` 展开成几个按顺序执行的 pass：

```python
_INTERNODE_SEQ = {
    ("dispatch", "v2"):    ("copystaging", "dispatch"),
    ("dispatch", "v2_ll"): ("copystaging", "dispatch_ll"),
    ("combine",  "v2"):    ("combinesync", "combinesyncbarrier", "combine",    "combineall"),
    ("combine",  "v2_ll"): ("combinesync", "combinesyncbarrier", "combine_ll", "combineall"),
}
```

各个 pass 的 grid 由 `ep_internode_spec.cpp` 的 `EpInterNodeGeometry` 决定，blockDim 一律是 `W × waveSize`：

| pass | grid | 用到 spec 的哪些字段 |
|---|---|---|
| `copystaging` | CU 数 | W |
| `dispatch` / `dispatch_ll` | B | B、W、R、QP、收尾的 QP 数 |
| `combinesync` | CU 数 | W |
| `combinesyncbarrier` | 1 个 block，只有 1 个 wave | 无 |
| `combine` / `combine_ll` | B | B、W、R、QP、load-first |
| `combineall` | CU 数 | W |

所以表里调的 B 和 R 只影响主 pass；W 影响每个 pass；family 只决定主 pass 用哪个。

`_build_internode_kernels()` 把 spec 变成 kernel：

1. 每个 spec 再按 `cfg.active_qp_counts` 展开成几个 QP 数不同的副本，供 `set_active_qps()` 切换。
2. `num_qp_to_drain = max(所有 spec 的 active_qps)`，编进每个 dispatch kernel，作为收尾时要处理的 QP 数。
3. 对每个 spec 生成 JIT 请求（`_internode_request`），逐个 pass 编译成 plan。plan 按 `(pass, B, W, R, QP, load_first)` 缓存，**键里没有 family**，所以两族共有的 pass（`copystaging`、`combinesync`、`combinesyncbarrier`、`combineall`）在几何和 QP 数相同时只编译一次；`load_first` 只有普通 `combine` 这个 pass 会带进键里。
4. 一个 spec 的几个 pass 用 `make_launch_group` 串成一组，存成 `kernels[phase][spec]`。

一句话：**spec 是「选哪套配置」的单位，pass 是「实际编译、实际启动」的单位。** 调度表和 `_pick()` 处理的是 spec；编译和启动处理的是 pass。`_pick()` 返回的 spec 和建 op 时当键用的 spec 来自同一个 `self._schedule`，所以一定能在 `kernels` 里查到。

### 2.5 V1

1. **默认不查表。** `MORI_EP_LAUNCH_CONFIG_MODE` 默认是 `MANUAL`，用 config 里的 `block_num` / `rdma_block_num` / `warp_num_per_block`（默认 80、0、8），或者每次调用传进来的值。
2. **设成 `AUTO` 才查表。** 建 op 时读进两个 JSON（dispatch、combine 各一个）；每次调用按 dtype、这次的 token 数、hidden、topk 现查一条，把几何作为启动参数传给预编译 kernel。
3. **kernel 类型不切换。** InterNodeV1、InterNodeV1LL、AsyncLL 等由 config 定死，不会因为这次 token 多少而换。

|              | V1                                                       | V2                                                   |
| ------------ | -------------------------------------------------------- | ---------------------------------------------------- |
| 默认用不用表 | 不用，要设 `AUTO`                                        | 用（HIP 后端）                                       |
| 什么时候决定 | 每次调用现查                                             | 建 op 时定好并编译                                   |
| 每档能变什么 | 几何                                                     | 几何、QP 数、load-first；`auto` 时还有 kernel 族     |
| 匹配规则     | 会放宽 hidden 和 topk，还会借别的型号的文件（见第 3 节） | 必须完全一致，不借                                   |
| 代价         | 每次调用一次 Python 查表                                 | 建 op 时按档编译，档越多编译越久；调用时没有额外开销 |

## 3. 表里没有对应规则时会怎样，V1 又是怎样

**一句话**：V2 宁可用默认值，也不借别的形状、别的型号的规则；V1 会尽量找一条"差不多"的规则来用。

| 情况                          | V2                                                   | V1（`AUTO` 模式）                                                       |
| ----------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| token 数比最大档还大          | 用最大档                                             | 用最大档                                                                |
| token 数比最小档还小          | 用最小档                                             | 用最小档                                                                |
| 本型号没有表文件（如 MI325X） | 不借。跨节点用默认几何；单机先查 Python 表，再用默认 | 借同架构、同 kernel 类型、同 EP 的其他型号的文件，取按文件名排序的第一个。例如 gfx942 的 InterNodeV1 EP16 只有 MI308X 的文件，MI300X 和 MI325X 都会借它；InterNodeV1LL EP16 排第一的是 MI300X 的 |
| 有表，但没有这个形状          | 用默认                                               | 先放宽 topk，再放宽 hidden，最后两者都放宽；专家数本来就不参与匹配      |
| 只有一相有规则                | 有规则的那相查表，另一相用默认                       | 和 V2 一样，两相各查各的                                                |
| 这个 dtype 没有规则           | 那一相用默认                                         | 那一相用默认                                                            |
| `MANUAL` 模式（V1 的默认）    | —                                                    | 根本不查表                                                              |

默认值是什么：

- **V2 跨节点**：两相都是 96/64/8，block 截到 CU 数（MI308X 上是 80/64/8），active QP 取 `min(2, num_qp_per_pe)`，几何不分档（`auto` 下仍按 512 切成 LL 和 V2 两档，配置相同）。
- **V2 单机**：Python 表（目前只有 MI355X 和 gfx1250 的条目，MI350X 共用 MI355X 的）；还没有就是 dispatch 64 个 block、每个 16 个 warp，combine 64 个 block、每个 8 个 warp。
- **V1 `AUTO`**：InterNodeV1 是 96/64/8，InterNodeV1LL 是 256/128/8，其他是 128/0/16。这时每次调用传的几何参数对 InterNodeV1 和 V1LL 不起作用。

**为什么 V2 不借**：借来的规则看起来"调过"，实际可能比默认值还慢，而且没有任何提示。MI308X 上就有现成的例子：修复 topk 9 的 combine 之前，把 256/8 的规则整套抄给 264/9 和 272/9，有几档比默认值慢了约三分之一。V1 的放宽匹配做的正是这件事：272/9 找不到 topk 9 的规则时，就会用 topk 8 的。

**怎么确认自己用的是哪条**：V2 的 rank 0 在建 op 时打印 `HIP V2 schedule: ...`，每张表标着 `found` 或 `missing`，下面列出每档的配置。看到 `missing`，或者列出的配置就是上面的默认值，就说明这个形状没有命中。

## 4. 默认值和接口有哪些变化

**一句话**：影响最大的是 `num_qp_per_pe` 默认值从 2 改成 8，其中对 AsyncLL 的影响没有测过；`internode_tuning_configs.lookup()` 破坏了兼容；其他都是新增功能，或者是表的来源变了。

| 接口 | main | 这个 PR | 谁会受影响 |
|---|---|---|---|
| `num_qp_per_pe` 默认值 | 2 | 8 | 所有跨节点 V2 用户，AsyncLL 影响最大 |
| `num_qp_per_pe` 取值检查 | ≥ 1 | 必须是 `int`，范围 1–64 | 传 numpy 整数或大于 64 的用户 |
| `active_qps`、`active_qp_counts`、`op.set_active_qps()` | 没有 | 新增 | 新功能，不影响老代码 |
| `MORI_EP_DISP_GEOM` / `MORI_EP_COMB_GEOM` | `B,R,W` | 可以多写第 4 个字段，表示 QP 数 | 兼容 |
| `internode_tuning_configs.lookup()` | 返回三元组 | 必须传 `experts_per_rank`，返回四元组 | **破坏兼容** |
| 跨节点用的表 | 只有 MI308X 的 Python 表，只覆盖 256/8 的 hidden 7168 和 6144；LL 和 V2 共用一套几何，所有 dtype 都用 fp8 那套 | MI308X、MI300X、MI355X 的 CSV，按 kernel 族、dtype、专家数分别匹配 | 三个型号的用户配置都会变；MI300X、MI355X 从默认几何变成调过的配置 |
| 单机 HIP 用的表 | Python 表 | 先查 IntraNodeV2 的 CSV | 单机用 HIP 后端的用户 |

没变的：`internode_kernel="auto"` 和切换点 512；V1 的 `num_qp_per_pe`（默认 1）。

### 4.1 `num_qp_per_pe` 从 2 改成 8

这个值的含义变了。在 main 上它既是**分配数**也是**发送数**：kernel 编译时直接用它（`numQpPerPe=cfg.num_qp_per_pe`），每次都在全部 2 条 QP 上发。现在它只表示分配多少条，每档实际发几条由表决定。分三种情况：

- **有表的形状**（比如 MI308X 的 DeepSeek-V3）：发送数改由表决定，每档 1、2 或 4 条；dispatch 按全 op 的最大值收尾，也就是 4 条。这是 PR 想要的效果，有整表 A/B 的数据支撑。
- **没有表的形状**：发送数取 `min(2, 8) = 2`，和以前一样。但分配数从 2 变成了 8：
  - 每对 GPU 占用的 QP、网卡和 CCO 上下文都变成原来的 4 倍。arena 里的 `recv_token_num` 区域也跟着变大，不过只多不到 1 KB。
  - 在 MI308X 上，分配数本身就会影响延迟：分配 64 条时，T ≤ 512 的每一相大约多 270 µs（见 tuning README）。分配 8 条和 2 条比有没有影响，**没有测过**。
- **AsyncLL**：在所有已分配的 QP 上发（`active_qps=cfg.num_qp_per_pe`），并且不接受 `active_qps`。所以默认从"在 2 条上发"悄悄变成了"在 8 条上发"。PR 里没有 AsyncLL 在这个变化前后的性能数据。**这是风险最大的一项。**

### 4.2 新增 `active_qps`、`active_qp_counts`、`set_active_qps()`

```python
cfg = EpDispatchCombineConfig(..., num_qp_per_pe=8,
                              active_qps=None,           # 设了就固定所有档、两相都用这个数，覆盖表
                              active_qp_counts=(2, 4))   # 额外预编译的 QP 数
op.set_active_qps(4)    # 只能切到 active_qp_counts 里列出的值，不会现场编译
```

- 只有跨节点能用。单机设了会直接 `ValueError`，AsyncLL 设了会被拒绝。
- `active_qp_counts` 里的每个值，都要在每一档的配置上各编译一份，建 op 会变慢。
- **容易被忽略的代价**：dispatch 收尾的 QP 数取所有可能值里的最大值，`active_qp_counts` 也算在内。往里加一个 8，所有档的 dispatch 都要按 8 条收尾，即使从来没切到 8 也一样。

### 4.3 更严格的取值检查

`num_qp_per_pe` 现在必须满足 `type(x) is int`，并且在 1 到 64 之间。上限 64 来自新的 combine barrier 协议：每次 combine 合计加 64，QP 0 加的是剩下的部分，必须为正数。

以前传 `np.int64(2)` 能用，现在会报错；`True` 这类布尔值也会被拒绝。

### 4.4 `internode_tuning_configs.lookup()` 破坏了兼容

```python
# main
lookup(world_size, hidden_dim, topk, num_tokens, dtype="fp8")
  -> {"dispatch": (B, R, W), "combine": (B, R, W)}       # dtype 查不到会退回 fp8 的规则

# 这个 PR
lookup(world_size, hidden_dim, topk, num_tokens, dtype="fp8", *,
       experts_per_rank, kernel_family="v2")
  -> {"dispatch": (B, R, W, QP), "combine": (B, R, W, QP), "combine_load_first": bool}
```

- `experts_per_rank` 是必填的关键字参数，不传会直接 `TypeError`。
- 返回的元组从 3 个元素变成 4 个，按三元组解包的代码会报错。
- dtype 查不到时不再退回 fp8 的规则，而是返回 `None`。

### 4.5 表的来源变了，行为也跟着变

- **MI300X、MI355X 跨节点用户**：以前没有表，走默认几何；现在会读到调过的 CSV 表，配置会变。其中 MI355X 的表还在确认中（见 `qp_status_mi355x.md`）。
- **单机 HIP 用户**：`quant_type` 为 none 时先查 IntraNodeV2 的 CSV，查不到再用 Python 表。gfx942 和 gfx950 单机默认走 FlyDSL 后端，不受影响。
- **表格式错误会让建 op 失败**：一格解析不了、多一列或少一列、两行的选择键重复，都会直接报错，不会跳过这一行。好处是表写错了立刻能发现；代价是装包时带了一张坏表，这个型号的所有用户都建不起 op。CI 新增了一个 CPU 任务专门加载所有表，就是为了防这种情况。
- **load-first**：main 上的 V2 跨节点 combine 没有 load-first，这是这个 PR 新加的功能，按档在表里声明，不算行为变化。
- **`EpDispatchCombineConfig.tuned()`**：HIP 后端下等同于普通构造函数；FlyDSL 后端保持原来的行为。

### 4.6 建议

- PR 描述里写明 `num_qp_per_pe` 的默认值变化，并补上 AsyncLL 分配 2 和 8 的对比数据；或者让 AsyncLL 单独保留 2 条。
- 考虑把默认值降到 4：表里最大只用到 4，`qp_status.md` 里也提过这个想法。需要先用整表 A/B 确认分配 4 时性能不变。
- 合入前搜一下 `internode_tuning_configs.lookup()` 的外部调用方，比如 aiter 和 sglang。

## 5. PR 里夹带的 combine 修复

### 5.1 是什么问题

**一句话**：V2LL 的 combine 有时会把某个节点发来的 token 当成"没有"而跳过。这些 token 在本节点 expert 上的结果没有送回去，最终输出悄悄出错：不报错，也不卡住。

先看 LL dispatch 怎么发数据：

- 发送方把自己的 token 每 warpSize 个（MI300 和 MI355 上是 64）分成一组，每组用一次 RDMA 写到对端的一个 slot，并附带一个"这组到了"的标志。数据和它的标志走同一条 QP，所以看到标志就一定有数据。各组按编号轮流分到多条 QP 上。
- 全部发完之后，发送方再单独发一个原子操作，告诉对端"我一共用了几个 slot"，下文叫**节点计数**。它固定走 QP 0。
- 不同 QP 之间没有先后保证，所以**节点计数可能比最后一组数据晚到**。

接收方的 dispatch 逐个 slot 等标志，节点计数只用来提前知道"后面的 slot 不会来了，不用再等"。如果发送方把所有 slot 都用满了，每个 slot 都等得到标志，dispatch 不看计数就能结束。

combine 紧接着开始。修复前，它直接读节点计数；如果计数还在路上，就读到 0，当作"这个节点没发 token"跳过。于是发送方的 token 在接收方 expert 上算出的结果没有发回去；发送方做最终求和时，读到的是缓冲区里上一轮留下的旧值。

```text
节点 A（发送方）                              节点 B（接收方）
每组数据写进 B 的 slot（分散在多条 QP 上） ──▶  每个 slot 的标志都到了，dispatch 结束
发"一共用了 N 个 slot"（QP 0，还在路上）       combine 开始，读节点计数：0
                                              → 以为 A 没发 token，跳过 A
                                       ──▶   节点计数这时才到，已经晚了
```

### 5.2 怎么修的

- 发送方写的节点计数原本就是 `slot 数 × warpSize + 1`，一个 token 都没有也写 1，所以 0 只可能表示"还没到"。这一侧没有改。
- 改的是 combine 这一侧（提交 `61254ac8`）：以前读到 0 就当成"这个节点没有 token"跳过；现在读到 0 就一直等，直到读到非 0。读到 1 才确定这个节点确实没有 token。
- 同一个提交还修了 V2 和 V2LL combine 的发布顺序。combine 的一个 64-token 块由多个 warp 一起算，最后算完的 warp 负责让网卡把这块发出去。修复前，没有保证其他 warp 写的数据这时已经对网卡可见。现在每个 warp 报告完成前先做 warp 内同步和 release fence，最后那个 warp 发出之前再做 acquire fence。

### 5.3 什么样的流量会触发

下面几条要同时满足：

1. **走 LL**：`internode_kernel="v2_ll"`，或者 `"auto"` 且这次调用不超过 512 个 token。
2. **跨节点**：至少两台机器。
3. **某个 rank 这次发的 token 把容量用满了**：`ceil(T/64) == ceil(容量/64)`，也就是 T 落在容量的最后一组里。例如容量 128 时 T 为 65–128，容量 512 时 T 为 449–512；容量不超过 64 时，只要有 token 就算满。这和 token 发往哪些 expert 无关：LL 不去重，每个 token 都会发给每个远端节点，用几个 slot 只取决于 T。
4. **combine 紧跟在 dispatch 后面**：中间没有同步，也几乎没有别的 GPU 工作。
5. **节点计数晚到得足够多**：要比"dispatch 结束到 combine 读计数"之间那几微秒还晚。计数是在发送方全部发完之后才单独发出的，只走 QP 0，和其他 QP 上的数据没有先后约束；网络越忙、各条 QP 的进度差得越多，它越可能落在最后。

另外，要**看得出错**，这一轮的数据还得和上一轮不同，否则读到的旧值恰好就是正确答案。

由此可以推出：

- `auto` 在容量超过 512 时不会触发：走 LL 的调用最多 512 个 token，填不满容量。
- 真实推理中，dispatch 和 combine 之间隔着 expert 的 GEMM（几十到几百微秒），计数通常早就到了，碰到的概率小得多。但这不是保证，比如这个 rank 的 expert 没分到 token、GEMM 很短，就可能碰上。
- 最容易触发的是基准测试和调优：容量等于 token 数，dispatch 和 combine 紧挨着跑。

### 5.4 为什么调优时才碰到

调优恰好同时满足上面所有条件，而且是反复满足：

- 每一档都把容量设成等于 token 数，也就是每一档都是满的；LL 族的每一档都会单独测。
- 计时循环里 dispatch 和 combine 背靠背跑，中间没有同步。
- 每个配置连续跑上千轮，还要换几十种几何和 1、2、4 个 QP，相当于在各种时序下反复试。

但即使在调优里也不容易"看见"：计时循环每轮输入都一样，出错那一轮读到的旧值正好等于正确值，要在 combine 之后、换一份数据检查才看得出来。后来加的压力测试选项（`902ede93`）就是照这个思路设计的：`--stress-full-node N` 让节点 N 每轮都发满容量；`--stress-check-interval K` 每 K 轮在 combine 之后原地比对结果，而不是在 dispatch 和 combine 之间插同步。

**证据现状**：当时的工作笔记只记了这个修复在 9 月 21 日的 A/B 之后加入、9 月 30 日的重扫已经包含，没有保存出错的现场。修复后，在 MI308X EP16 上用这两个选项做了压力测试：每例 2 万轮、每 50 轮比对一次，覆盖 v2_ll 的不齐负载、节点 0 满容量（512 和 4096）以及 `auto`，全部通过。但还没有在修复前的代码上把问题复现出来。

### 5.5 为什么 CI 没测出来

CI 的双机任务（`.github/workflows/ci_cco.yml` 里的 `cco-internode-test`，两台 MI355X）有三类用例，每一类都缺了至少一个条件：

| CI 步骤                                                            | 满足了什么                           | 缺了什么                                                                                                                        |
| ------------------------------------------------------------------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `--cmd test`：v2 128、v2 4096、v2_ll 128，以及两个切换 QP 数的用例 | 每轮都发满容量，每轮换数据并检查结果 | 每轮 dispatch 之后先 `synchronize` 加 `barrier` 再跑 combine，计数早就到了                                                      |
| `--cmd bench`：128、4096                                           | 满容量，背靠背                       | 只跑 `v2`，没有 `v2_ll`；计时循环不检查结果，而且每轮数据一样，错了也看不出                                                     |
| `--cmd stress`：`auto`、容量 4096、2000 轮                         | 背靠背，每轮换数据                   | 容量 4096 时，`auto` 只有不超过 512 个 token 的调用走 LL，最多用到 64 个 slot 中的 8 个，从来填不满；当时也只查卡死，不比对结果 |

单机的 CI 任务没有跨节点 LL，更碰不到。现在 CI 的 stress 步骤仍然没有用 `--stress-full-node` 和 `--stress-check-interval`。

### 5.6 建议

在 CI 双机任务里加一个同时满足这些条件的用例：

```bash
--cmd stress --kernel-type v2_ll --max-tokens 128 --rounds 2000 \
    --stress-full-node 0 --stress-check-interval 50 --num-qp 8 --active-qps 4
```

最好再在修复前的代码上跑一次，确认这个用例真能把问题抓出来。否则它只能说明"修复后没问题"，说明不了"测试抓得住"。
