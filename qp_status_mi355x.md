# MI355X EPv2 调优：交接（2026-11-03）

分支：`ROCm/mori` 的 `dev/analyze_v1`（HEAD `7148de8bc`）。目标：MI355X 的 V2 表覆盖到和 MI308X 相同的形状、dtype 和 token 档，按 `qp_status.md` 里 308x 的流程调（几何 + 每档 QP 数 + load-first，整表 A/B 决定是否采纳）。

## 1. 现状

- **EP16 五组都测完了**（11-02 20:56 到 11-03 07:21），已经验证过的结果写进了仓库的两张跨节点表（working tree，**未 commit**）。两张表都从 77 行变成 233 行，和 308x 一样。
- **还没做完**：一部分结果要在修正后重测（§3），EP8 还没跑。剩下的全部由 `remaining.sh` 一条命令跑完，约 6,000 次测量，按这次的速度 1.5 到 2 小时（§5）。
- 调优工具的全部源码在本文附录 A（不在仓库里），按 A.1 解包就能用（§4、§6）。这次 campaign 的数据在 `/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori_355_fx/campaigns/`。

## 2. 仓库里的改动（未 commit）

| 文件 | 改动 |
|---|---|
| `tuning_configs/gfx950_mi355x_InterNodeV2_ep16.csv` | 77 → 233 行：新增 156 行，改了 28 行已有规则 |
| `tuning_configs/gfx950_mi355x_InterNodeV2LL_ep16.csv` | 77 → 233 行：新增 156 行，改了 32 行已有规则 |
| `tuning_configs/README.md` | MI355X 的 QP、load-first、覆盖、模型和测量条件几段 |
| `qp_status_mi355x.md` | 本文件 |

EP8 表（`gfx950_mi355x_IntraNodeV2_ep8.csv`）没有改。

写入规则由 `apply_existing.py`（附录 A）生成，规则是：

- **采纳的档**（整表 A/B 里 D+C 快 1% 以上且三轮都更快）：写新规则，metric 列取确认 A/B 中新表那一臂。
- **没采纳、原表也没有规则的档**：写一条固定为它今天实际运行的配置的规则（下一档的规则或默认值），metric 列取确认 A/B 中原表那一臂。不写的话，它会落到邻档的新规则上（§3 问题 1）。
- **排除**：Kimi K3 V2LL 的 T2048、T4096 两档（采纳时用的几何会算错，§3 问题 2），按今天的配置写；FP8 combine V2 256/8 h7168 的 T4，它 A/B 时依赖的 fp8 dispatch 规则后来变了，按今天的配置写。

写入后逐个 cell 校验过：590 个 (cell, phase) 里，未采纳的档实际生效的配置和今天完全一样，采纳的档和测过的完全一样；三张表重写一遍逐字节不变。

### 采纳的档在确认 A/B 里的变化（新表对原表，每档 D+C，两种 dispatch dtype 相加）

| 组 | 形状 | V2 采纳档数：平均（最好） | V2LL 采纳档数：平均（最好） |
|---|---|---|---|
| e16-dsr1 | DeepSeek-V3/R1 256/8 h7168 | 5：−2.6%（−5.2%） | 8：−5.4%（−14.4%） |
| e16-rest | 256/6 h7168 | 1：−0.8% | 2：−1.8%（−2.4%） |
| | DeepSeek-V4.1-Flash 384/6 h5120 | 8：−4.2%（−5.9%） | 6：−2.7%（−4.4%） |
| | DeepSeek-V4-Pro 384/6 h7168 | 2：−4.3%（−4.7%） | 5：−1.5%（−2.0%） |
| e16-new（对默认几何） | GLM-5 256/8 h6144 | 11：−9.8%（−17.8%） | 7：−5.2%（−7.6%） |
| | 272/9 h7168（共享专家融合，replica 路由） | 11：−9.6%（−16.8%） | 10：−4.9%（−9.9%） |
| | Kimi K3 896/16 h3584 | 10：−9.6%（−20.0%） | 8：−5.4%（−12.8%） |
| e16-fp4 | 256/8 FP4 dispatch（存 3584） | 7：−3.6%（−4.8%） | 4：−3.7%（−5.4%） |
| e16-fp8c | 256/8 h6144 对称 FP8 combine | 7：−8.3%（−18.8%） | 1：−1.5% |
| | 256/8 h7168 对称 FP8 combine | 7：−8.1%（−19.0%） | 5：−2.5%（−3.7%） |

所有采纳的档在确认 A/B 里都更快（最差 −0.8%）。没采纳的档保持今天的配置。

数据质量：约 4 万次测量；19 次没过 README 的噪声门限；约 415 次卡顿或被干扰的测量已隔离并补测；用 10-12/13 的旧 harness 日志预剪枝第 1 轮时留的 238 个探针，没有一个进入第 2 轮，说明预剪枝可靠。QP 因子模型在 355 上也成立：dispatch 时间几乎不随 load-first 变（中位差 0.5% 到 0.7%），combine 时间几乎不随 dispatch 的 QP 数变（中位差 0.5% 到 1.3%）。

表里每个 op 的最大 QP 数是 4（V2LL 的 272/9 和 Kimi K3 是 2）。V2 有 119 条规则设了 QP 数，V2LL 有 42 条。

## 3. 发现的问题

**问题 1（工具的 bug，已修）：合并时没采纳、原表又没有规则的档不写规则，结果落到了邻档的新规则上。** 确认 A/B 里因此变慢的档：FP4 V2LL T64 +11.4%、T8 +5.1%；FP8 combine V2LL +2.6% 到 +8.5%；GLM-5 V2LL T64/T128 +4.7%/+3.8%。原表本来就有规则的 dsr1、rest 不受影响。现在合并和落表都会给这种档写一条固定今天配置的规则（`fxtables.pinned_rule`）。仓库里的表已经按修正后的规则写了，但 dsr1、rest、new 的确认 A/B 要在修正后的合并表上重跑一次，FP4 和 FP8 combine 要换到修正后的 base 上重跑 A/B（§5 的 `remaining.sh`）。

**问题 2（疑似 kernel bug，要报给 kernel 那边）：Kimi K3（896/16 h3584）的 V2LL dispatch 用 256 block × 16 warp 时间歇性算错。**

- 几何 256/128/16 和 256/170/16，bf16 和 fp8 都有，T128、T256、T2048 出现过；EP16（2 × 8 MI355X），分配 8 个 QP、每档用 2 个。
- 54 次测量错了 4 次，每次都是 16 个 rank 里有 1 个没过恒等 expert 的校验。其他形状上用 256 × 16 测了上千次，一次没错。
- 证据：`mori_355_fx/campaigns/e16-new/evidence/g/r1/geo-v2_ll-dispatch-256-128-16/v2_ll.896x16h3584.bf16-bf16.T128.json` 等四个文件（`status: wrong`）。
- 处理：这两个几何在这个形状上整个排除（`fxanalyze.wrong_configs`：一个配置在某个形状、某个 phase 上错过一次，就在这个形状上都不能选）。受影响的两条规则已重选：V2LL T2048 bf16 改为 256/170/4，T4096 fp8 改为 256/170/8，待重测。仓库里这两档先按今天的配置（默认 96/64/8）写。
- 生产里只要有人把这个形状固定成这种几何，就会出现静默的结果错误。

**其他**：torchrun 的主端口原来从 36000 起，落在节点的临时端口段（32768–60999）里，一次 launch 撞上 EADDRINUSE。现在主端口用 24000–29998，launch 前先试 bind；一个节点的 launcher 出错后 30 秒内结束整个 launch。

## 4. 工具（源码在附录 A）

按 `qp_status.md` 的描述逆向的 308x fx 流程，测量和正确性检查全部用 harness（`test_dispatch_combine_v2_internode.py`）自己的函数。机器相关的设置（仓库、工作目录、两台节点、容器、网卡）都从环境变量和 `$FX_WORKDIR/machine.env` 读，脚本里没有写死。

| 文件 | 作用 |
|---|---|
| `fxconfig.py` | 设置：`MORI_REPO`、`FX_WORKDIR`（默认 `~/mori_ep_tuning`）、`FX_*`，读 `$FX_WORKDIR/machine.env`；容器里每次运行的环境 |
| `machine.env.example` | `machine.env` 的模板，值是这次的 49（node0）+ 50（node1） |
| `fxcells.py` | cell、形状和组的定义（纯 Python，host 和容器里都用） |
| `fxdriver.py` | 测量入口（容器里，torchrun）。每个 cell 建一个 op（capacity = token 数），`MORI_EP_V2_TUNING_DIR` 指向变体表，记录 `kernel_config()` 实际选的配置，恒等 expert 校验，3 × (20 + 30) 轮去掉每轮第一轮，README 噪声门限，证据写 JSON。FP4 用 bf16 cell 的同一组随机数取最近的 e2m1 打包，解码放在两个计时段之间 |
| `fxtables.py` | 变体表（种子、几何、QP/LF、合并），全部经仓库的 `write_v2_table` / `read_v2_table` 校验；每个 cell 的预期配置用仓库的 `resolve_schedule` / `lookup` 算 |
| `fxanalyze.py` | 选择规则：每个 cell 剪枝、几何（≥1% 且每轮都更快）、QP/LF 因子模型、整表 A/B、按档合并、卡顿检测、错误配置排除 |
| `fxrun.py` | 两节点启动（podman + ssh）、CI runner 冻结/恢复、KFD 扫描外来 GPU 进程、干扰隔离、看门狗 |
| `fxcampaign.py` | 命令行。机器：`where / make-container / install-mori / preflight / ci-freeze / ci-resume / ci-status`；campaign：`init / plan / run / analyze / chain / land / status / archive / rebase` |
| `fxharness.py` | 用旧 harness sweep 日志预剪枝第 1 轮（带探针） |
| `apply_existing.py` | 把已有结果写进仓库表（§2） |
| `remaining.sh` | 这次剩下的全部步骤（§5），需要这次的五个 `e16-*` campaign |
| `run_all.sh` | 从头跑整个 MI355X campaign（EP16 五组加 EP8），2 × 8 MI355X 上约 11 到 12 小时 |
| `fxsim.py`、`tests/` | 离线模拟和单元测试（`python3 -m unittest discover -s tests`，不需要 GPU） |

每组的阶段：`g1`（每个候选都测，37 个几何/phase；EP8 18 个）→ 按 cell 剪枝 → `g2`（离最优 5% 以内，至少 5 个）→ 选几何 → `q`（d、c ∈ {1, 2, 4}，V2 加 (q,q)+LF，两轮）→ 选 QP/LF → `ab`（原表对新表，3 轮交替）→ 按档合并 → `cf`（原表对合并表，3 轮）→ `vf`（每个 cell 换 20 组输入校验）→ 落表。和 308x 的不同：第 2 轮按 cell 剪枝（308x 是任一档接近最优就整条保留）；第 1 轮可以用旧 harness 日志预剪枝；所有新形状并在一个 campaign 里跑。

每组的目录 `$FX_WORKDIR/campaigns/<组>/`：`evidence/`（每次测量的 JSON）、`reports/`（每个阶段的文字报告，`cf.txt` 是确认 A/B）、`picks/`（选择结果）、`tables/`（变体表）、`archive/`（被替换的旧数据，没有删）、`logs/`。

实测速度：driver 里每个 cell 约 0.5 秒，含启动约 0.7 到 0.8 秒（308x 估的是 2 秒）。

## 5. 剩下的步骤

`remaining.sh` 按顺序做：

1. e16-dsr1、e16-rest：在修正后的合并表上重跑确认 A/B 和 `vf`。
2. e16-new：A/B、确认 A/B、`vf`（Kimi 两档已重选几何；这两档原来的 QP 数据是旧几何上测的，已丢弃，QP 回到默认，由 A/B 决定）。
3. 用 git HEAD 的 MI355X 表加这三组，重建 `$FX_WORKDIR/landed/`。
4. e16-fp4、e16-fp8c：`rebase` 到 `$FX_WORKDIR/landed/` 上，A/B、确认 A/B、`vf`、落表。
5. EP8（preset `ep8`：GLM-5 256/8 h6144、264/9 h7168 replica、384/6 h5120 重筛）从头跑，落到同一个目录。
6. 把 `$FX_WORKDIR/landed/` 的三张 MI355X 表覆盖到仓库，替换现在这版。

开始前它会先跑 `preflight`，两个节点都通过才继续；然后冻结 node0 上的 CI runner（如果有），最后恢复。

跑完以后要做的：看 `campaigns/*/reports/cf.txt` 和 `vf` 的结果（`fxcampaign.py status <组>`）；更新 README 里 EP8 的 Coverage 和 Measurement conditions；把 Kimi 的问题报给 kernel 那边；commit 表、README 和本文件。

## 6. 换机器

需要：两台 8 卡 MI355X，节点之间能免密 ssh，RDMA 能通。工具目录、仓库和 `$FX_WORKDIR` 在两台机器上（以及容器里）要是同一个路径：用共享文件系统，或者在 `machine.env` 里设 `FX_SYNC=rsync`。

1. 按附录 A.1 把脚本解包到一个目录（下面记作 `$TOOL`），然后：

   ```bash
   export MORI_REPO=/path/to/mori        # 这个分支的 checkout
   export FX_WORKDIR=~/mori_ep_tuning    # campaigns、日志、wheel 都放这里
   mkdir -p $FX_WORKDIR && cp $TOOL/machine.env.example $FX_WORKDIR/machine.env
   # 按新机器改 machine.env：容器名和镜像、挂载、node1 地址、网卡、FX_RUN_ENV；node0 是 CI runner 时设 FX_CI_FREEZE=1
   ```

   `MORI_REPO` 和 `FX_WORKDIR` 只能用环境变量设：`machine.env` 在 `$FX_WORKDIR` 里，读它之前这两个值就定了。不设 `MORI_REPO` 时默认是 `$TOOL` 往上两级。

2. 容器和 mori（两台都会做）：

   ```bash
   cd $TOOL
   python3 fxcampaign.py where             # 生效的设置、每台的容器状态
   python3 fxcampaign.py make-container
   python3 fxcampaign.py install-mori      # 构建这个分支的 mori，装进两台的容器
   python3 fxcampaign.py preflight         # 两台都要 MATCH、model=mi355x
   ```

3. 冒烟：

   ```bash
   python3 fxcampaign.py init smoke --preset test-ep16
   python3 fxcampaign.py run smoke g1 --sample 24 --chunk 24
   python3 fxcampaign.py status smoke      # 每个 cell 几秒、有没有错
   ```

4. 开跑。接着这次的进度跑（§5）要先把这次的五个 campaign 拷过去；从头跑整套用 `run_all.sh`：

   ```bash
   mkdir -p $FX_WORKDIR/campaigns
   cp -a /shared/amdgpu/home/jiahao_zhou_qle/zqz/mori_355_fx/campaigns/e16-* $FX_WORKDIR/campaigns/
   nohup setsid ./remaining.sh > $FX_WORKDIR/remaining.log 2>&1 < /dev/null &
   # 从头跑：nohup setsid ./run_all.sh > $FX_WORKDIR/run_all.log 2>&1 < /dev/null &
   ```

换到别的机器对表没有影响（表按 GPU 型号，不按机器）：10-12 测过 49+50 和 49+51，中位差 0.15%，最大 0.97%。但网卡、交换机配置不同的话，最好让 `remaining.sh` 里的原表一臂在新机器上重新测（它会测），看 `cf.txt` 里没采纳的档是不是在 ±1% 以内。

## 7. 注意事项

- **CI runner**：node0 是 GitHub Actions 自托管 runner 时，测量期间要 SIGSTOP 它的 Runner.Listener（`machine.env` 里设 `FX_CI_FREEZE=1`，`fxcampaign.py ci-freeze` 在两个作业之间冻结），结束后 `ci-resume`。CI 作业是连续排的（CCO unit test 约 28 分钟），冻结要等一个空档。
- **空闲等待**：每个阶段第一次 launch 前要求 GPU 连续空闲 `--quiet` 秒（默认 120，`remaining.sh` 用 30）。看到外来 GPU 进程会停掉当前 launch，隔离在最后一次干净检查之后结束的 cell，空闲后再测。
- **卡死**：一个 launch 里 300 秒没有新 cell（第一个 cell 1200 秒）就结束它，卡住的 cell 算一次失败，三次后放弃。
- **网卡**：宿主机重配网卡或 bond 后，长期运行的容器要重启，或进容器 `ibv_devices` 核对网卡数（qp_status.md §8.2 的教训）。
- **JIT**：每个新的 token 档第一次 JIT 编译要 2 到 6 秒，之后同一档的变体共用。
- **证据可续跑**：每个阶段只测没有证据的条目；要重测就用 `fxcampaign.py archive <组> <阶段>` 把旧证据挪走（不会删）。

## 附录 A：调优脚本全文

调优工具的全部文件：Python、shell、`machine.env` 模板，以及单元测试和它用的三张表。A.2 里每个文件是一个 ``#### `路径` `` 标题加一个代码块，路径相对工具目录。这次 campaign 是用 `/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori_355_fx/` 里的那份跑的；这一版是之后整理的可移植版本，机器相关的设置挪到了 `machine.env`（§4、§6）。

### A.1 解包

把 A.2 的文件写到 `$TOOL`，有 `#!` 的文件设成可执行，然后跑单元测试（不需要 GPU）：

```bash
SRC=qp_status_mi355x.md          # 本文件
TOOL=~/mori_ep_tuning/tool       # 解包目录
python3 - "$SRC" "$TOOL" <<'EOF'
import os, re, sys
src, dest = sys.argv[1], sys.argv[2]
fence = "`" * 3
lines = open(src, encoding="utf-8").read().split("\n")
i = 0
while i < len(lines):
    m = re.fullmatch(r"#### `([^`]+)`", lines[i])
    if m and lines[i + 2].startswith(fence):
        end = lines.index(fence, i + 3)
        path = os.path.join(dest, m.group(1))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[i + 3:end]) + "\n")
        if lines[i + 3].startswith("#!"):
            os.chmod(path, 0o755)
        i = end
    i += 1
EOF
cd "$TOOL" && python3 -m unittest discover -s tests
```

### A.2 文件

#### `fxconfig.py`

```python
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Where things are: the repo, the work directory, the node pair.

Nothing site-specific is built in. Settings come from the environment; before
that, `$FX_WORKDIR/machine.env` (KEY=VALUE lines, `#` comments) is read into
the environment for every key not already set there. See machine.env.example.

  MORI_REPO          repo root (default: the repo this file is in)
  FX_WORKDIR         campaigns, logs, manifests, wheels (default ~/mori_ep_tuning).
                     The tool, the repo and this directory must be at the same
                     path on both nodes and inside their containers -- a
                     shared filesystem, or FX_SYNC=rsync for the tool and the
                     work directory
  FX_CONTAINER       container name on both nodes
  FX_IMAGE           image `make-container` creates it from
  FX_CONTAINER_MOUNTS  extra `host:container` mounts for `make-container`
  FX_NODE0_LABEL     short name of node0 (this host) in logs
  FX_PEER            node1, as ssh reaches it (EP16 only)
  FX_PEER_LABEL      short name of node1 in logs
  FX_PEER_KEY_ALIAS  known_hosts name to check node1 against (optional)
  FX_MASTER_ADDR     node0's address as node1 reaches it (torchrun master)
  FX_IFNAME          node0's interface for the gloo / mori sockets
  FX_RUN_ENV         extra KEY=VALUE pairs for every run in the container,
                     e.g. the NIC: "MORI_DEVICE_NIC=ionic MORI_DISABLE_IONIC_CCQE=1"
  FX_CI_FREEZE       1: SIGSTOP the GitHub runner on node0 while measuring
  FX_RUNNER_DIAG     that runner's _diag directory (its job state is read there)
  FX_HARNESS_LOGS    colon-separated old harness sweep log dirs (fxharness, optional)
  FX_SYNC            rsync: copy the tool and the campaign's tables and
                     manifests to node1 before each launch
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("MORI_REPO") or HERE.parents[1]).resolve()
WORKDIR = Path(os.environ.get("FX_WORKDIR") or Path.home() / "mori_ep_tuning").resolve()


def _load_machine_env(path: Path):
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_machine_env(WORKDIR / "machine.env")


def get(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


# Run environment inside the container, for every launch. NIC-specific
# settings belong in FX_RUN_ENV.
BASE_RUN_ENV = {
    "MORI_GPU_ARCHS": "gfx950",
    "MORI_JIT_ARCH": "gfx950",
    "MORI_SKIP_PRECOMPILE": "1",
    "MORI_SHMEM_HEAP_SIZE": "16G",
    "OMP_NUM_THREADS": "1",
    "GPU_PER_NODE": "8",
    "MORI_APP_LOG_LEVEL": "warn",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def run_env() -> dict:
    env = dict(BASE_RUN_ENV)
    for pair in (get("FX_RUN_ENV") or "").split():
        if "=" in pair:
            key, value = pair.split("=", 1)
            env[key] = value
    return env


def harness_log_dirs() -> list:
    return [Path(p) for p in (get("FX_HARNESS_LOGS") or "").split(":") if p]
```

#### `machine.env.example`

```bash
# Copy to $FX_WORKDIR/machine.env (default FX_WORKDIR: ~/mori_ep_tuning) and
# edit. Read by every fxcampaign.py command; a variable set in the environment
# wins over the file. Values below: mi355-gpu-49 (node0) + mi355-gpu-50 (node1),
# where the 2026-11 campaign ran.

# Container on both nodes (fxcampaign.py make-container creates it).
FX_CONTAINER=mori_ep_tuning
FX_IMAGE=localhost/rocm/mori:ci
# Extra mounts, space-separated host:container[:opts]. The repo and FX_WORKDIR
# are mounted at their own path unless a mount here already covers them. On the
# ionic (Pollara) NICs the container needs the host's verbs provider:
FX_CONTAINER_MOUNTS=/shared:/shared /etc/libibverbs.d:/etc/libibverbs.d:ro /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0:/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libionic.so.1.1.39:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.39 /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so

# node0 is the host the commands run on; node1 is reached with ssh (keys, no
# password). EP8 needs node0 only; leave FX_PEER empty then.
FX_NODE0_LABEL=49
FX_PEER=10.2.80.10
FX_PEER_LABEL=50
FX_PEER_KEY_ALIAS=mi355-gpu-50
# node0's address and interface on the network node1 reaches it through
# (torchrun master, gloo and mori bootstrap sockets).
FX_MASTER_ADDR=10.2.80.20
FX_IFNAME=enp193s0f1np1

# Extra environment for every run in the container. ionic NICs:
FX_RUN_ENV=MORI_DEVICE_NIC=ionic MORI_DISABLE_IONIC_CCQE=1

# node0 is a GitHub Actions self-hosted runner: freeze it while measuring
# (SIGSTOP between two jobs, SIGCONT at the end). Its _diag directory is found
# from the running listener; FX_RUNNER_DIAG overrides that.
FX_CI_FREEZE=1
# FX_RUNNER_DIAG=/path/to/actions-runner/_diag

# Old harness sweep logs for pre-pruning round 1 (optional).
FX_HARNESS_LOGS=/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori_v2_refresh_20261012/logs_ll16:/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori_v2_refresh_20261012/logs_ep8:/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori_v2_refresh_20261012/logs_dsv41:/shared/amdgpu/home/jiahao_zhou_qle/zqz/mori_v2_refresh_20261012/logs_dsv4pro

# No filesystem shared by both nodes: rsync the tool, the harness, the tables
# and the manifests of each launch to node1 (same paths), and the wheel.
# FX_SYNC=rsync
```

#### `fxcells.py`

```python
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Cells, shapes and groups of an MI355X EPv2 tuning campaign.

Pure Python: imported on the host (no torch) and by the driver in the container.

A cell is one measured configuration: kernel family, shape, dispatch and combine
dtype, token count. Its token count is also the op's capacity and the ceiling of
the table bucket it exercises, as in the README protocol.

A group is the unit a campaign tunes: one family and shape, the dispatch dtypes
that share the group's combine rule, the token buckets, and which phases the
group owns. A dispatch rule is selected on its own dtype's cells; a combine rule
on every cell of the group that uses it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

KERNEL_TYPE = {"v2": "InterNodeV2", "v2_ll": "InterNodeV2LL", "intra": "IntraNodeV2"}
EP_SIZE = {"v2": 16, "v2_ll": 16, "intra": 8}
INTERNODE = ("v2", "v2_ll")
TABLE_DTYPES = ("bf16", "fp8_e4m3", "fp4")
PHASES = ("dispatch", "combine")

TOKENS_SMALL = (4, 8, 16, 32)
TOKENS_LARGE = (64, 128, 256, 512, 1024, 2048, 4096)
TOKENS_ALL = TOKENS_SMALL + TOKENS_LARGE

# Config defaults the op falls back to when a phase has no rule (dispatch_combine_op
# _resolve_geometry for internode; hip_tuning_configs._hip_default for intra).
INTERNODE_DEFAULT = (96, 64, 8)
INTRA_DEFAULT = {"dispatch": (64, 0, 16), "combine": (64, 0, 8)}
DEFAULT_ACTIVE_QPS = 2
NUM_CUS = 256


@dataclasses.dataclass(frozen=True)
class Shape:
    num_experts: int
    topk: int
    hidden: int  # logical width; an fp4 dispatch rule stores hidden // 2
    routing: str = "uniform"

    @property
    def tag(self) -> str:
        routing = "" if self.routing == "uniform" else f"-{self.routing}"
        return f"{self.num_experts}x{self.topk}h{self.hidden}{routing}"


@dataclasses.dataclass(frozen=True)
class Cell:
    family: str
    shape: Shape
    dispatch_dtype: str
    combine_dtype: str
    tokens: int

    @property
    def world(self) -> int:
        return EP_SIZE[self.family]

    @property
    def experts_per_rank(self) -> int:
        return self.shape.num_experts // self.world

    @property
    def key(self) -> str:
        return (
            f"{self.family}.{self.shape.tag}.{self.dispatch_dtype}-{self.combine_dtype}"
            f".T{self.tokens}"
        )

    def selector(self, phase: str) -> tuple:
        """The rule this cell's bucket reads in `phase`, as
        hip_tuning_configs._table_key() spells it."""
        if phase == "dispatch":
            width = (
                self.shape.hidden // 2
                if self.dispatch_dtype == "fp4"
                else self.shape.hidden
            )
            return (
                self.shape.num_experts,
                self.shape.topk,
                width,
                0,
                self.dispatch_dtype,
                False,
                "",
                self.tokens,
            )
        return (
            self.shape.num_experts,
            self.shape.topk,
            self.shape.hidden,
            1,
            self.combine_dtype,
            False,
            "none",
            self.tokens,
        )

    def to_json(self) -> dict:
        return {
            "family": self.family,
            "num_experts": self.shape.num_experts,
            "topk": self.shape.topk,
            "hidden": self.shape.hidden,
            "routing": self.shape.routing,
            "dispatch_dtype": self.dispatch_dtype,
            "combine_dtype": self.combine_dtype,
            "tokens": self.tokens,
        }

    @staticmethod
    def from_json(d: dict) -> "Cell":
        return Cell(
            d["family"],
            Shape(
                d["num_experts"], d["topk"], d["hidden"], d.get("routing", "uniform")
            ),
            d["dispatch_dtype"],
            d["combine_dtype"],
            d["tokens"],
        )


@dataclasses.dataclass(frozen=True)
class Group:
    """What one campaign tunes for one family and shape.

    `owns` is the set of phases whose rules this group writes. A dispatch-only
    group (FP4) measures against the combine rule another group owns; a
    combine-only group (symmetric FP8) against the dispatch rule another owns.
    """

    family: str
    shape: Shape
    dispatch_dtypes: tuple
    combine_dtype: str = "bf16"
    tokens: tuple = TOKENS_ALL
    owns: tuple = PHASES

    def __post_init__(self):
        if self.family not in KERNEL_TYPE:
            raise ValueError(f"unknown family {self.family!r}")
        bad = [
            d
            for d in self.dispatch_dtypes + (self.combine_dtype,)
            if d not in TABLE_DTYPES
        ]
        if bad or self.combine_dtype == "fp4":
            raise ValueError(f"unsupported dtypes {bad or [self.combine_dtype]}")
        if self.shape.num_experts % EP_SIZE[self.family]:
            raise ValueError(
                f"{self.shape} does not divide over EP{EP_SIZE[self.family]}"
            )
        if not set(self.owns) <= set(PHASES) or not self.owns:
            raise ValueError(f"owns must be a non-empty subset of {PHASES}")

    @property
    def tag(self) -> str:
        return f"{self.family}.{self.shape.tag}.{'+'.join(self.dispatch_dtypes)}-{self.combine_dtype}"

    def cells(self) -> list:
        return [
            Cell(self.family, self.shape, d, self.combine_dtype, t)
            for t in self.tokens
            for d in self.dispatch_dtypes
        ]

    def rule_selectors(self, phase: str) -> list:
        """Selectors of the rules this group owns in `phase`, one per bucket
        (and per dispatch dtype for dispatch)."""
        if phase not in self.owns:
            return []
        if phase == "dispatch":
            return [c.selector("dispatch") for c in self.cells()]
        return sorted({c.selector("combine") for c in self.cells()})

    def to_json(self) -> dict:
        return {
            "family": self.family,
            "shape": dataclasses.asdict(self.shape),
            "dispatch_dtypes": list(self.dispatch_dtypes),
            "combine_dtype": self.combine_dtype,
            "tokens": list(self.tokens),
            "owns": list(self.owns),
        }

    @staticmethod
    def from_json(d: dict) -> "Group":
        return Group(
            d["family"],
            Shape(**d["shape"]),
            tuple(d["dispatch_dtypes"]),
            d.get("combine_dtype", "bf16"),
            tuple(d.get("tokens", TOKENS_ALL)),
            tuple(d.get("owns", PHASES)),
        )


def cells_sharing_rule(cells, phase: str, selector: tuple) -> list:
    return [c for c in cells if c.selector(phase) == selector]


def safe_name(text: str) -> str:
    """A file name for a cell key or variant id."""
    return "".join(ch if ch.isalnum() or ch in "-_.+" else "_" for ch in text)


def spec_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def geometry_candidates(family: str, phase: str, num_cus: int = NUM_CUS) -> list:
    """The harness's quick grid (test_dispatch_combine_v2_internode._tune):
    blocks 8, 16 and powers of two up to the CU count plus the CU count, warps
    4/8/16, and for internode an RDMA split of 1/2 and 2/3 of the grid."""
    blocks = {b for b in (8, 16) if b < num_cus}
    b = 32
    while b <= num_cus:
        blocks.add(b)
        b <<= 1
    blocks.add(num_cus)
    warps = (4, 8, 16)
    if family == "intra":
        return [(blk, 0, w) for blk in sorted(blocks) for w in warps]
    out = []
    for blk in sorted(blocks):
        for w in warps:
            for r in sorted({blk // 2, blk * 2 // 3}):
                if 1 <= r < blk:
                    out.append((blk, r, w))
    return out


def default_geometry(family: str, phase: str) -> tuple:
    return INTRA_DEFAULT[phase] if family == "intra" else INTERNODE_DEFAULT
```

#### `fxtables.py`

```python
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Variant table sets on top of the repo's own V2 table code.

Host side, no torch: `tuning()` loads hip_tuning_configs.py from the repo with
the torch-only modules it imports stubbed out, so every table written here goes
through the repo's write_v2_table() checks, and `expected()` resolves a cell's
schedule with the repo's own resolve_schedule() / lookup().

A variant is a directory holding the three gfx950_mi355x tables. The op caches a
table by path (hip_tuning_configs._V2_RULES_CACHE), so a variant directory is
written once and never modified afterwards.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from fxcells import (
    DEFAULT_ACTIVE_QPS,
    EP_SIZE,
    INTERNODE,
    INTERNODE_DEFAULT,
    KERNEL_TYPE,
    Cell,
    Group,
)

from fxconfig import REPO  # noqa: E402

REPO_TABLES = REPO / "python/mori/ops/dispatch_combine_v2/tuning_configs"
ARCH, MODEL = "gfx950", "mi355x"
METRIC_FIELDS = (
    "avg_latency_us",
    "bandwidth_gbps",
    "avg_rdma_bandwidth_gbps",
    "avg_xgmi_bandwidth_gbps",
    "avg_ll_bandwidth_gbps",
    "bandwidth_metric",
)
_TUNING = None


def tuning():
    """hip_tuning_configs, loaded from the repo without torch or the mori package."""
    global _TUNING
    if _TUNING is not None:
        return _TUNING
    if "mori.ops.dispatch_combine_v2.hip_tuning_configs" in sys.modules:
        _TUNING = sys.modules["mori.ops.dispatch_combine_v2.hip_tuning_configs"]
        return _TUNING
    src = REPO / "python"

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    for pkg in ("mori", "mori.ops", "mori.ops.dispatch_combine_v2"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []
            sys.modules[pkg] = mod
    utils = load("mori.ops.utils", src / "mori/ops/utils.py")
    sys.modules["mori.ops"].utils = utils
    # Table dtypes are config strings; the stub maps them onto themselves so the
    # lookup functions can be called with strings instead of torch dtypes.
    stub = types.ModuleType("mori.ops.tuning_config")
    stub.DTYPE_TO_CONFIG_STR = {
        s: s for s in ("bf16", "fp8_e4m3", "fp8_e4m3_fnuz", "fp4", "fp32")
    }
    stub._report_rule_added = lambda *a, **k: None
    stub._report_rule_change = lambda *a, **k: None
    sys.modules["mori.ops.tuning_config"] = stub
    _TUNING = load(
        "mori.ops.dispatch_combine_v2.hip_tuning_configs",
        src / "mori/ops/dispatch_combine_v2/hip_tuning_configs.py",
    )
    return _TUNING


def table_name(family: str) -> str:
    return tuning().v2_table_filename(ARCH, MODEL, KERNEL_TYPE[family], EP_SIZE[family])


def rule_key(rule: dict) -> tuple:
    return tuning()._table_key(rule)


def rule_from_selector(selector: tuple) -> dict:
    num_experts, topk, hidden, phase_order, dtype, zero_copy, quant, tokens = selector
    rule = {
        "num_experts": num_experts,
        "topk": topk,
        "phase": "dispatch" if phase_order == 0 else "combine",
        "dtype": dtype,
        "hidden_dim": hidden,
        "num_tokens": tokens,
    }
    if phase_order == 1:
        rule["zero_copy"] = bool(zero_copy)
        rule["quant_type"] = quant or "none"
    return rule


class TableSet:
    """The three MI355X tables as {family: {selector: rule}}."""

    def __init__(self, rules=None):
        self.rules = {f: dict(rules.get(f, {})) if rules else {} for f in KERNEL_TYPE}

    @classmethod
    def load(cls, directory) -> "TableSet":
        directory = Path(directory)
        out = cls()
        for family in KERNEL_TYPE:
            path = directory / table_name(family)
            if path.is_file():
                for rule in tuning().read_v2_table(path, KERNEL_TYPE[family]):
                    out.rules[family][rule_key(rule)] = dict(rule)
        return out

    def copy(self) -> "TableSet":
        return TableSet(
            {f: {k: dict(r) for k, r in rs.items()} for f, rs in self.rules.items()}
        )

    def save(self, directory) -> Path:
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory}: a variant directory is written once")
        directory.mkdir(parents=True, exist_ok=True)
        for family in KERNEL_TYPE:
            tuning().write_v2_table(
                directory / table_name(family),
                list(self.rules[family].values()),
                KERNEL_TYPE[family],
            )
        return directory

    def write_into(self, directory) -> Path:
        """Overwrite the three tables in an existing directory (landing)."""
        directory = Path(directory)
        for family in KERNEL_TYPE:
            tuning().write_v2_table(
                directory / table_name(family),
                list(self.rules[family].values()),
                KERNEL_TYPE[family],
            )
        tuning()._V2_RULES_CACHE.clear()
        return directory

    def get(self, family: str, selector: tuple):
        return self.rules[family].get(selector)

    def upsert(self, family: str, selector: tuple, **fields) -> dict:
        rule = self.rules[family].get(selector)
        if rule is None:
            rule = rule_from_selector(selector)
            self.rules[family][selector] = rule
        for name, value in fields.items():
            if value is None:
                rule.pop(name, None)
            else:
                rule[name] = value
        return rule

    def drop_metrics(self, family: str, selector: tuple):
        rule = self.rules[family].get(selector)
        if rule is not None:
            for name in METRIC_FIELDS:
                rule.pop(name, None)

    def geometry(self, family: str, selector: tuple):
        rule = self.get(family, selector)
        if rule is None:
            return None
        return (rule["block_num"], rule["rdma_block_num"], rule["warp_per_block"])


# ---------------------------------------------------------------- variants


def effective_rule_geometry(tables: TableSet, family: str, selector: tuple):
    """The geometry the op reads for `selector`'s bucket from `tables`: the rule
    with the tightest ceiling at or above the bucket, else the last one
    (internode_buckets), else the config default. A missing T4 bucket therefore
    runs the T64 rule, as the op does."""
    num_experts, topk, hidden, order, dtype, zero_copy, quant, tokens = selector
    same = sorted(
        (
            sel
            for sel in tables.rules[family]
            if sel[:7] == (num_experts, topk, hidden, order, dtype, zero_copy, quant)
        ),
        key=lambda sel: sel[7],
    )
    if not same:
        from fxcells import default_geometry

        return default_geometry(family, "dispatch" if order == 0 else "combine")
    pick = next((sel for sel in same if tokens <= sel[7]), same[-1])
    return tables.geometry(family, pick)


def seed(base: TableSet, groups, materialize: bool = True) -> TableSet:
    """The geometry-screening seed: the base tables with each owned rule's active
    QP count and load-first back at the default (qp_status.md §10.1). With
    `materialize`, a bucket with no rule gets one holding the geometry it runs
    today, so every later stage has a rule to change."""
    out = base.copy()
    for group in groups:
        for phase in group.owns:
            for sel in group.rule_selectors(phase):
                if out.get(group.family, sel) is not None:
                    out.upsert(
                        group.family, sel, active_qps=None, combine_load_first=None
                    )
                    out.drop_metrics(group.family, sel)
                elif materialize:
                    block, rdma, warp = effective_rule_geometry(base, group.family, sel)
                    out.upsert(
                        group.family,
                        sel,
                        block_num=block,
                        rdma_block_num=rdma,
                        warp_per_block=warp,
                    )
    return out


def with_geometry(
    seed_tables: TableSet, groups, family: str, phase: str, geometry
) -> TableSet:
    """Every owned `phase` rule of `family` in `groups` set to `geometry`
    (block, rdma, warp[, active_qps]); missing rules are added, so a shape with
    no rows gets one per bucket. The other phase keeps the seed's rules."""
    out = seed_tables.copy()
    block, rdma, warp, *qp = geometry
    for group in groups:
        if group.family != family:
            continue
        for sel in group.rule_selectors(phase):
            out.upsert(
                family,
                sel,
                block_num=block,
                rdma_block_num=rdma,
                warp_per_block=warp,
                active_qps=qp[0] if qp else None,
            )
            out.drop_metrics(family, sel)
    return out


def with_qp(
    tables: TableSet, groups, family: str, d: int, c: int, load_first: bool
) -> TableSet:
    """Every owned rule of `family` in `groups` at dispatch count d, combine
    count c and (InterNodeV2 only) the given load-first."""
    if family not in INTERNODE:
        raise ValueError("active QP counts are internode only")
    if load_first and family != "v2":
        raise ValueError("load-first is read by InterNodeV2 only")
    out = tables.copy()
    for group in groups:
        if group.family != family:
            continue
        for sel in group.rule_selectors("dispatch"):
            _require(out, family, sel)
            out.upsert(family, sel, active_qps=d)
        for sel in group.rule_selectors("combine"):
            _require(out, family, sel)
            out.upsert(
                family,
                sel,
                active_qps=c,
                combine_load_first=True if load_first else None,
            )
    return out


def with_picks(tables: TableSet, picks) -> TableSet:
    """Apply picks: [{family, selector, fields}] where fields name the rule's
    geometry/flag columns (None removes one)."""
    out = tables.copy()
    for pick in picks:
        sel = tuple(pick["selector"])
        out.upsert(pick["family"], sel, **pick["fields"])
        out.drop_metrics(pick["family"], sel)
    return out


def pinned_rule(base_dir, cell: Cell, phase: str, num_qp_per_pe: int = 8) -> dict:
    """A rule for `cell`'s bucket that runs what `base_dir` runs there today
    (geometry, active QP count, load-first), for a bucket that has no rule of
    its own and would otherwise fall onto a neighbour's new rule."""
    spec = expected(base_dir, cell, num_qp_per_pe)[phase]
    fields = {
        "block_num": spec["block_num"],
        "rdma_block_num": spec["rdma_block_num"] or 0,
        "warp_per_block": spec["warp_per_block"],
    }
    if cell.family in INTERNODE and spec["active_qps"] not in (
        None,
        DEFAULT_ACTIVE_QPS,
    ):
        fields["active_qps"] = spec["active_qps"]
    if phase == "combine" and cell.family == "v2" and spec["load_first"]:
        fields["combine_load_first"] = True
    return fields


def merge_buckets(
    base: TableSet,
    final: TableSet,
    groups,
    accepted,
    base_dir=None,
    num_qp_per_pe: int = 8,
) -> TableSet:
    """`base` with final's rules for every accepted (family, shape tag, tokens)
    bucket of `groups`: every owned rule of that bucket moves together, because
    the bucket's dispatch rules and combine rule were chosen as a set.

    A bucket that is not accepted keeps what it runs today. Where `base` has a
    rule for it, that rule stays; where it has none, the bucket ran a
    neighbour's rule or the default, and the accepted buckets' new rules would
    change that, so with `base_dir` it gets a rule pinning today's
    configuration."""
    out = base.copy()
    accepted = {tuple(a) for a in accepted}
    for group in groups:
        for cell in group.cells():
            take = (group.family, group.shape.tag, cell.tokens) in accepted
            for phase in group.owns:
                sel = cell.selector(phase)
                if take:
                    rule = final.get(group.family, sel)
                    if rule is None:
                        out.rules[group.family].pop(sel, None)
                    else:
                        out.rules[group.family][sel] = dict(rule)
                elif base_dir is not None and base.get(group.family, sel) is None:
                    out.upsert(
                        group.family,
                        sel,
                        **pinned_rule(base_dir, cell, phase, num_qp_per_pe),
                    )
    return out


def _require(tables, family, sel):
    if tables.get(family, sel) is None:
        raise KeyError(f"{family} has no rule {sel}; select a geometry first")


# ---------------------------------------------------------------- expected


def _fake_cfg(cell: Cell, num_qp_per_pe: int):
    world = EP_SIZE[cell.family]
    return SimpleNamespace(
        world_size=world,
        hidden_dim=cell.shape.hidden,
        num_experts_per_token=cell.shape.topk,
        num_experts_per_rank=cell.experts_per_rank,
        dispatch_dtype=cell.dispatch_dtype,
        combine_dtype=cell.combine_dtype,
        internode_kernel=cell.family,
        internode_auto_ll_max_tokens=512,
        num_qp_per_pe=num_qp_per_pe,
        active_qps=None,
        active_qp_counts=None,
        async_ll=False,
        is_internode=cell.family in INTERNODE,
        _pinned_geometry=frozenset(),
        dispatch_block_num=INTERNODE_DEFAULT[0],
        dispatch_rdma_block_num=INTERNODE_DEFAULT[1],
        warp_num_per_block=INTERNODE_DEFAULT[2],
        combine_block_num=INTERNODE_DEFAULT[0],
        combine_rdma_block_num=INTERNODE_DEFAULT[1],
        combine_warp_num_per_block=INTERNODE_DEFAULT[2],
        schedule=None,
    )


def expected(table_dir, cell: Cell, num_qp_per_pe: int = 8) -> dict:
    """What op.kernel_config(cell.tokens) will report for `cell` built on the
    tables in `table_dir`, resolved by the repo's own code."""
    h = tuning()
    previous = os.environ.get("MORI_EP_V2_TUNING_DIR")
    os.environ["MORI_EP_V2_TUNING_DIR"] = str(table_dir)
    for name in ("MORI_EP_DISP_GEOM", "MORI_EP_COMB_GEOM"):
        os.environ.pop(name, None)
    try:
        if cell.family in INTERNODE:
            schedule = h.resolve_schedule(_fake_cfg(cell, num_qp_per_pe))
            entry = h.pick_bucket(schedule, cell.tokens)
            return {
                phase: {
                    "family": spec.family,
                    "block_num": spec.block_num,
                    "rdma_block_num": spec.rdma_block_num,
                    "warp_per_block": spec.warp_per_block,
                    "active_qps": spec.active_qps,
                    "load_first": bool(spec.load_first),
                }
                for phase, spec in (
                    ("dispatch", entry.dispatch),
                    ("combine", entry.combine),
                )
            }
        short = {"bf16": "bf16", "fp8_e4m3": "fp8", "fp4": "fp4"}[cell.dispatch_dtype]
        t = h.lookup(
            EP_SIZE["intra"],
            cell.shape.hidden,
            cell.shape.topk,
            dtype=short,
            experts_per_rank=cell.experts_per_rank,
            quant_type="none",
            dispatch_dtype=cell.dispatch_dtype,
            combine_dtype=cell.combine_dtype,
        )
        if t["schedule"]:
            _, db, dw, cb, cw = h.pick_bucket(t["schedule"], cell.tokens)
        else:
            db, dw, cb, cw = (
                t["dispatch_block_num"],
                t["warp_num_per_block"],
                t["combine_block_num"],
                t["combine_warp_num_per_block"],
            )
        return {
            "dispatch": {
                "family": "intra",
                "block_num": db,
                "rdma_block_num": None,
                "warp_per_block": dw,
                "active_qps": None,
                "load_first": False,
            },
            "combine": {
                "family": "intra",
                "block_num": cb,
                "rdma_block_num": None,
                "warp_per_block": cw,
                "active_qps": None,
                "load_first": False,
            },
        }
    finally:
        # Variant directories never change after they are written, so the
        # path-keyed rule cache stays valid; write_into() clears it.
        if previous is None:
            os.environ.pop("MORI_EP_V2_TUNING_DIR", None)
        else:
            os.environ["MORI_EP_V2_TUNING_DIR"] = previous


def table_geometries(tables: TableSet, family: str, phase: str) -> list:
    """Distinct (block, rdma, warp) a family's rules use in one phase."""
    order = 0 if phase == "dispatch" else 1
    found = {
        (r["block_num"], r["rdma_block_num"], r["warp_per_block"])
        for sel, r in tables.rules[family].items()
        if sel[3] == order
    }
    return sorted(found)


__all__ = [
    "REPO",
    "REPO_TABLES",
    "TableSet",
    "tuning",
    "table_name",
    "rule_key",
    "rule_from_selector",
    "seed",
    "with_geometry",
    "with_qp",
    "with_picks",
    "merge_buckets",
    "expected",
    "table_geometries",
    "DEFAULT_ACTIVE_QPS",
    "Group",
]
```

#### `fxdriver.py`

```python
#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Measure the cells of one manifest, one op per cell, under the README protocol.

Runs inside the container, one launch per node (fxrun.py starts it):

  EP16  tools/run_internode_test.sh --entry <this file> ... -- --manifest M
  EP8   torchrun --nnodes=1 --nproc_per_node=1 <this file> --manifest M

Each rank is spawned the way the v2 harness spawns its workers (one torchrun
process per node, --spawn workers under it), and everything that touches the op
is the harness's own code, imported from the repo: input generation and routing
(_generate_round), the timed passes and their grand-mean metrics (_row_metrics,
which hands back every rank's raw rounds), the RDMA token count. What is added
here is what the harness does not do: FP4 dispatch input (the bf16 cell's random
numbers rounded to e2m1 and packed, so routing is the same as the bf16 cell) and
its decode, placed where the FP8 cast sits, between the two timed phases.

Per cell: set MORI_EP_V2_TUNING_DIR to the cell's variant directory, build the
op with capacity = tokens, record the configuration kernel_config() selected,
check one dispatch+combine against the identity-expert golden, then time
passes x (warmup + rounds) and write the evidence JSON (rank 0). The op is
closed before the next cell.
"""

import argparse
import importlib.util
import json
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fxcells  # noqa: E402

# This file lives in <repo>/tools/ep_v2_tuning/.
REPO = Path(os.environ.get("MORI_REPO") or HERE.parents[1])
HARNESS = (
    REPO / "tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py"
)

# README "Measurement conditions": each phase mean/median <= 1.10 and
# max/median <= 5, per-pass D+C means within 1.20x of each other.
GATE_MEAN_MEDIAN = 1.10
GATE_MAX_MEDIAN = 5.0
GATE_PASS_SPREAD = 1.20

E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--spawn", type=int, default=8)
    # run_internode_test.sh always passes these three; the manifest carries the
    # real values, and --num-qp is checked against it.
    p.add_argument("--cmd", default=None)
    p.add_argument("--num-qp", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    args, _unknown = p.parse_known_args(argv)
    return args


def _load_harness():
    spec = importlib.util.spec_from_file_location("fx_v2_harness", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- fp4


class Fp4:
    """OCP e2m1, packed two per byte with the even element in the low nibble
    (tests/python/ops/dispatch_combine_v2/_data.py, torch float4_e2m1fn_x2)."""

    def __init__(self, torch, device):
        self.torch = torch
        levels = torch.tensor(E2M1_LEVELS, dtype=torch.float32, device=device)
        self.edges = (levels[1:] + levels[:-1]) / 2
        lut = torch.cat((levels, -levels))
        self.lut_f32 = lut
        self.lut_bf16 = lut.to(torch.bfloat16)

    def encode(self, x):
        """(packed float4_e2m1fn_x2 [rows, cols // 2], the e2m1 values as f32)."""
        torch = self.torch
        x = x.float()
        code = torch.bucketize(x.abs().clamp(max=E2M1_LEVELS[-1]), self.edges).to(
            torch.uint8
        )
        nibble = code | ((x < 0).to(torch.uint8) << 3)
        packed = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).contiguous()
        return packed.view(torch.float4_e2m1fn_x2), self.lut_f32[nibble.long()]

    def decode(self, packed):
        """bf16 [rows, 2 * cols] from packed [rows, cols]; e2m1 is exact in bf16."""
        torch = self.torch
        b = packed.view(torch.uint8)
        lo = self.lut_bf16[(b & 0xF).long()]
        hi = self.lut_bf16[(b >> 4).long()]
        return torch.stack((lo, hi), dim=-1).reshape(b.shape[0], b.shape[1] * 2)


# ---------------------------------------------------------------- stats


def summarize(rank_rows, passes, rounds, drop):
    """Medians, means and the README gates from _row_metrics' raw rows: per rank
    four metadata values, then pass-major, round-major (dispatch_us, combine_us),
    dropped rounds included."""
    disp, comb, per_pass = [], [], []
    for p in range(passes):
        dp, cp = [], []
        for row in rank_rows:
            for r in range(drop, rounds):
                off = 4 + 2 * (p * rounds + r)
                dp.append(float(row[off]))
                cp.append(float(row[off + 1]))
        per_pass.append(statistics.fmean(dp) + statistics.fmean(cp))
        disp += dp
        comb += cp

    def phase(values):
        med = statistics.median(values)
        mean = statistics.fmean(values)
        return {
            "median": med,
            "mean": mean,
            "min": min(values),
            "max": max(values),
            "mean_over_median": mean / med,
            "max_over_median": max(values) / med,
        }

    d, c = phase(disp), phase(comb)
    spread = max(per_pass) / min(per_pass)
    gates = {
        "dispatch_mean_median": d["mean_over_median"] <= GATE_MEAN_MEDIAN,
        "combine_mean_median": c["mean_over_median"] <= GATE_MEAN_MEDIAN,
        "dispatch_max_median": d["max_over_median"] <= GATE_MAX_MEDIAN,
        "combine_max_median": c["max_over_median"] <= GATE_MAX_MEDIAN,
        "pass_spread": spread <= GATE_PASS_SPREAD,
    }
    return {
        "dispatch": d,
        "combine": c,
        "total_median": d["median"] + c["median"],
        "total_mean": d["mean"] + c["mean"],
        "per_pass_total_mean": per_pass,
        "pass_spread": spread,
        "gates": gates,
        "gates_ok": all(gates.values()),
        "samples_per_phase": len(disp),
    }


def _spec(kernel_config):
    return {
        "family": kernel_config.family,
        "block_num": kernel_config.block_num,
        "rdma_block_num": kernel_config.rdma_block_num,
        "warp_per_block": kernel_config.warp_per_block,
        "active_qps": kernel_config.active_qps,
        "load_first": bool(kernel_config.load_first),
    }


def _write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(tmp, path)
    os.chmod(path, 0o664)


# ---------------------------------------------------------------- one cell


def _verify(torch, op, cfg, comm, inp, ref, idx, wts, sc, convert, fp8):
    """One dispatch + combine against the analytic golden: an identity expert
    makes combine[t] = U[t] * input[t] and out_weights[t] = U[t] * wts[t], U[t]
    the distinct destination ranks of token t. Tolerances are the harness's
    (_verify_once). Weights are always folded here."""
    dispatch_out = op.dispatch(inp, wts, sc, idx, return_routing=True)
    torch.cuda.synchronize()
    comm.barrier()
    out, out_w = op.combine(convert(dispatch_out[0]), wts, routing=dispatch_out[5])
    torch.cuda.synchronize()
    comm.barrier()
    pes = idx.long() // cfg.num_experts_per_rank
    unique = (
        torch.stack(
            [(pes[:, j : j + 1] != pes[:, :j]).all(dim=1) for j in range(pes.shape[1])],
            dim=1,
        )
        .sum(dim=1, keepdim=True)
        .float()
    )
    got = out.float()
    expected = unique * ref
    eps = 3e-1 if fp8 else 8e-3
    hidden_ok = bool(
        ((got - expected).abs() <= eps * unique * ref.abs().clamp(min=1.0)).all()
    )
    weights_ok = (
        True
        if out_w is None
        else bool(((out_w.float() - unique * wts.float()).abs() <= 2e-3 * unique).all())
    )
    return hidden_ok and weights_ok


def run_cell(env, entry, proto, index, total):
    torch, H = env["torch"], env["harness"]
    dist_handle, comm, device = env["dist"], env["comm"], env["device"]
    EpDispatchCombineConfig, EpDispatchCombineOp = env["config"], env["op"]
    rank, npes = dist_handle.rank, dist_handle.world
    cell = fxcells.Cell.from_json(entry["cell"])
    record = {
        "schema": 1,
        "id": entry["id"],
        "stage": entry.get("stage"),
        "round": entry.get("round"),
        "arm": entry.get("arm"),
        "variant": entry["variant"],
        "cell": entry["cell"],
        "cell_key": cell.key,
        "table_dir": entry["table_dir"],
        "expect": entry.get("expect"),
        "launch": env["launch"],
        "host": socket.gethostname(),
        "protocol": proto,
        "t_start": time.time(),
    }
    if fxcells.EP_SIZE[cell.family] != npes:
        raise SystemExit(
            f"{cell.key} needs EP{fxcells.EP_SIZE[cell.family]}, launch has {npes} ranks"
        )

    os.environ["MORI_EP_V2_TUNING_DIR"] = entry["table_dir"]
    for name in ("MORI_EP_DISP_GEOM", "MORI_EP_COMB_GEOM"):
        os.environ.pop(name, None)
    dtypes = env["dtypes"]
    disp, comb = dtypes[cell.dispatch_dtype], dtypes[cell.combine_dtype]
    is_fp4 = cell.dispatch_dtype == "fp4"
    fp8 = cell.dispatch_dtype == "fp8_e4m3"
    op = None
    try:
        if disp is None:
            raise RuntimeError("this torch has no float4_e2m1fn_x2")
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=npes,
            hidden_dim=cell.shape.hidden,
            max_num_inp_token_per_rank=cell.tokens,
            num_experts_per_rank=cell.experts_per_rank,
            num_experts_per_token=cell.shape.topk,
            data_type=disp if disp is comb else torch.bfloat16,
            dispatch_data_type=None if disp is comb else disp,
            combine_data_type=None if disp is comb else comb,
            scale_dim=proto["scale_dim"],
            scale_type_size=4 if proto["scale_dim"] else 0,
            quant_type="none",
            gpu_per_node=env["gpu_per_node"],
            num_qp_per_pe=proto["num_qp_per_pe"],
            active_qp_counts=(
                tuple(entry["active_qp_counts"])
                if entry.get("active_qp_counts")
                else None
            ),
            internode_kernel=(
                cell.family if cell.family in fxcells.INTERNODE else "auto"
            ),
            kernel_backend="hip",
        )
        op = EpDispatchCombineOp(cfg, comm)
        comm.barrier()
        selected = op.kernel_config(cell.tokens)
        record["effective"] = {
            "dispatch": _spec(selected.dispatch),
            "combine": _spec(selected.combine),
        }

        rng = torch.Generator(device=device)
        rng.manual_seed(proto["seed"] + rank)
        gen_dtype = torch.bfloat16 if is_fp4 else disp
        inp, idx, wts, sc = H._generate_round(
            rng, cfg, cell.tokens, device, gen_dtype, cell.shape.routing
        )
        if is_fp4:
            inp, ref = env["fp4"].encode(inp)
            convert = env["fp4"].decode
        else:
            ref = inp.float()
            convert = (lambda t: t.to(comb)) if disp is not comb else (lambda t: t)

        verified = _verify(torch, op, cfg, comm, inp, ref, idx, wts, sc, convert, fp8)
        failed_ranks = dist_handle.allreduce_sum(0 if verified else 1)
        # Repeated checks on fresh inputs (verify_rounds > 1): an intermittent
        # wrong result shows up in some runs only.
        failed_checks = int(failed_ranks > 0)
        for i in range(1, proto.get("verify_rounds", 1)):
            vrng = torch.Generator(device=device)
            vrng.manual_seed(proto["seed"] + rank + 7919 * i)
            vinp, vidx, vwts, vsc = H._generate_round(
                vrng, cfg, cell.tokens, device, gen_dtype, cell.shape.routing
            )
            if is_fp4:
                vinp, vref = env["fp4"].encode(vinp)
            else:
                vref = vinp.float()
            ok = _verify(
                torch, op, cfg, comm, vinp, vref, vidx, vwts, vsc, convert, fp8
            )
            bad = dist_handle.allreduce_sum(0 if ok else 1)
            failed_checks += int(bad > 0)
            failed_ranks = max(failed_ranks, bad)
        record["verify_rounds"] = proto.get("verify_rounds", 1)
        record["failed_checks"] = failed_checks
        record["verified"] = failed_checks == 0
        record["failed_ranks"] = failed_ranks
        if failed_checks:
            record["status"] = "wrong"
        elif proto.get("timing") is False:
            record["status"] = "ok"
        else:
            captured = {}
            timing = SimpleNamespace(
                tuning_reps=proto["passes"],
                rounds=proto["rounds"],
                warmup=proto["warmup"],
                drop_rounds=proto["drop_rounds"],
                max_tokens=cell.tokens,
            )
            combine_weights = wts if proto.get("bench_weights") else None
            metrics = H._row_metrics(
                op,
                cfg,
                timing,
                dist_handle,
                (inp, idx, wts, sc, combine_weights),
                convert,
                cell.family == "v2_ll",
                evidence_callback=captured.update,
            )
            record["metrics"] = metrics
            record["metrics_unrounded"] = captured["metrics_unrounded"]
            record["stats"] = summarize(
                captured["rank_rows"],
                proto["passes"],
                proto["rounds"],
                proto["drop_rounds"],
            )
            if proto.get("keep_samples"):
                record["rank_rows"] = [
                    [round(v, 2) if isinstance(v, float) else v for v in row]
                    for row in captured["rank_rows"]
                ]
            record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - record it and move on to the next cell
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-4000:]
    finally:
        if op is not None:
            try:
                op.close()
            except Exception as exc:  # noqa: BLE001
                record.setdefault("close_error", f"{type(exc).__name__}: {exc}")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        comm.barrier()
    record["t_end"] = time.time()
    if rank == 0:
        _write_json(entry["evidence"], record)
        stats = record.get("stats") or {}
        eff = record.get("effective") or {}

        def g(phase):
            s = eff.get(phase) or {}
            return f"{s.get('block_num')}/{s.get('rdma_block_num')}/{s.get('warp_per_block')}q{s.get('active_qps')}{'LF' if s.get('load_first') else ''}"

        print(
            f"# CELL {index + 1}/{total} {record['status']} {cell.key} {entry['variant']} "
            f"D={stats.get('dispatch', {}).get('median', float('nan')):.1f} "
            f"C={stats.get('combine', {}).get('median', float('nan')):.1f} "
            f"gates={'ok' if stats.get('gates_ok') else 'FAIL' if stats else '-'} "
            f"eff=D{g('dispatch')} C{g('combine')} "
            f"{record['t_end'] - record['t_start']:.1f}s"
            + (f" {record.get('error')}" if record.get("error") else ""),
            flush=True,
        )
    return record["status"]


# ---------------------------------------------------------------- process tree


def worker(args):
    import torch

    harness = _load_harness()
    from mori.cco import Communicator
    from mori.ops.dispatch_combine_v2 import (
        EpDispatchCombineConfig,
        EpDispatchCombineOp,
    )

    manifest = json.loads(Path(args.manifest).read_text())
    proto, cells = manifest["protocol"], manifest["cells"]
    if args.num_qp is not None and args.num_qp != proto["num_qp_per_pe"]:
        raise SystemExit(
            f"--num-qp {args.num_qp} != manifest num_qp_per_pe {proto['num_qp_per_pe']}"
        )
    dist_handle = harness.Dist()
    rank, npes = dist_handle.rank, dist_handle.world
    device = torch.device("cuda", dist_handle.local_rank)
    gpu_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    env = {
        "torch": torch,
        "harness": harness,
        "dist": dist_handle,
        "device": device,
        "gpu_per_node": gpu_per_node,
        "config": EpDispatchCombineConfig,
        "op": EpDispatchCombineOp,
        "launch": manifest.get("launch_id"),
        "fp4": Fp4(torch, device) if hasattr(torch, "float4_e2m1fn_x2") else None,
        "dtypes": {
            "bf16": torch.bfloat16,
            "fp8_e4m3": torch.float8_e4m3fn,
            # Absent from older torch builds; only an FP4 cell needs it.
            "fp4": getattr(torch, "float4_e2m1fn_x2", None),
        },
    }
    max_tokens = max(c["cell"]["tokens"] for c in cells)
    max_hidden = max(c["cell"]["hidden"] for c in cells)
    # The harness's VMM budget for one op at the largest cell of this launch.
    win_bytes = npes * max(max_tokens, 64) * max_hidden * 4 * 2 + (1 << 24)
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = dist_handle.bcast_uid(uid)
    counts = {}
    with Communicator.init(
        npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)
    ) as comm:
        env["comm"] = comm
        comm.barrier()
        if rank == 0:
            print(
                f"# LAUNCH {manifest.get('launch_id')} cells={len(cells)} world={npes} "
                f"gpu_per_node={gpu_per_node} protocol={json.dumps(proto, sort_keys=True)}",
                flush=True,
            )
        for i, entry in enumerate(cells):
            status = run_cell(env, entry, proto, i, len(cells))
            counts[status] = counts.get(status, 0) + 1
    if rank == 0:
        print(
            f"# DONE {manifest.get('launch_id')} {json.dumps(counts, sort_keys=True)}",
            flush=True,
        )
    dist_handle.shutdown()
    return 0


def _spawn_entry(local_rank, argv, node_rank, num_nodes, ranks_per_node):
    """The harness's _spawn_entry: rewrite the rank environment, then run."""
    os.environ["_MORI_EP_SPAWN_CHILD"] = "1"
    os.environ["RANK"] = str(node_rank * ranks_per_node + local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(num_nodes * ranks_per_node)
    os.environ["LOCAL_WORLD_SIZE"] = str(ranks_per_node)
    code = worker(_parse(argv))
    if code:
        raise SystemExit(code)


def main(argv):
    args = _parse(argv)
    if args.spawn and not os.environ.get("_MORI_EP_SPAWN_CHILD"):
        import torch

        if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) > 1:
            raise SystemExit(
                "use --nproc_per_node=1: this entry spawns the ranks itself"
            )
        torch.multiprocessing.spawn(
            _spawn_entry,
            args=(
                argv,
                int(os.environ["RANK"]),
                int(os.environ["WORLD_SIZE"]),
                args.spawn,
            ),
            nprocs=args.spawn,
            join=True,
        )
        return 0
    return worker(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

#### `fxanalyze.py`

```python
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Selection rules over driver evidence. Pure Python, host side.

The rules are qp_status.md's:

* A measurement counts only if the cell verified, passed the README noise
  gates, and ran the configuration its variant table names (kernel_config()
  in effect == the table's, every phase, every field).
* A run whose dispatch or combine median is more than 2x another run of the same
  variant and cell is a stall (qp_status.md §3, §7.2); it is dropped and the
  cell re-measured.
* Geometry: a candidate is eligible only if measured in both rounds. It replaces
  the seed only if it is >= 1% faster on the mean over rounds and faster in
  every round. Dispatch rules are chosen per (dtype, bucket) on that cell's
  D+C; a combine rule on the D+C summed over every cell that shares it.
* QP / load-first: the factor model of §1 -- D(d, c) per cell, C(c, LF) per cell
  -- and per bucket the (d per dtype, c, LF) with the lowest predicted sum,
  replacing (2, 2, the bucket's existing load-first) only if >= 1% faster and
  faster in every round.
* Whole-table A/B: per bucket, the D+C summed over the bucket's dtypes; the new
  rules are taken only if >= 1% faster and faster in every round.

New here: round 2 is pruned per cell (candidates within `window` of that cell's
best in round 1, at least `min_keep`), instead of keeping a candidate for every
cell as soon as it is near-best in any one.
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

from fxcells import INTERNODE, Cell, Group

FIELDS = (
    "family",
    "block_num",
    "rdma_block_num",
    "warp_per_block",
    "active_qps",
    "load_first",
)
STALL_RATIO = 2.0


# ---------------------------------------------------------------- evidence


def load_records(directory) -> list:
    out = []
    for path in sorted(Path(directory).rglob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rec["_path"] = str(path)
        out.append(rec)
    return out


def effective_matches(rec) -> bool:
    exp, eff = rec.get("expect"), rec.get("effective")
    if not exp or not eff:
        return False
    return all(
        exp[phase].get(f) == eff[phase].get(f)
        for phase in ("dispatch", "combine")
        for f in FIELDS
    )


def problems(rec) -> list:
    out = []
    if rec.get("status") != "ok":
        out.append(rec.get("status", "missing"))
    if rec.get("status") == "ok" and not rec.get("verified"):
        out.append("not-verified")
    stats = rec.get("stats") or {}
    timed = (rec.get("protocol") or {}).get("timing", True) is not False
    if rec.get("status") == "ok" and timed and not stats.get("gates_ok"):
        failed = [k for k, v in (stats.get("gates") or {}).items() if not v]
        out.append("gates:" + ",".join(failed))
    if rec.get("status") == "ok" and not effective_matches(rec):
        out.append("config-mismatch")
    return out


def is_valid(rec) -> bool:
    return not problems(rec)


def value(rec) -> float:
    """The per-measurement statistic: D median + C median (qp_status.md §3)."""
    return rec["stats"]["total_median"]


def find_stalls(records) -> list:
    """Runs of one (stage, variant, cell) whose dispatch or combine median is
    more than STALL_RATIO x another run's."""
    by = collections.defaultdict(list)
    for rec in records:
        if is_valid(rec):
            by[(rec.get("stage"), rec["variant"], rec["cell_key"])].append(rec)
    stalled = []
    for runs in by.values():
        if len(runs) < 2:
            continue
        for phase in ("dispatch", "combine"):
            low = min(r["stats"][phase]["median"] for r in runs)
            stalled += [
                r for r in runs if r["stats"][phase]["median"] > STALL_RATIO * low
            ]
    seen, out = set(), []
    for r in stalled:
        if r["_path"] not in seen:
            seen.add(r["_path"])
            out.append(r)
    return out


def find_block_stalls(records, ratio=STALL_RATIO, run_ratio=4.0, run_length=3) -> list:
    """Stalls inside one cell's block of a geometry round, where most variants
    are measured once and find_stalls() has nothing to compare against:

    * the block's two seed runs (first and last) disagree by more than `ratio`
      in either phase -- the whole block is measured again;
    * `run_length` or more consecutive runs (in time) each over `run_ratio` x
      the block median -- the slow mode qp_status.md §3 describes takes several
      consecutive cells of one launch, which shuffled candidates rarely do.
    """
    blocks = collections.defaultdict(list)
    for rec in records:
        if rec.get("stage") == "g" and is_valid(rec):
            blocks[(rec.get("round"), rec["cell_key"])].append(rec)
    out = []
    for runs in blocks.values():
        seeds = [r for r in runs if r["variant"] in ("seed", "seed-end")]
        if len(seeds) == 2 and any(
            max(s["stats"][ph]["median"] for s in seeds)
            > ratio * min(s["stats"][ph]["median"] for s in seeds)
            for ph in ("dispatch", "combine")
        ):
            out += runs
            continue
        runs = sorted(runs, key=lambda r: r.get("t_start", 0))
        med = statistics.median(value(r) for r in runs)
        streak = []
        for r in runs + [None]:
            if r is not None and value(r) > run_ratio * med:
                streak.append(r)
                continue
            if len(streak) >= run_length:
                out += streak
            streak = []
    return out


def _shape_tag(cell_json):
    from fxcells import Shape

    return Shape(
        cell_json["num_experts"],
        cell_json["topk"],
        cell_json["hidden"],
        cell_json.get("routing", "uniform"),
    ).tag


def wrong_configs(records) -> set:
    """Configurations that produced a wrong result anywhere: {(family, shape
    tag, variant)}. A geometry or QP variant that is wrong once on a shape is
    not eligible anywhere on that shape -- the failures seen on MI355X were
    intermittent (1 rank in 16, a few runs in 30), so a run that passed does
    not clear it."""
    out = set()
    for rec in records:
        if rec.get("status") == "wrong" and rec["variant"] not in (
            "seed",
            "seed-end",
            "base",
        ):
            out.add((rec["cell"]["family"], _shape_tag(rec["cell"]), rec["variant"]))
    return out


def wrong_cells(records, variant) -> set:
    """Cell keys where `variant` produced a wrong result."""
    return {
        r["cell_key"]
        for r in records
        if r.get("status") == "wrong" and r["variant"] == variant
    }


def index(records):
    """{(round, variant, cell_key): record} of valid records; later files win."""
    out = {}
    for rec in records:
        if is_valid(rec):
            out[(str(rec.get("round")), rec["variant"], rec["cell_key"])] = rec
    return out


# ---------------------------------------------------------------- geometry


def geometry_variant_id(family: str, phase: str, geometry) -> str:
    return f"geo-{family}-{phase}-" + "-".join(str(x) for x in geometry)


def seed_value(idx, rnd, cell_key):
    """Mean of the round's seed measurements of a cell (seed, seed-end)."""
    vals = [
        value(idx[(rnd, v, cell_key)])
        for v in ("seed", "seed-end")
        if (rnd, v, cell_key) in idx
    ]
    return statistics.fmean(vals) if vals else None


def _rule_cells(group: Group, phase: str):
    """{rule selector: [cells reading that rule]} for the group's owned rules."""
    out = collections.defaultdict(list)
    if phase not in group.owns:
        return out
    for cell in group.cells():
        out[cell.selector(phase)].append(cell)
    return out


def _sum_over(idx, rnd, variant, cells):
    vals = []
    for cell in cells:
        rec = idx.get((rnd, variant, cell.key))
        if rec is None:
            return None
        vals.append(value(rec))
    return sum(vals)


def _seed_sum(idx, rnd, cells):
    vals = [seed_value(idx, rnd, c.key) for c in cells]
    return None if any(v is None for v in vals) else sum(vals)


def prune(
    idx, groups, candidates, rnd="1", window=0.05, min_keep=5, wrong=frozenset()
) -> dict:
    """Round-2 candidate lists: {family: {phase: {selector(str): [geometry]}}}.
    A rule keeps the candidates within `window` of its best round-1 value (the
    seed counts as a candidate), at least `min_keep` of them. Configurations in
    `wrong` (wrong_configs()) are never kept."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for group in groups:
        for phase in group.owns:
            for sel, cells in _rule_cells(group, phase).items():
                scored = []
                for geom in candidates[group.family][phase]:
                    vid = geometry_variant_id(group.family, phase, geom)
                    if (group.family, group.shape.tag, vid) in wrong:
                        continue
                    v = _sum_over(idx, rnd, vid, cells)
                    if v is not None:
                        scored.append((v, list(geom)))
                seed = _seed_sum(idx, rnd, cells)
                pool = sorted(scored)
                best = min(
                    [v for v, _ in pool] + ([seed] if seed is not None else []),
                    default=None,
                )
                if best is None:
                    keep = [g for _, g in pool]
                else:
                    keep = [g for v, g in pool if v <= best * (1 + window)]
                    for _, g in pool:
                        if len(keep) >= min_keep:
                            break
                        if g not in keep:
                            keep.append(g)
                out[group.family][phase][json.dumps(list(sel))] = keep
    return {f: dict(p) for f, p in out.items()}


def select_geometry(
    idx, groups, lists, rounds=("1", "2"), margin=0.01, wrong=frozenset()
) -> list:
    """One pick per owned rule: {family, selector, phase, geometry, seed?,
    scores}. `lists` is prune()'s output: the round-2 candidates per rule.
    Configurations in `wrong` are not eligible."""
    picks = []
    for group in groups:
        for phase in group.owns:
            for sel, cells in _rule_cells(group, phase).items():
                seed_by_round = {r: _seed_sum(idx, r, cells) for r in rounds}
                if any(v is None for v in seed_by_round.values()):
                    picks.append(
                        {
                            "family": group.family,
                            "phase": phase,
                            "selector": list(sel),
                            "geometry": None,
                            "reason": "seed not measured in every round",
                        }
                    )
                    continue
                best = None
                for geom in lists[group.family][phase].get(json.dumps(list(sel)), []):
                    vid = geometry_variant_id(group.family, phase, geom)
                    if (group.family, group.shape.tag, vid) in wrong:
                        continue
                    per_round = {r: _sum_over(idx, r, vid, cells) for r in rounds}
                    if any(v is None for v in per_round.values()):
                        continue
                    score = statistics.fmean(per_round.values())
                    if best is None or score < best[0]:
                        best = (score, geom, per_round)
                seed_score = statistics.fmean(seed_by_round.values())
                pick = {
                    "family": group.family,
                    "phase": phase,
                    "selector": list(sel),
                    "seed_by_round": seed_by_round,
                    "seed_score": seed_score,
                }
                if best is not None:
                    score, geom, per_round = best
                    every = all(per_round[r] < seed_by_round[r] for r in rounds)
                    pick.update(
                        best_geometry=geom,
                        best_by_round=per_round,
                        best_score=score,
                        change=score / seed_score - 1,
                    )
                    if score <= (1 - margin) * seed_score and every:
                        pick["geometry"] = geom
                        pick["reason"] = "faster"
                    else:
                        pick["geometry"] = None
                        pick["reason"] = "not >=1% faster in every round"
                else:
                    pick["geometry"] = None
                    pick["reason"] = "no candidate measured in every round"
                picks.append(pick)
    return picks


# ---------------------------------------------------------------- QP / LF


def qp_variant_id(family: str, d: int, c: int, load_first: bool) -> str:
    return f"qp-{family}-d{d}-c{c}" + ("-lf" if load_first else "")


def qp_grid(family: str, counts, with_lf: bool = True) -> list:
    """(d, c, lf) variants: every (d, c), plus (q, q, LF) on InterNodeV2."""
    grid = [(d, c, False) for d in counts for c in counts]
    if with_lf and family == "v2":
        grid += [(q, q, True) for q in counts]
    return grid


def _phase_median(rec, phase):
    return rec["stats"][phase]["median"]


def factor_model(idx, cell: Cell, grid, rnd):
    """D(d, c) and C(c, lf) of one cell in one round, each averaged over the
    variants that measured it (dispatch does not see load-first; combine does
    not see the dispatch count)."""
    dvals, cvals = collections.defaultdict(list), collections.defaultdict(list)
    for d, c, lf in grid:
        rec = idx.get((rnd, qp_variant_id(cell.family, d, c, lf), cell.key))
        if rec is None:
            continue
        dvals[(d, c)].append(_phase_median(rec, "dispatch"))
        cvals[(c, lf)].append(_phase_median(rec, "combine"))
    return (
        {k: statistics.fmean(v) for k, v in dvals.items()},
        {k: statistics.fmean(v) for k, v in cvals.items()},
    )


def model_consistency(idx, cells, grid, rounds) -> dict:
    """How well the factor model holds: spread of D(d, c) across load-first and
    of C(c, lf) across d, as max/min - 1 over the variants that measured it."""
    d_spread, c_spread = [], []
    for cell in cells:
        for rnd in rounds:
            dvals, cvals = collections.defaultdict(list), collections.defaultdict(list)
            for d, c, lf in grid:
                rec = idx.get((rnd, qp_variant_id(cell.family, d, c, lf), cell.key))
                if rec is not None:
                    dvals[(d, c)].append(_phase_median(rec, "dispatch"))
                    cvals[(c, lf)].append(_phase_median(rec, "combine"))
            d_spread += [max(v) / min(v) - 1 for v in dvals.values() if len(v) > 1]
            c_spread += [max(v) / min(v) - 1 for v in cvals.values() if len(v) > 1]

    def summary(values):
        if not values:
            return None
        values = sorted(values)
        return {
            "n": len(values),
            "median": values[len(values) // 2],
            "p90": values[int(0.9 * (len(values) - 1))],
            "max": values[-1],
        }

    return {"dispatch_vs_lf": summary(d_spread), "combine_vs_d": summary(c_spread)}


def qp_groups(groups) -> list:
    """Groups whose active QP counts a QP stage tunes: internode groups owning
    both phases. (308x likewise kept its symmetric-FP8 combine rows at their
    earlier counts, and tuned the FP4 dispatch count with its geometry.)"""
    return [
        g
        for g in groups
        if g.family in INTERNODE and set(g.owns) == {"dispatch", "combine"}
    ]


def select_qp(
    idx,
    groups,
    counts,
    base_lf,
    rounds=("1", "2"),
    margin=0.01,
    with_lf=True,
    wrong=frozenset(),
) -> list:
    """One pick per (family, shape, bucket): {dispatch: {cell key: d}, combine:
    c, load_first} or no change. `base_lf(cell)` is the bucket's load-first
    before the campaign; the baseline every pick has to beat is (2, 2, that).
    A (d, c, lf) variant in `wrong` is left out of the model for that shape."""
    picks = []
    for group in qp_groups(groups):
        grid = [
            (d, c, lf)
            for d, c, lf in qp_grid(group.family, counts, with_lf)
            if (group.family, group.shape.tag, qp_variant_id(group.family, d, c, lf))
            not in wrong
        ]
        lf_options = [False, True] if (group.family == "v2" and with_lf) else [False]
        by_tokens = collections.defaultdict(list)
        for cell in group.cells():
            by_tokens[cell.tokens].append(cell)
        for tokens, cells in sorted(by_tokens.items()):
            models = {
                r: [factor_model(idx, cell, grid, r) for cell in cells] for r in rounds
            }

            def predict(rnd, d_by_cell, c, lf):
                total = 0.0
                for cell, (dm, cm) in zip(cells, models[rnd]):
                    key_d, key_c = (d_by_cell[cell.key], c), (c, lf)
                    if key_d not in dm or key_c not in cm:
                        return None
                    total += dm[key_d] + cm[key_c]
                return total

            def best_d(c):
                """Per cell, the dispatch count with the lowest D at combine count c
                over the rounds (D does not depend on load-first)."""
                choice = {}
                for i, cell in enumerate(cells):
                    scored = []
                    for d in counts:
                        vals = [models[r][i][0].get((d, c)) for r in rounds]
                        if all(v is not None for v in vals):
                            scored.append((statistics.fmean(vals), d))
                    if not scored:
                        return None
                    choice[cell.key] = min(scored)[1]
                return choice

            lf0 = bool(base_lf(cells[0])) and group.family == "v2"
            base_d = {cell.key: 2 for cell in cells}
            base_by_round = {r: predict(r, base_d, 2, lf0) for r in rounds}
            pick = {
                "family": group.family,
                "shape": group.shape.tag,
                "tokens": tokens,
                "baseline": {"dispatch": base_d, "combine": 2, "load_first": lf0},
                "baseline_by_round": base_by_round,
                "choice": None,
            }
            if any(v is None for v in base_by_round.values()):
                pick["reason"] = "baseline not measured in every round"
                picks.append(pick)
                continue
            best = None
            for c in counts:
                d_choice = best_d(c)
                if d_choice is None:
                    continue
                for lf in lf_options:
                    per_round = {r: predict(r, d_choice, c, lf) for r in rounds}
                    if any(v is None for v in per_round.values()):
                        continue
                    score = statistics.fmean(per_round.values())
                    if best is None or score < best[0]:
                        best = (
                            score,
                            {"dispatch": d_choice, "combine": c, "load_first": lf},
                            per_round,
                        )
            base_score = statistics.fmean(base_by_round.values())
            if best is None:
                pick["reason"] = "no configuration measured in every round"
            else:
                score, choice, per_round = best
                every = all(per_round[r] < base_by_round[r] for r in rounds)
                pick.update(
                    best=choice, best_by_round=per_round, change=score / base_score - 1
                )
                if score <= (1 - margin) * base_score and every:
                    pick.update(choice=choice, reason="faster")
                else:
                    pick["reason"] = "not >=1% faster in every round"
            picks.append(pick)
    return picks


# ---------------------------------------------------------------- A/B


def ab_buckets(
    idx, groups, arm_a, arm_b, rounds=("1", "2", "3"), margin=0.01, wrong_b=frozenset()
) -> list:
    """Per (family, shape, tokens): D+C summed over the bucket's cells in each
    round, arm B against arm A, and whether B is >= margin faster on the mean
    and faster in every round. A bucket where arm B gave a wrong result on any
    cell (`wrong_b`, cell keys) is never taken."""
    out = []
    for group in groups:
        by_tokens = collections.defaultdict(list)
        for cell in group.cells():
            by_tokens[cell.tokens].append(cell)
        for tokens, cells in sorted(by_tokens.items()):
            a = {r: _sum_over(idx, r, arm_a, cells) for r in rounds}
            b = {r: _sum_over(idx, r, arm_b, cells) for r in rounds}
            row = {
                "family": group.family,
                "shape": group.shape.tag,
                "tokens": tokens,
                "a_by_round": a,
                "b_by_round": b,
            }
            if any(c.key in wrong_b for c in cells):
                row.update(take=False, reason=f"{arm_b} gave a wrong result")
            elif any(v is None for v in list(a.values()) + list(b.values())):
                row.update(take=False, reason="missing rounds")
            else:
                ratio = statistics.median(b[r] / a[r] for r in rounds)
                mean_change = (
                    statistics.fmean(b.values()) / statistics.fmean(a.values()) - 1
                )
                every = all(b[r] < a[r] for r in rounds)
                row.update(
                    median_ratio=ratio,
                    change=mean_change,
                    every_round=every,
                    take=mean_change <= -margin and every,
                )
                per_dtype = {}
                for cell in cells:
                    ra = [idx[(r, arm_a, cell.key)] for r in rounds]
                    rb = [idx[(r, arm_b, cell.key)] for r in rounds]
                    per_dtype[cell.dispatch_dtype] = {
                        "a": statistics.median(value(x) for x in ra),
                        "b": statistics.median(value(x) for x in rb),
                        "change": statistics.median(
                            value(y) / value(x) for x, y in zip(ra, rb)
                        )
                        - 1,
                    }
                row["per_dtype"] = per_dtype
            out.append(row)
    return out


def landing_metrics(idx, cell: Cell, arm: str, phase: str, rounds=("1", "2", "3")):
    """The table metric columns for one rule from the run of `arm` at `cell`
    whose D+C is the median of the rounds that passed the gates."""
    runs = [idx[(r, arm, cell.key)] for r in rounds if (r, arm, cell.key) in idx]
    if not runs:
        return None
    runs.sort(key=value)
    m = runs[len(runs) // 2]["metrics"][phase]
    if cell.family == "intra":
        return {
            "avg_latency_us": m["avg_latency_us"],
            "bandwidth_gbps": m["avg_xgmi_bandwidth_gbps"],
            "avg_rdma_bandwidth_gbps": None,
            "avg_xgmi_bandwidth_gbps": m["avg_xgmi_bandwidth_gbps"],
            "avg_ll_bandwidth_gbps": None,
            "bandwidth_metric": m["bandwidth_metric"],
        }
    return {
        k: m[k]
        for k in (
            "avg_latency_us",
            "bandwidth_gbps",
            "avg_rdma_bandwidth_gbps",
            "avg_xgmi_bandwidth_gbps",
            "avg_ll_bandwidth_gbps",
            "bandwidth_metric",
        )
    }


def metric_cell(group: Group, phase: str, selector: tuple):
    """The cell whose run supplies a rule's metric columns: a dispatch rule's
    own cell; for a combine rule the bf16-dispatch cell when there is one
    (README: FP8 and FP4 dispatch rows are measured with the BF16 combine rule)."""
    cells = [c for c in group.cells() if c.selector(phase) == selector]
    if phase == "combine":
        bf16 = [c for c in cells if c.dispatch_dtype == "bf16"]
        return (bf16 or cells)[0]
    return cells[0]
```

#### `fxrun.py`

```python
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Launch driver manifests on the MI355X nodes. Host side, no torch.

The node pair comes from fxconfig (FX_* settings, $FX_WORKDIR/machine.env).
With FX_CI_FREEZE=1 the GitHub runner on node0 is frozen between two jobs
(SIGSTOP of Runner.Listener) and resumed afterwards. A launch only starts once
neither node has a GPU user outside our container (KFD scan) for `quiet_s`; a
watcher samples both nodes during the launch. After a launch:

* cells whose evidence finished after the last clean sample before the first
  foreign process are moved to quarantine/interference and run again
  (qp_status.md §8.3);
* a launch is cut short as soon as the watcher sees a foreign process, and the
  cells it did not reach are retried in the next launch;
* every launch only carries cells with no evidence yet, so an interrupted
  stage resumes where it stopped.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import fxconfig

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "fxdriver.py"


def log(msg):
    stamp = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


@dataclasses.dataclass
class Machine:
    """The node pair, from fxconfig (FX_* settings, $FX_WORKDIR/machine.env)."""

    repo: str = dataclasses.field(default_factory=lambda: str(fxconfig.REPO))
    container: str = dataclasses.field(
        default_factory=lambda: fxconfig.get("FX_CONTAINER", "mori_ep_tuning")
    )
    node0_label: str = dataclasses.field(
        default_factory=lambda: fxconfig.get("FX_NODE0_LABEL", "node0")
    )
    peer: str = dataclasses.field(default_factory=lambda: fxconfig.get("FX_PEER"))
    peer_label: str = dataclasses.field(
        default_factory=lambda: fxconfig.get("FX_PEER_LABEL", "node1")
    )
    peer_key_alias: str = dataclasses.field(
        default_factory=lambda: fxconfig.get("FX_PEER_KEY_ALIAS")
    )
    master_addr: str = dataclasses.field(
        default_factory=lambda: fxconfig.get("FX_MASTER_ADDR")
    )
    ifname: str = dataclasses.field(default_factory=lambda: fxconfig.get("FX_IFNAME"))
    ci_freeze: bool = dataclasses.field(
        default_factory=lambda: fxconfig.get("FX_CI_FREEZE") == "1"
    )
    runner_diag: str = dataclasses.field(
        default_factory=lambda: fxconfig.get("FX_RUNNER_DIAG", "")
    )
    sync: str = dataclasses.field(default_factory=lambda: fxconfig.get("FX_SYNC", ""))
    run_env: dict = dataclasses.field(default_factory=fxconfig.run_env)
    unset: tuple = (
        "MORI_EP_TUNING_CONFIG",
        "MORI_EP_DISP_GEOM",
        "MORI_EP_COMB_GEOM",
        "MORI_EP_ROUND_SERIES",
        "MORI_EP_V2_TUNING_DIR",
    )

    @property
    def ssh(self):
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if self.peer_key_alias:
            cmd += ["-o", f"HostKeyAlias={self.peer_key_alias}"]
        return cmd + [self.peer]

    def require_ep16(self):
        missing = [
            name
            for name, value in (
                ("FX_PEER", self.peer),
                ("FX_MASTER_ADDR", self.master_addr),
                ("FX_IFNAME", self.ifname),
            )
            if not value
        ]
        if missing:
            raise SystemExit(
                f"EP16 needs {', '.join(missing)} (see machine.env.example; "
                f"settings are read from {fxconfig.WORKDIR / 'machine.env'})"
            )

    def sync_to_peer(self, paths):
        """rsync each path to the same place on node1 (FX_SYNC=rsync)."""
        if self.sync != "rsync":
            return
        rsh = " ".join(shlex.quote(a) for a in self.ssh[:-1])
        for path in paths:
            path = Path(path)
            target = path if path.is_dir() else path.parent
            sh(self.ssh + [f"mkdir -p {shlex.quote(str(target))}"])
            src = f"{path}/" if path.is_dir() else str(path)
            dst = f"{self.peer}:{path}/" if path.is_dir() else f"{self.peer}:{path}"
            r = subprocess.run(
                ["rsync", "-a", "-e", rsh, src, dst],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if r.returncode:
                raise SystemExit(
                    f"rsync {path} to {self.peer} failed: {r.stderr.strip()[-400:]}"
                )


def sh(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- CI runner


class CiRunner:
    """The GitHub runner on node0 (refresh_driver.py freeze_ci / resume_ci)."""

    def __init__(self, machine: Machine):
        self.diag = (
            Path(machine.runner_diag) if machine.runner_diag else self._find_diag()
        )

    def _find_diag(self):
        """<runner>/_diag of the running listener, <runner>/bin/Runner.Listener."""
        pid = self.listener_pid()
        if pid is None:
            return None
        try:
            return Path(os.readlink(f"/proc/{pid}/exe")).parent.parent / "_diag"
        except OSError:
            return None

    @staticmethod
    def listener_pid():
        out = sh(["pgrep", "-f", "Runner.Listener run"]).stdout.split()
        return int(out[0]) if out else None

    @staticmethod
    def worker_running():
        return bool(sh(["pgrep", "-f", "Runner.Worker"]).stdout.strip())

    @staticmethod
    def proc_state(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ")[1].split()[0]
        except OSError:
            return None

    def last_job_state(self):
        if self.diag is None:
            return None
        logs = sorted(self.diag.glob("Runner_*.log"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return None
        tail = sh(["tail", "-n", "400", str(logs[-1])]).stdout
        states = re.findall(r"JobState: (\w+)", tail)
        return states[-1] if states else None

    def frozen(self):
        pid = self.listener_pid()
        return pid is not None and self.proc_state(pid) == "T"

    def freeze(self, max_wait_s):
        """SIGSTOP the listener between two jobs. True if this call froze it."""
        pid = self.listener_pid()
        if pid is None:
            raise SystemExit("no Runner.Listener on this host")
        if self.proc_state(pid) == "T":
            return False
        if self.diag is None or not self.diag.is_dir():
            raise SystemExit(
                f"runner _diag directory not found ({self.diag}); set FX_RUNNER_DIAG"
            )
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            if not self.worker_running() and self.last_job_state() == "Online":
                os.kill(pid, signal.SIGSTOP)
                time.sleep(1.5)
                if not self.worker_running() and self.last_job_state() == "Online":
                    log(f"froze Runner.Listener {pid} between jobs")
                    return True
                os.kill(pid, signal.SIGCONT)
                log("a job arrived while freezing; resumed and waiting again")
            time.sleep(0.3)
        raise SystemExit(f"runner never idle within {max_wait_s}s; nothing frozen")

    def resume(self):
        pid = self.listener_pid()
        if pid is not None:
            os.kill(pid, signal.SIGCONT)
            time.sleep(1)
            log(f"resumed Runner.Listener {pid}: state={self.proc_state(pid)}")


# ---------------------------------------------------------------- GPU users


KFD_SCAN = r"""
own=$(readlink /proc/$(podman inspect -f '{{.State.Pid}}' %s 2>/dev/null)/ns/pid 2>/dev/null)
for p in $(ls /sys/class/kfd/kfd/proc 2>/dev/null); do
  ns=$(readlink /proc/$p/ns/pid 2>/dev/null) || ns=unreadable
  [ -n "$own" ] && [ "$ns" = "$own" ] && continue
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-160)
  # An exiting process keeps its KFD entry after its namespaces and cmdline are gone.
  [ "$ns" = unreadable ] && [ -z "$cmd" ] && continue
  echo "$p $cmd"
done
"""


class Nodes:
    def __init__(self, machine: Machine, two_nodes: bool):
        self.m = machine
        self.two = two_nodes
        self.scan = KFD_SCAN % machine.container
        self.cleanup_cmd = (
            "podman exec %s bash -c \"pkill -9 -f '[f]xdriver.py|[t]est_dispatch_combine_v2_internode|"
            "[m]ultiprocessing.spawn|[t]orchrun|[r]un_internode_test' ; sleep 2 ; true\""
            % machine.container
        )

    def peer(self, cmd, timeout=60):
        return sh(self.m.ssh + [cmd], timeout=timeout)

    def foreign(self):
        local = sh(["bash", "-c", self.scan]).stdout.strip().splitlines()
        out = [f"{self.m.node0_label}:{x}" for x in local]
        if self.two:
            remote = self.peer(self.scan).stdout.strip().splitlines()
            out += [f"{self.m.peer_label}:{x}" for x in remote]
        return out

    def cleanup(self):
        sh(["bash", "-c", self.cleanup_cmd])
        if self.two:
            self.peer(self.cleanup_cmd)

    def container_up(self):
        def up(result):
            return result.returncode == 0 and result.stdout.strip() == "true"

        inspect = f"podman inspect -f '{{{{.State.Running}}}}' {self.m.container}"
        ok = up(sh(["bash", "-c", inspect]))
        if self.two:
            ok = ok and up(self.peer(inspect))
        return ok

    def wait_idle(self, last_busy, quiet_s=120, max_wait_s=6 * 3600, poll_s=5):
        """Block until neither node has had a foreign GPU user for quiet_s since
        `last_busy`. Returns the updated last-busy time, or None on timeout."""
        start = time.time()
        reported = 0.0
        while time.time() - start < max_wait_s:
            busy = self.foreign()
            now = time.time()
            if busy:
                last_busy = now
                if now - reported > 120:
                    log(f"  waiting for GPUs: {busy[0][:140]} (+{len(busy) - 1} more)")
                    reported = now
            elif now - last_busy >= quiet_s:
                return last_busy
            time.sleep(poll_s)
        return None


class Watch(threading.Thread):
    """Samples both nodes while a launch runs; keeps (time, foreign items)."""

    def __init__(self, nodes: Nodes, ci: CiRunner, period=2.0):
        super().__init__(daemon=True)
        self.nodes, self.ci, self.period = nodes, ci, period
        # Only a runner this tool froze is watched; another one's jobs show up
        # as foreign GPU users anyway.
        self.has_ci = nodes.m.ci_freeze and ci.listener_pid() is not None
        self.samples = []
        self.stop_flag = threading.Event()
        self.dirty = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            t = time.time()
            try:
                items = list(self.nodes.foreign())
                if self.has_ci and not self.ci.frozen():
                    items.append(f"{self.nodes.m.node0_label}:ci-listener-not-frozen")
                if self.has_ci and self.ci.worker_running():
                    items.append(f"{self.nodes.m.node0_label}:ci-worker-running")
            except (
                Exception
            ) as exc:  # noqa: BLE001 - a failed probe must not end the run
                items = [f"probe-error:{exc}"]
            self.samples.append((t, items))
            if items:
                self.dirty.set()
            self.stop_flag.wait(self.period)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=15)

    def clean_until(self):
        """Time of the last clean sample before the first dirty one, or None if
        the launch stayed clean."""
        last_clean = None
        for t, items in self.samples:
            if items:
                return last_clean if last_clean is not None else 0.0
            last_clean = t
        return None


# ---------------------------------------------------------------- launches


def _inner(m: Machine, body: str, extra_env=None) -> str:
    env = dict(m.run_env, **(extra_env or {}))
    assign = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    unset = " ".join(f"-u {k}" for k in m.unset)
    return f"cd {shlex.quote(m.repo)} && env {unset} {assign} {body}"


def ep16_commands(
    m: Machine,
    manifest: Path,
    port: int,
    timeout_s: int,
    logs,
    num_qp: int,
    max_tokens: int,
    extra_env=None,
):
    cmds = []
    for rank in (0, 1):
        launcher = [
            "tools/run_internode_test.sh",
            "--rank",
            str(rank),
            "--master-addr",
            m.master_addr,
            "--master-port",
            str(port),
            "--ifname",
            m.ifname,
            "--entry",
            str(DRIVER),
            "--cmd",
            "test",
            "--num-qp",
            str(num_qp),
            "--max-tokens",
            str(max_tokens),
            "--",
            "--manifest",
            str(manifest),
        ]
        body = (
            f"MORI_INTERNODE_TIMEOUT={timeout_s} "
            + " ".join(shlex.quote(a) for a in launcher)
            + f" > {shlex.quote(str(logs[rank]))} 2>&1"
        )
        inner = [
            "podman",
            "exec",
            m.container,
            "bash",
            "-c",
            _inner(m, body, extra_env),
        ]
        cmds.append(
            inner if rank == 0 else m.ssh + [" ".join(shlex.quote(a) for a in inner)]
        )
    return cmds


def ep8_command(
    m: Machine, manifest: Path, port: int, timeout_s: int, log_path, extra_env=None
):
    body = (
        f"timeout {timeout_s} torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 "
        f"--master_port={port} {shlex.quote(str(DRIVER))} --manifest {shlex.quote(str(manifest))}"
        f" > {shlex.quote(str(log_path))} 2>&1"
    )
    return ["podman", "exec", m.container, "bash", "-c", _inner(m, body, extra_env)]


def _evidence_done(entry) -> bool:
    path = Path(entry["evidence"])
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text())
    except ValueError:
        return False
    # A wrong result is deterministic for a configuration; an error is retried.
    return rec.get("status") in ("ok", "wrong")


def quarantine(paths, root: Path, reason: str):
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        dest = (
            root
            / reason
            / datetime.datetime.now().strftime("%m%d_%H%M%S")
            / p.parent.name
        )
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest / p.name))


def _port_free(port) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


class Runner:
    # Master ports stay below the nodes' ephemeral range (32768-60999): a port in
    # it can be taken by any outgoing connection, and torchrun's store then fails
    # with EADDRINUSE while the other node waits in its rendezvous.
    PORT_LOW, PORT_HIGH = 24000, 29998

    def __init__(
        self,
        machine: Machine,
        workdir: Path,
        two_nodes: bool,
        dry_run=False,
        quiet_s=120,
        ci_wait=3600,
    ):
        self.m = machine
        self.workdir = Path(workdir)
        self.two = two_nodes
        self.dry_run = dry_run
        self.quiet_s = quiet_s
        self.ci_wait = ci_wait
        self.port = self.PORT_LOW + (os.getpid() % 1000) * 4
        self.nodes = Nodes(machine, two_nodes)
        self.ci = CiRunner(machine)
        self.we_froze = False
        # The first launch waits for a full quiet window; later ones only after
        # a foreign user was seen (qp_status.md §7.4, fx_last_busy).
        self.last_busy_file = self.workdir / "last_busy"
        try:
            self.last_busy = float(self.last_busy_file.read_text())
        except (OSError, ValueError):
            self.last_busy = time.time()

    def __enter__(self):
        if self.two:
            self.m.require_ep16()
        if not self.dry_run:
            if not self.nodes.container_up():
                raise SystemExit(
                    f"container {self.m.container} is not running on every node "
                    f"(fxcampaign.py make-container, or podman start)"
                )
            if self.m.ci_freeze and self.ci.listener_pid():
                self.we_froze = self.ci.freeze(self.ci_wait)
        return self

    def __exit__(self, *exc):
        if self.we_froze:
            self.ci.resume()
        return False

    def run(
        self,
        entries,
        protocol,
        label,
        chunk_cells=120,
        seconds_per_cell=20,
        max_attempts=3,
        extra_env=None,
    ):
        """Measure every entry without evidence. Entries of one cell stay in one
        launch (A/B pairs, a cell's candidate block)."""
        attempts = {}
        while True:
            pending = [
                e
                for e in entries
                if not _evidence_done(e) and attempts.get(e["id"], 0) < max_attempts
            ]
            if not pending:
                break
            chunk = _take_chunk(pending, chunk_cells)
            for e in chunk:
                attempts[e["id"]] = attempts.get(e["id"], 0) + 1
            stuck = self._launch(chunk, protocol, label, seconds_per_cell, extra_env)
            if stuck is not None:
                # A cell that hung the launch is not retried as often.
                attempts[stuck["id"]] = attempts.get(stuck["id"], 0) + 1
                log(f"{label}: {stuck['id']} stopped making progress")
            if self.dry_run:
                break
        missing = [e for e in entries if not _evidence_done(e)]
        if missing and not self.dry_run:
            log(
                f"{label}: {len(missing)} entries still without evidence after {max_attempts} attempts"
            )
        return missing

    def _launch(
        self,
        chunk,
        protocol,
        label,
        seconds_per_cell,
        extra_env,
        first_cell_s=1200,
        cell_s=300,
    ):
        """One launch of `chunk`. Returns the entry the launch got stuck on, if
        the watchdog stopped it: no new evidence for `cell_s` (`first_cell_s`
        before the first, which pays for JIT compiles and communicator setup)."""
        for _ in range(200):
            self.port += 2
            if self.port > self.PORT_HIGH:
                self.port = self.PORT_LOW
            if self.dry_run or _port_free(self.port):
                break
        for e in chunk:
            Path(e["evidence"]).parent.mkdir(parents=True, exist_ok=True)
        launch_id = (
            f"{label}-{datetime.datetime.now().strftime('%m%d_%H%M%S')}-{self.port}"
        )
        mdir = self.workdir / "manifests" / label
        mdir.mkdir(parents=True, exist_ok=True)
        manifest = mdir / f"{launch_id}.json"
        manifest.write_text(
            json.dumps(
                {"launch_id": launch_id, "protocol": protocol, "cells": chunk}, indent=1
            )
            + "\n"
        )
        ldir = self.workdir / "logs" / label
        ldir.mkdir(parents=True, exist_ok=True)
        timeout_s = int(600 + seconds_per_cell * len(chunk))
        max_tokens = max(e["cell"]["tokens"] for e in chunk)
        if self.two:
            logs = [ldir / f"{launch_id}_node{r}.log" for r in (0, 1)]
            cmds = ep16_commands(
                self.m,
                manifest,
                self.port,
                timeout_s,
                logs,
                protocol["num_qp_per_pe"],
                max_tokens,
                extra_env,
            )
        else:
            logs = [ldir / f"{launch_id}_node0.log"]
            cmds = [
                ep8_command(self.m, manifest, self.port, timeout_s, logs[0], extra_env)
            ]
        if self.dry_run:
            log(f"[dry-run] {launch_id}: {len(chunk)} cells")
            for c in cmds:
                print("   ", " ".join(shlex.quote(a) for a in c)[:600])
            return None
        if self.two and self.m.sync == "rsync":
            # Without a shared filesystem node1 needs the tool, the launcher,
            # the harness, the variant tables this launch reads, the manifest
            # and the log directory. mori itself comes from the installed wheel.
            repo = Path(self.m.repo)
            tables = sorted({str(Path(e["table_dir"])) for e in chunk})
            self.m.sync_to_peer(
                [
                    HERE,
                    repo / "tools/run_internode_test.sh",
                    repo
                    / "tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py",
                    *tables,
                    manifest,
                ]
            )
            sh(self.m.ssh + [f"mkdir -p {shlex.quote(str(ldir))}"])
        last_busy = self.nodes.wait_idle(self.last_busy, self.quiet_s)
        if last_busy is None:
            log("GPUs stayed busy; skipping this launch")
            return None
        self.last_busy = last_busy
        self.nodes.cleanup()
        watch = Watch(self.nodes, self.ci)
        watch.start()
        t0 = time.time()
        procs = []
        for i, c in enumerate(reversed(cmds)):  # node1 first, as refresh_driver does
            procs.append(
                subprocess.Popen(
                    c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            )
            if i == 0 and len(cmds) > 1:
                time.sleep(1)
        hard = t0 + timeout_s + 180
        killed = None
        stuck = None
        done_count, last_progress = 0, time.time()
        failed_at = None
        while time.time() < hard and any(p.poll() is None for p in procs):
            if watch.dirty.is_set() and killed is None:
                killed = "interference"
                log(f"{launch_id}: foreign GPU user seen; stopping the launch")
                self.nodes.cleanup()
            # One node's launcher died (bad port, crash): the other one would
            # wait in its rendezvous or collectives until the timeout.
            if failed_at is None and any(p.poll() not in (None, 0) for p in procs):
                failed_at = time.time()
            if (
                killed is None
                and failed_at is not None
                and time.time() - failed_at > 30
            ):
                killed = "node-failed"
                log(
                    f"{launch_id}: a node's launcher exited with an error; stopping the launch"
                )
                self.nodes.cleanup()
            count = sum(Path(e["evidence"]).exists() for e in chunk)
            if count != done_count:
                done_count, last_progress = count, time.time()
            limit = first_cell_s if done_count == 0 else cell_s
            if killed is None and time.time() - last_progress > limit:
                killed = "watchdog"
                stuck = next(
                    (e for e in chunk if not Path(e["evidence"]).exists()), None
                )
                log(f"{launch_id}: no new cell for {limit}s; stopping the launch")
                self.nodes.cleanup()
            time.sleep(5)
        for p in procs:
            if p.poll() is None:
                p.kill()
        codes = [p.wait() for p in procs]
        watch.stop()
        self.nodes.cleanup()
        done = [e for e in chunk if Path(e["evidence"]).exists()]
        clean_until = watch.clean_until()
        if clean_until is not None:
            self.last_busy = time.time()
            self.last_busy_file.write_text(str(self.last_busy))
            late = []
            for e in done:
                try:
                    rec = json.loads(Path(e["evidence"]).read_text())
                except ValueError:
                    continue
                if rec.get("t_end", 0) > clean_until:
                    late.append(e["evidence"])
            quarantine(late, self.workdir / "quarantine", "interference")
            seen = sorted({i for _, items in watch.samples for i in items})
            log(f"{launch_id}: interference {seen[:3]}; quarantined {len(late)} cells")
        log(
            f"{launch_id}: exit {codes} in {time.time() - t0:.0f}s, "
            f"{len(done)}/{len(chunk)} cells with evidence"
            + (f" ({killed})" if killed else "")
        )
        return stuck


def _take_chunk(pending, chunk_cells):
    """Up to chunk_cells entries, never splitting one cell's entries."""
    by_cell = {}
    for e in pending:
        by_cell.setdefault(e.get("block") or e["cell_key"], []).append(e)
    chunk = []
    for block in by_cell.values():
        if chunk and len(chunk) + len(block) > chunk_cells:
            break
        chunk += block
    return chunk
```

#### `fxcampaign.py`

```python
#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""MI355X EPv2 tuning campaigns: qp_status.md's fx flow on the 355 nodes.

Machine (settings: fxconfig.py, $FX_WORKDIR/machine.env):
  where [--get KEY]             settings in effect, container state per node
  make-container                create / start the container on each node
  install-mori                  build this repo's mori, install it on each node
  preflight                     CPU-only checks of the container on each node
  ci-freeze | ci-resume | ci-status   the GitHub runner on node0 (FX_CI_FREEZE=1)

Campaigns ($FX_WORKDIR/campaigns/NAME):
  init    NAME --preset P       campaign directory, base and seed tables, candidates
  plan    NAME [STAGE]          entries and machine-time estimate per stage
  run     NAME STAGE            measure a stage (resumable; stalls and gate
                                failures are quarantined and re-measured)
  analyze NAME STAGE            prune | geo | qp | ab | cf
  chain   NAME                  every stage in order, analysis in between
  land    NAME [--into DIR]     merged rules + metrics from the confirm A/B
  status  NAME                  progress and the measured seconds per cell
  archive NAME STAGE...         move a stage's evidence aside (to re-measure it)
  rebase  NAME --base DIR       new base tables; geometry and QP picks are kept

Stages (EP16; EP8 has no q):
  g1  geometry round 1: every candidate, both owned phases, seed at both ends
      of each cell's block
  g2  geometry round 2: per rule, the candidates within `window` of its best
      in g1 (at least `min_keep`)
  q   QP / load-first: every (d, c) and (q, q, LF) on the selected geometry,
      2 rounds
  ab  base tables against the new ones, 3 rounds, arms adjacent and alternated
  cf  base against the per-bucket merge, 3 rounds
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fxanalyze as A  # noqa: E402
import fxconfig  # noqa: E402
import fxtables as T  # noqa: E402
from fxcells import (  # noqa: E402
    INTERNODE,
    TOKENS_ALL,
    TOKENS_LARGE,
    Cell,
    Group,
    Shape,
    default_geometry,
    geometry_candidates,
    safe_name,
)
from fxrun import Machine, Runner, log, quarantine  # noqa: E402

CAMPAIGNS = fxconfig.WORKDIR / "campaigns"
PROTOCOL = {
    "passes": 3,
    "rounds": 30,
    "warmup": 20,
    "drop_rounds": 1,
    "num_qp_per_pe": 8,
    "scale_dim": 0,
    "seed": 4242,
    "bench_weights": False,
    "keep_samples": False,
}
SECONDS_PER_CELL = 2.0  # 308x driver throughput (qp_status.md §3, §7.4)
OVERHEAD = 1.4
# vf: no timing, 20 identity-expert checks per cell on fresh inputs, for the
# intermittent wrong results seen on 896/16 V2LL dispatch at 256 x 16.
STAGE_PROTOCOL = {"vf": {"verify_rounds": 20, "timing": False}}


# ---------------------------------------------------------------- presets


def _both(family, shape, dtypes, tokens=TOKENS_ALL):
    return Group(family, shape, dtypes, "bf16", tokens)


PRESETS = {
    # The four shapes MI355X already has, re-screened with geometry and QP
    # together, with T4-T32 added where MI308X has them.
    "ep16-existing": lambda: [
        g
        for fam in INTERNODE
        for g in (
            _both(fam, Shape(256, 6, 7168), ("bf16",), TOKENS_LARGE),
            _both(fam, Shape(256, 8, 7168), ("bf16", "fp8_e4m3")),
            _both(fam, Shape(384, 6, 5120), ("bf16", "fp8_e4m3")),
            _both(fam, Shape(384, 6, 7168), ("bf16", "fp8_e4m3"), TOKENS_LARGE),
        )
    ],
    # DeepSeek-V3/R1 alone (256/8 h7168, both families): the quickest first set.
    "ep16-dsr1": lambda: [
        _both(fam, Shape(256, 8, 7168), ("bf16", "fp8_e4m3")) for fam in INTERNODE
    ],
    # ep16-existing without 256/8 h7168, for after ep16-dsr1 has landed.
    "ep16-existing-rest": lambda: [
        g
        for fam in INTERNODE
        for g in (
            _both(fam, Shape(256, 6, 7168), ("bf16",), TOKENS_LARGE),
            _both(fam, Shape(384, 6, 5120), ("bf16", "fp8_e4m3")),
            _both(fam, Shape(384, 6, 7168), ("bf16", "fp8_e4m3"), TOKENS_LARGE),
        )
    ],
    # The shapes MI308X has and MI355X does not.
    "ep16-new": lambda: [
        g
        for fam in INTERNODE
        for g in (
            _both(fam, Shape(256, 8, 6144), ("bf16", "fp8_e4m3")),
            _both(fam, Shape(272, 9, 7168, "replica"), ("bf16", "fp8_e4m3")),
            _both(fam, Shape(896, 16, 3584), ("bf16", "fp8_e4m3")),
        )
    ],
    # FP4 dispatch rules (stored at the packed width), geometry x dispatch QP
    # count, against the bf16 combine rule the bf16 group owns.
    "ep16-fp4": lambda: [
        Group(fam, Shape(256, 8, 7168), ("fp4",), "bf16", TOKENS_ALL, ("dispatch",))
        for fam in INTERNODE
    ],
    # Symmetric FP8 combine rules, against the fp8 dispatch rules.
    "ep16-fp8c": lambda: [
        Group(
            fam,
            Shape(256, 8, hidden),
            ("fp8_e4m3",),
            "fp8_e4m3",
            TOKENS_ALL,
            ("combine",),
        )
        for fam in INTERNODE
        for hidden in (7168, 6144)
    ],
    # Small campaigns for the offline end-to-end test (fxsim).
    "test-ep16": lambda: [
        g
        for fam in INTERNODE
        for g in (
            _both(fam, Shape(256, 8, 7168), ("bf16", "fp8_e4m3"), (4, 64, 512, 4096)),
            _both(fam, Shape(896, 16, 3584), ("bf16", "fp8_e4m3"), (16, 1024)),
        )
    ],
    "test-ep8": lambda: [
        _both(
            "intra", Shape(264, 9, 7168, "replica"), ("bf16", "fp8_e4m3"), (64, 4096)
        ),
    ],
    # EP8: the two shapes MI355X lacks, and 384/6 h5120, which MI308X re-screened.
    "ep8": lambda: [
        _both("intra", Shape(256, 8, 6144), ("bf16", "fp8_e4m3"), TOKENS_LARGE),
        _both(
            "intra", Shape(264, 9, 7168, "replica"), ("bf16", "fp8_e4m3"), TOKENS_LARGE
        ),
        _both("intra", Shape(384, 6, 5120), ("bf16", "fp8_e4m3"), TOKENS_LARGE),
    ],
}
QP_COUNTS = [1, 2, 4]


# ---------------------------------------------------------------- campaign


class Campaign:
    def __init__(self, name):
        self.root = CAMPAIGNS / name
        self.spec = json.loads((self.root / "campaign.json").read_text())
        self.groups = [Group.from_json(g) for g in self.spec["groups"]]
        self.kind = self.spec["kind"]
        self.protocol = self.spec["protocol"]
        self.candidates = {
            fam: {ph: [tuple(g) for g in gs] for ph, gs in phases.items()}
            for fam, phases in self.spec["candidates"].items()
        }
        self._tables = {}

    # paths
    def tables_dir(self, variant):
        return self.root / "tables" / safe_name(variant)

    def evidence_dir(self, stage):
        return self.root / "evidence" / stage

    def picks_path(self, name):
        return self.root / "picks" / f"{name}.json"

    def save_picks(self, name, data):
        path = self.picks_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, default=list) + "\n")

    def load_picks(self, name):
        return json.loads(self.picks_path(name).read_text())

    def report(self, name, lines):
        path = self.root / "reports" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))

    # tables
    def tableset(self, variant) -> T.TableSet:
        if variant not in self._tables:
            self._tables[variant] = T.TableSet.load(self.ensure(variant))
        return self._tables[variant]

    def ensure(self, variant) -> Path:
        """The directory of a variant, written on first use."""
        if variant in ("seed-end",):
            variant = "seed"
        path = self.tables_dir(variant)
        if path.exists():
            return path
        if variant == "seed":
            T.seed(self.tableset("base"), self.groups).save(path)
        elif variant.startswith("geo-"):
            _, fam, phase, *nums = variant.split("-")
            T.with_geometry(
                self.tableset("seed"), self.groups, fam, phase, tuple(map(int, nums))
            ).save(path)
        elif variant.startswith("qp-"):
            parts = variant.split("-")
            fam, d, c = parts[1], int(parts[2][1:]), int(parts[3][1:])
            lf = len(parts) > 4 and parts[4] == "lf"
            T.with_qp(
                self.tableset("geo"), A.qp_groups(self.groups), fam, d, c, lf
            ).save(path)
        else:
            raise FileNotFoundError(
                f"{path}: variant {variant!r} has not been built (run its analysis)"
            )
        return path

    # entries
    def entry(self, stage, rnd, variant, cell: Cell, block, arm=None):
        table_dir = self.ensure(variant)
        return {
            "id": f"{stage}/r{rnd}/{variant}/{cell.key}",
            "stage": stage,
            "round": str(rnd),
            "variant": variant,
            "arm": arm,
            "cell": cell.to_json(),
            "cell_key": cell.key,
            "table_dir": str(table_dir),
            "expect": T.expected(table_dir, cell, self.protocol["num_qp_per_pe"]),
            "evidence": str(
                self.evidence_dir(stage)
                / f"r{rnd}"
                / safe_name(variant)
                / f"{safe_name(cell.key)}.json"
            ),
            "block": block,
        }

    def geometry_variants(
        self, group: Group, cell: Cell, lists=None, full_if_missing=False
    ):
        """Variant ids of the cell's candidates in every owned phase: all of
        them, or the ones `lists` keeps for the cell's rule. A rule `lists`
        does not name gets every candidate when full_if_missing (round-1
        pre-pruning covers only some rules), none otherwise."""
        out = []
        for phase in group.owns:
            sel = json.dumps(list(cell.selector(phase)))
            pool = self.candidates[group.family][phase]
            if lists is not None:
                kept = lists.get(group.family, {}).get(phase, {}).get(sel)
                if kept is not None or not full_if_missing:
                    pool = [tuple(g) for g in (kept or [])]
            out += [A.geometry_variant_id(group.family, phase, g) for g in pool]
        return sorted(set(out))

    def stage_entries(self, stage):
        entries = []
        if stage in ("g1", "g2"):
            rnd = stage[1]
            lists, full_if_missing = None, False
            if stage == "g2":
                lists = self.load_picks("prune")
            elif self.picks_path("r1_lists").exists():
                lists, full_if_missing = self.load_picks("r1_lists"), True
            for group in self.groups:
                for cell in group.cells():
                    block = f"g/r{rnd}/{cell.key}"
                    variants = self.geometry_variants(
                        group, cell, lists, full_if_missing
                    )
                    random.Random(f"{stage}/{cell.key}").shuffle(variants)
                    for v in ["seed"] + variants + ["seed-end"]:
                        entries.append(self.entry("g", rnd, v, cell, block))
        elif stage == "q":
            for rnd in ("1", "2"):
                for group in A.qp_groups(self.groups):
                    for cell in group.cells():
                        block = f"q/r{rnd}/{cell.key}"
                        grid = [
                            A.qp_variant_id(group.family, d, c, lf)
                            for d, c, lf in A.qp_grid(group.family, QP_COUNTS)
                        ]
                        random.Random(f"q{rnd}/{cell.key}").shuffle(grid)
                        entries += [self.entry("q", rnd, v, cell, block) for v in grid]
        elif stage in ("ab", "cf"):
            new = "final" if stage == "ab" else "merged"
            self.ensure(new)
            for rnd in (1, 2, 3):
                for group in self.groups:
                    for cell in group.cells():
                        arms = ["base", new] if rnd % 2 else [new, "base"]
                        block = f"{stage}/r{rnd}/{cell.key}"
                        entries += [
                            self.entry(stage, rnd, arm, cell, block, arm=arm)
                            for arm in arms
                        ]
        elif stage == "vf":
            self.ensure("merged")
            for group in self.groups:
                for cell in group.cells():
                    entries.append(
                        self.entry(
                            "vf", 1, "merged", cell, f"vf/{cell.key}", arm="merged"
                        )
                    )
        else:
            raise SystemExit(f"unknown stage {stage!r}")
        return entries

    def records(self, stage):
        return A.load_records(self.evidence_dir(stage))


# ---------------------------------------------------------------- commands


def cmd_init(opts):
    root = CAMPAIGNS / opts.name
    if root.exists():
        raise SystemExit(f"{root} exists")
    groups = PRESETS[opts.preset]()
    kind = "ep8" if all(g.family == "intra" for g in groups) else "ep16"
    if kind == "ep16" and any(g.family == "intra" for g in groups):
        raise SystemExit("a campaign is either EP8 or EP16")
    base = T.TableSet.load(opts.base)
    candidates = {}
    for fam in sorted({g.family for g in groups}):
        candidates[fam] = {}
        for phase in ("dispatch", "combine"):
            pool = set(geometry_candidates(fam, phase))
            pool |= set(T.table_geometries(base, fam, phase))
            pool.add(default_geometry(fam, phase))
            pool = sorted(pool)
            if (
                any(g.family == fam and g.owns == ("dispatch",) for g in groups)
                and phase == "dispatch"
            ):
                # FP4: the dispatch count is screened with the geometry, because
                # the bucket's combine already runs its own count (qp_status §11.1).
                pool = [(*g, q) for g in pool for q in QP_COUNTS]
            candidates[fam][phase] = [list(g) for g in pool]
    spec = {
        "name": opts.name,
        "preset": opts.preset,
        "kind": kind,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_source": str(opts.base),
        "groups": [g.to_json() for g in groups],
        "protocol": dict(PROTOCOL),
        "candidates": candidates,
        "prune": {"window": 0.05, "min_keep": 5},
        "margin": 0.01,
    }
    root.mkdir(parents=True)
    (root / "campaign.json").write_text(json.dumps(spec, indent=1) + "\n")
    base.save(root / "tables" / "base")
    c = Campaign(opts.name)
    c.ensure("seed")
    log(
        f"{root}: {len(groups)} groups, {sum(len(g.cells()) for g in groups)} cells, kind {kind}"
    )
    for fam, phases in candidates.items():
        log(
            f"  {fam}: candidates "
            + ", ".join(f"{ph} {len(gs)}" for ph, gs in phases.items())
        )


def _estimate(n):
    return n * SECONDS_PER_CELL / 3600, n * SECONDS_PER_CELL * OVERHEAD / 3600


def cmd_plan(opts):
    c = Campaign(opts.name)
    stages = [opts.stage] if opts.stage else ["g1"]
    for stage in stages:
        entries = c.stage_entries(stage)
        raw, mach = _estimate(len(entries))
        print(
            f"{stage}: {len(entries)} measurements, ~{raw:.1f} h raw, ~{mach:.1f} h with stalls/waits"
        )


def _machine(opts):
    m = Machine()
    for name in (
        "container",
        "peer",
        "peer_label",
        "peer_key_alias",
        "master_addr",
        "ifname",
    ):
        value = getattr(opts, name, None)
        if value:
            setattr(m, name, value)
    return m


def _repair(c: Campaign, stage, records) -> int:
    """Quarantine stalled runs and gate failures once, so the next pass of the
    stage measures them again. Returns how many were moved."""
    state_path = c.root / "repairs.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    moved = []
    for rec in A.find_stalls(records) + A.find_block_stalls(records):
        if state.get(rec["id"], 0) < 2 and rec not in [r for _, r in moved]:
            moved.append(("stall", rec))
    for rec in records:
        probs = A.problems(rec)
        if (
            any(p.startswith("gates") or p == "error" for p in probs)
            and state.get(rec["id"], 0) < 1
        ):
            moved.append(("gates" if probs[0].startswith("gates") else "error", rec))
    for reason, rec in moved:
        state[rec["id"]] = state.get(rec["id"], 0) + 1
        quarantine([rec["_path"]], c.root / "quarantine", reason)
    state_path.write_text(json.dumps(state, indent=1) + "\n")
    return len(moved)


def cmd_run(opts):
    c = Campaign(opts.name)
    entries = c.stage_entries(opts.stage)
    if getattr(opts, "limit", None):
        entries = entries[: opts.limit]
    if getattr(opts, "sample", None):
        # Spread over the whole stage, so a smoke test reaches both families,
        # both dtypes and the token range; the evidence counts for the stage.
        step = max(1, len(entries) // opts.sample)
        entries = [dict(e, block=e["id"]) for e in entries[::step][: opts.sample]]
    raw, mach = _estimate(len(entries))
    log(f"{opts.stage}: {len(entries)} measurements (~{raw:.1f} h raw)")
    protocol = dict(c.protocol, **STAGE_PROTOCOL.get(opts.stage, {}))
    if getattr(opts, "simulate", False):
        import fxsim

        for attempt in range(3):
            fxsim.simulate(entries, protocol, seed=attempt)
            moved = (
                _repair(c, opts.stage, c.records(entries[0]["stage"])) if entries else 0
            )
            if not moved:
                break
            log(
                f"{opts.stage}: re-measuring {moved} stalled / gate-failed runs (simulated)"
            )
        summarize_stage(c, opts.stage, entries)
        return
    with Runner(
        _machine(opts),
        c.root,
        two_nodes=c.kind == "ep16",
        dry_run=opts.dry_run,
        quiet_s=opts.quiet,
        ci_wait=opts.ci_wait,
    ) as runner:
        if not entries:
            log(f"{opts.stage}: nothing to measure")
            return
        for attempt in range(3):
            runner.run(entries, protocol, opts.stage, chunk_cells=opts.chunk)
            if opts.dry_run:
                return
            stage_dir = entries[0]["stage"]
            moved = _repair(c, opts.stage, c.records(stage_dir))
            if not moved:
                break
            log(f"{opts.stage}: re-measuring {moved} stalled / gate-failed runs")
    summarize_stage(c, opts.stage, entries)


def summarize_stage(c, stage, entries):
    by_problem = {}
    for e in entries:
        path = Path(e["evidence"])
        if not path.exists():
            by_problem["missing"] = by_problem.get("missing", 0) + 1
            continue
        rec = json.loads(path.read_text())
        for p in A.problems(rec) or ["ok"]:
            by_problem[p.split(":")[0]] = by_problem.get(p.split(":")[0], 0) + 1
    log(f"{stage}: " + ", ".join(f"{k} {v}" for k, v in sorted(by_problem.items())))


def _fmt_geom(g):
    return "/".join(str(x) for x in g) if g else "seed"


def analyze_prune(c: Campaign):
    records = c.records("g")
    idx = A.index(records)
    lists = A.prune(
        idx,
        c.groups,
        c.candidates,
        rnd="1",
        wrong=A.wrong_configs(records),
        **c.spec["prune"],
    )
    c.save_picks("prune", lists)
    lines = [
        "# round-2 candidates per rule (window %.0f%%, min %d)"
        % (100 * c.spec["prune"]["window"], c.spec["prune"]["min_keep"])
    ]
    total = kept = 0
    for fam, phases in lists.items():
        for phase, rules in phases.items():
            for sel, geoms in rules.items():
                total += len(c.candidates[fam][phase])
                kept += len(geoms)
                lines.append(f"{fam:6s} {phase:8s} {sel}: {len(geoms)} kept")
    lines.append(
        f"kept {kept} of {total} rule-candidate pairs ({kept / max(total, 1):.0%})"
    )
    if c.picks_path("r1_probes").exists():
        # A probe is a candidate the old harness sweeps would have dropped. One
        # that round 1 keeps means those sweeps no longer rank this kernel and
        # protocol, and the round-1 pre-pruning should not be trusted.
        probes = c.load_picks("r1_probes")
        hits = total_probes = 0
        for fam, phases in probes.items():
            for phase, rules in phases.items():
                for sel, geoms in rules.items():
                    kept_here = {
                        tuple(g) for g in lists.get(fam, {}).get(phase, {}).get(sel, [])
                    }
                    total_probes += len(geoms)
                    for g in geoms:
                        if tuple(g) in kept_here:
                            hits += 1
                            lines.append(
                                f"PROBE KEPT {fam} {phase} {sel}: {_fmt_geom(g)}"
                            )
        lines.append(
            f"harness probes kept in round 2: {hits} of {total_probes}"
            + (
                "  <- re-run g1 without picks/r1_lists.json"
                if hits > max(1, total_probes // 20)
                else ""
            )
        )
    c.report("prune", lines)


def analyze_geo(c: Campaign):
    records = c.records("g")
    idx = A.index(records)
    wrong = A.wrong_configs(records)
    picks = A.select_geometry(
        idx, c.groups, c.load_picks("prune"), margin=c.spec["margin"], wrong=wrong
    )
    c.save_picks("geo", picks)
    c.save_picks("wrong_g", sorted(wrong))
    seed = c.tableset("seed")
    chosen = [
        {
            "family": p["family"],
            "selector": p["selector"],
            "fields": _geometry_fields(p["geometry"]),
        }
        for p in picks
        if p.get("geometry")
    ]
    path = c.tables_dir("geo")
    if path.exists():
        shutil.rmtree(path)
    T.with_picks(seed, chosen).save(path)
    c._tables.pop("geo", None)
    lines = [
        "# geometry picks (seed kept unless >=1% faster and faster in every round)"
    ]
    for fam, shape, vid in sorted(wrong):
        lines.append(f"EXCLUDED (wrong result) {fam} {shape} {vid}")
    for p in picks:
        lines.append(
            f"{p['family']:6s} {p['phase']:8s} {tuple(p['selector'])}: "
            f"{_fmt_geom(p.get('geometry'))}  best {_fmt_geom(p.get('best_geometry'))} "
            f"{100 * p.get('change', 0):+.1f}%  ({p.get('reason')})"
        )
    lines.append(f"{len(chosen)} of {len(picks)} rules changed")
    c.report("geo", lines)


def _geometry_fields(geom):
    block, rdma, warp, *qp = geom
    return {
        "block_num": block,
        "rdma_block_num": rdma,
        "warp_per_block": warp,
        "active_qps": qp[0] if qp else None,
    }


def _base_lf(c: Campaign):
    base_dir = c.tables_dir("base")
    return lambda cell: T.expected(base_dir, cell, c.protocol["num_qp_per_pe"])[
        "combine"
    ]["load_first"]


def analyze_qp(c: Campaign):
    geo = c.tableset("geo")
    base_lf = _base_lf(c)
    fields = []
    lines = ["# QP / load-first picks against (2, 2, existing load-first)"]
    if c.kind == "ep16" and A.qp_groups(c.groups):
        records = c.records("q")
        # QP runs count only on the geometry the selected table has: a rule
        # whose geometry changed after the QP stage (a geometry excluded for a
        # wrong result) gets no QP pick until it is measured again.
        geo_dir = c.tables_dir("geo")
        cur = {}
        kept = []
        for rec in records:
            key = rec["cell_key"]
            if key not in cur:
                e = T.expected(
                    geo_dir, Cell.from_json(rec["cell"]), c.protocol["num_qp_per_pe"]
                )
                cur[key] = {
                    ph: (
                        e[ph]["block_num"],
                        e[ph]["rdma_block_num"],
                        e[ph]["warp_per_block"],
                    )
                    for ph in ("dispatch", "combine")
                }
            eff = rec.get("effective") or {}
            got = {
                ph: tuple(
                    (eff.get(ph) or {}).get(f)
                    for f in ("block_num", "rdma_block_num", "warp_per_block")
                )
                for ph in ("dispatch", "combine")
            }
            if got == cur[key]:
                kept.append(rec)
        if len(kept) != len(records):
            lines.append(
                f"# {len(records) - len(kept)} QP runs dropped: measured on a geometry the table no longer has"
            )
        records = kept
        idx = A.index(records)
        picks = A.select_qp(
            idx,
            c.groups,
            QP_COUNTS,
            base_lf,
            margin=c.spec["margin"],
            wrong=A.wrong_configs(records),
        )
        c.save_picks("qp", picks)
        consistency = {
            g.tag: A.model_consistency(
                idx, g.cells(), A.qp_grid(g.family, QP_COUNTS), ("1", "2")
            )
            for g in A.qp_groups(c.groups)
        }
        c.save_picks("qp_model", consistency)
        for g in A.qp_groups(c.groups):
            for p in [
                p
                for p in picks
                if p["family"] == g.family and p["shape"] == g.shape.tag
            ]:
                choice = p.get("choice") or p["baseline"]
                cells = {
                    cell.key: cell for cell in g.cells() if cell.tokens == p["tokens"]
                }
                for key, d in choice["dispatch"].items():
                    fields.append(
                        {
                            "family": g.family,
                            "selector": list(cells[key].selector("dispatch")),
                            "fields": {"active_qps": d if p.get("choice") else None},
                        }
                    )
                any_cell = next(iter(cells.values()))
                fields.append(
                    {
                        "family": g.family,
                        "selector": list(any_cell.selector("combine")),
                        "fields": {
                            "active_qps": (
                                choice["combine"] if p.get("choice") else None
                            ),
                            "combine_load_first": (
                                True if choice["load_first"] else None
                            ),
                        },
                    }
                )
                lines.append(
                    f"{g.family:6s} {g.shape.tag:20s} T{p['tokens']:<5}: "
                    + (
                        f"d={list(choice['dispatch'].values())} c={choice['combine']}"
                        f"{' LF' if choice['load_first'] else ''}"
                        if p.get("choice")
                        else "baseline"
                    )
                    + f"  best {100 * p.get('change', 0):+.1f}% ({p.get('reason')})"
                )
        lines.append("# factor model: " + json.dumps(consistency))
    # Rules no QP stage tuned keep the load-first they run today.
    tuned = {json.dumps([f["family"], f["selector"]]) for f in fields}
    for g in c.groups:
        if g.family != "v2" or "combine" not in g.owns:
            continue
        for cell in g.cells():
            sel = list(cell.selector("combine"))
            if json.dumps([g.family, sel]) not in tuned and base_lf(cell):
                fields.append(
                    {
                        "family": g.family,
                        "selector": sel,
                        "fields": {"combine_load_first": True},
                    }
                )
                tuned.add(json.dumps([g.family, sel]))
    path = c.tables_dir("final")
    if path.exists():
        shutil.rmtree(path)
    T.with_picks(geo, fields).save(path)
    c._tables.pop("final", None)
    c.report("qp", lines)


def _ab_report(c, stage, arm_a, arm_b):
    records = c.records(stage)
    idx = A.index(records)
    rows = A.ab_buckets(
        idx,
        c.groups,
        arm_a,
        arm_b,
        margin=c.spec["margin"],
        wrong_b=A.wrong_cells(records, arm_b),
    )
    lines = [
        f"# {stage}: {arm_b} against {arm_a}, D+C summed over each bucket's dtypes"
    ]
    for r in rows:
        if "change" not in r:
            lines.append(
                f"{r['family']:6s} {r['shape']:20s} T{r['tokens']:<5} {r['reason']}"
            )
            continue
        per = " ".join(
            f"{d} {100 * v['change']:+.1f}%" for d, v in r["per_dtype"].items()
        )
        lines.append(
            f"{r['family']:6s} {r['shape']:20s} T{r['tokens']:<5} {100 * r['change']:+6.1f}% "
            f"every_round={r['every_round']} take={r['take']}  [{per}]"
        )
    changes = [r["change"] for r in rows if "change" in r]
    if changes:
        lines.append(
            f"mean change {100 * statistics.fmean(changes):+.2f}%, "
            f"{sum(r['take'] for r in rows)} of {len(rows)} buckets taken"
        )
    return rows, lines


def analyze_ab(c: Campaign):
    rows, lines = _ab_report(c, "ab", "base", "final")
    c.save_picks("ab", rows)
    accepted = [(r["family"], r["shape"], r["tokens"]) for r in rows if r.get("take")]
    path = c.tables_dir("merged")
    if path.exists():
        shutil.rmtree(path)
    T.merge_buckets(
        c.tableset("base"),
        c.tableset("final"),
        c.groups,
        accepted,
        base_dir=c.tables_dir("base"),
        num_qp_per_pe=c.protocol["num_qp_per_pe"],
    ).save(path)
    c._tables.pop("merged", None)
    c.report("ab", lines)


def analyze_cf(c: Campaign):
    rows, lines = _ab_report(c, "cf", "base", "merged")
    c.save_picks("cf", rows)
    c.report("cf", lines)


ANALYSES = {
    "prune": analyze_prune,
    "geo": analyze_geo,
    "qp": analyze_qp,
    "ab": analyze_ab,
    "cf": analyze_cf,
}


def cmd_analyze(opts):
    ANALYSES[opts.stage](Campaign(opts.name))


def cmd_chain(opts):
    c = Campaign(opts.name)
    steps = [("run", "g1"), ("analyze", "prune"), ("run", "g2"), ("analyze", "geo")]
    if c.kind == "ep16" and A.qp_groups(c.groups):
        steps += [("run", "q")]
    steps += [
        ("analyze", "qp"),
        ("run", "ab"),
        ("analyze", "ab"),
        ("run", "cf"),
        ("analyze", "cf"),
        ("run", "vf"),
    ]
    for kind, stage in steps:
        log(f"== {kind} {stage}")
        opts.stage = stage
        (cmd_run if kind == "run" else cmd_analyze)(opts)
        if opts.dry_run and kind == "run":
            log("dry run: stopping before the analysis that needs evidence")
            return


def cmd_land(opts):
    c = Campaign(opts.name)
    into = Path(opts.into) if opts.into else c.root / "landed"
    if not opts.into:
        if into.exists():
            shutil.rmtree(into)
        shutil.copytree(c.tables_dir("base"), into)
    target = T.TableSet.load(into)
    merged = c.tableset("merged")
    idx = A.index(c.records("cf"))
    accepted = {
        (r["family"], r["shape"], r["tokens"])
        for r in c.load_picks("ab")
        if r.get("take")
    }
    # A bucket whose merged configuration failed a correctness check (cf, vf)
    # is not landed, whatever its timing.
    cf_records = c.records("cf")
    vf_records = c.records("vf")
    wrong = A.wrong_cells(cf_records, "merged") | A.wrong_cells(vf_records, "merged")
    if vf_records:
        log(
            f"vf: {len(vf_records)} cells checked, {len(A.wrong_cells(vf_records, 'merged'))} wrong"
        )
    else:
        log("vf: not run; landing on the cf checks alone")
    landed = 0
    pinned = 0
    for g in c.groups:
        for cell in g.cells():
            bucket_cells = [x.key for x in g.cells() if x.tokens == cell.tokens]
            if any(k in wrong for k in bucket_cells):
                log(
                    f"NOT LANDED (wrong result): {g.family} {g.shape.tag} T{cell.tokens} {cell.dispatch_dtype}"
                )
                for phase in g.owns:
                    sel = cell.selector(phase)
                    if target.rules[g.family].get(sel) is None:
                        target.upsert(
                            g.family,
                            sel,
                            **T.pinned_rule(
                                c.tables_dir("base"),
                                cell,
                                phase,
                                c.protocol["num_qp_per_pe"],
                            ),
                        )
                        pinned += 1
                continue
            if (g.family, g.shape.tag, cell.tokens) not in accepted:
                # Not taken: only a rule pinning today's configuration, where
                # there was no rule (merge_buckets), so neighbours do not move it.
                for phase in g.owns:
                    sel = cell.selector(phase)
                    if (
                        c.tableset("base").get(g.family, sel) is None
                        and merged.get(g.family, sel) is not None
                        and target.rules[g.family].get(sel) is None
                    ):
                        rule = {
                            k: v
                            for k, v in merged.get(g.family, sel).items()
                            if k not in T.METRIC_FIELDS
                        }
                        metrics = A.landing_metrics(
                            idx, A.metric_cell(g, phase, sel), "merged", phase
                        )
                        if metrics:
                            rule.update(
                                {k: v for k, v in metrics.items() if v is not None}
                            )
                        target.rules[g.family][sel] = rule
                        pinned += 1
                continue
            for phase in g.owns:
                sel = cell.selector(phase)
                rule = merged.get(g.family, sel)
                if rule is None:
                    continue
                metrics = A.landing_metrics(
                    idx, A.metric_cell(g, phase, sel), "merged", phase
                )
                new_rule = dict(rule)
                for name in T.METRIC_FIELDS:
                    new_rule.pop(name, None)
                if metrics:
                    new_rule.update({k: v for k, v in metrics.items() if v is not None})
                target.rules[g.family][sel] = new_rule
                landed += 1
    target.write_into(into)
    log(f"landed {landed} rules (+{pinned} pinning today's configuration) into {into}")


def _archive(c: Campaign, stages=(), tables=(), picks=()):
    """Move evidence, tables and picks out of the way (never deleted)."""
    dest = c.root / "archive" / time.strftime("%m%d_%H%M%S")
    moved = []
    for kind, names in (("evidence", stages), ("tables", tables), ("picks", picks)):
        for name in names:
            src = c.root / kind / (f"{name}.json" if kind == "picks" else name)
            if src.exists():
                (dest / kind).mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest / kind / src.name))
                moved.append(f"{kind}/{src.name}")
    if moved:
        log(f"{c.root.name}: archived {', '.join(moved)} -> {dest}")
    return dest


def cmd_archive(opts):
    """Set stages aside before re-running them on changed tables."""
    c = Campaign(opts.name)
    _archive(c, stages=opts.stages, picks=[s for s in opts.stages if s in ("ab", "cf")])


def cmd_rebase(opts):
    """Re-anchor a campaign on other base tables (e.g. what the campaigns it
    depends on landed), keeping its measured geometry rounds and picks: the
    base, seed and every table built on them are rebuilt, and the A/B stages
    have to be measured again."""
    c = Campaign(opts.name)
    _archive(
        c,
        stages=("ab", "cf", "vf"),
        tables=("base", "seed", "geo", "final", "merged"),
        picks=("geo", "qp", "ab", "cf"),
    )
    T.TableSet.load(opts.base).save(c.tables_dir("base"))
    c._tables.clear()
    c.ensure("seed")
    c.spec["base_source"] = str(opts.base)
    c.spec.setdefault("rebased", []).append(
        {"base": str(opts.base), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    )
    (c.root / "campaign.json").write_text(json.dumps(c.spec, indent=1) + "\n")
    log(
        f"{c.root.name}: rebased on {opts.base}; next: analyze geo, analyze qp, run ab, ..."
    )


PREFLIGHT = r"""
import hashlib, sys
import torch, mori
import mori.ops.dispatch_combine_v2.hip_tuning_configs as h
repo = sys.argv[1]
mine = hashlib.sha256(open(h.__file__, "rb").read()).hexdigest()[:12]
want = hashlib.sha256(open(repo + "/python/mori/ops/dispatch_combine_v2/hip_tuning_configs.py", "rb").read()).hexdigest()[:12]
print(f"torch {torch.__version__} fp4={hasattr(torch, 'float4_e2m1fn_x2')} gpus={torch.cuda.device_count()}")
print(f"mori {mori.__file__}")
print(f"hip_tuning_configs {mine} (repo {want}) {'MATCH' if mine == want else 'DIFFERENT: install this branch'}")
print(f"csv tables: {hasattr(h, 'read_v2_table')}  model={h._gpu.detect_model()} arch={h._gpu.arch_name()} cus={h._gpu.cu_count()}")
"""


def _nodes(m: Machine):
    """(label, command prefix) of node0 and, when FX_PEER is set, node1."""
    import shlex

    nodes = [(f"node0 ({m.node0_label})", lambda argv: argv)]
    if m.peer:
        nodes.append(
            (
                f"node1 ({m.peer_label}, {m.peer})",
                lambda argv: m.ssh + [" ".join(shlex.quote(a) for a in argv)],
            )
        )
    return nodes


def _sh(cmd, timeout):
    import subprocess

    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def cmd_preflight(opts):
    """CPU-only checks inside the container on every node, before a campaign:
    the installed mori is this branch's, torch has FP4, the GPUs are MI355X."""
    m = _machine(opts)
    if not m.peer:
        print(
            "FX_PEER is not set: checking node0 only (enough for EP8, EP16 needs node1)"
        )
    ok = True
    for where, wrap in _nodes(m):
        r = _sh(
            wrap(["podman", "exec", m.container, "python3", "-c", PREFLIGHT, m.repo]),
            180,
        )
        print(f"== {where}: exit {r.returncode}")
        print(r.stdout.strip() or r.stderr.strip()[-800:])
        ok = (
            ok
            and r.returncode == 0
            and "MATCH" in r.stdout
            and "model=mi355x" in r.stdout
        )
    if not ok:
        raise SystemExit(
            "preflight failed: `fxcampaign.py make-container`, then `fxcampaign.py install-mori`"
        )


def cmd_where(opts):
    """The settings in effect, and whether the container runs on each node."""
    m = _machine(opts)
    values = {
        "repo": str(fxconfig.REPO),
        "workdir": str(fxconfig.WORKDIR),
        "container": m.container,
    }
    if opts.get:
        print(values[opts.get])
        return
    env_file = fxconfig.WORKDIR / "machine.env"
    print(f"repo     {fxconfig.REPO}")
    print(f"workdir  {fxconfig.WORKDIR}  (campaigns/, run_logs/, wheels/)")
    print(
        f"settings {env_file}"
        + ("" if env_file.is_file() else "  -- missing: copy machine.env.example there")
    )
    for key in (
        "FX_CONTAINER",
        "FX_IMAGE",
        "FX_CONTAINER_MOUNTS",
        "FX_NODE0_LABEL",
        "FX_PEER",
        "FX_PEER_LABEL",
        "FX_PEER_KEY_ALIAS",
        "FX_MASTER_ADDR",
        "FX_IFNAME",
        "FX_RUN_ENV",
        "FX_CI_FREEZE",
        "FX_RUNNER_DIAG",
        "FX_SYNC",
        "FX_HARNESS_LOGS",
    ):
        print(f"  {key:20s} {fxconfig.get(key, '-')}")
    for where, wrap in _nodes(m):
        try:
            r = _sh(
                wrap(
                    [
                        "podman",
                        "ps",
                        "-a",
                        "--filter",
                        f"name=^{m.container}$",
                        "--format",
                        "{{.Status}}",
                    ]
                ),
                30,
            )
            state = r.stdout.strip() or (
                r.stderr.strip()[-200:] if r.returncode else "no such container"
            )
        except Exception as exc:  # noqa: BLE001 - report and go on
            state = f"unreachable: {exc}"
        print(f"{where}: container {m.container}: {state}")


def cmd_make_container(opts):
    """Create the container on every node (or start it if it exists):
    privileged, host network and IPC, the repo and the work directory at the
    same path as on the host, plus FX_CONTAINER_MOUNTS."""
    m = _machine(opts)
    image = opts.image or fxconfig.get("FX_IMAGE", "localhost/rocm/mori:ci")
    extra = (fxconfig.get("FX_CONTAINER_MOUNTS") or "").split()
    covered = [Path(spec.split(":")[1]) for spec in extra if ":" in spec]
    mounts = list(extra)
    fxconfig.WORKDIR.mkdir(parents=True, exist_ok=True)
    for path in (fxconfig.REPO, fxconfig.WORKDIR):
        if not any(path == c or c in path.parents for c in covered):
            mounts.append(f"{path}:{path}")
    run = [
        "podman",
        "run",
        "-d",
        "--name",
        m.container,
        "--privileged",
        "--network",
        "host",
        "--ipc",
        "host",
        "--group-add",
        "video",
        "--security-opt",
        "label=disable",
    ]
    for spec in mounts:
        run += ["-v", spec]
    run += [image, "sleep", "infinity"]
    for where, wrap in _nodes(m):
        state = _sh(
            wrap(
                [
                    "podman",
                    "ps",
                    "-a",
                    "--filter",
                    f"name=^{m.container}$",
                    "--format",
                    "{{.State}}",
                ]
            ),
            60,
        )
        if state.stdout.strip():
            r = _sh(wrap(["podman", "start", m.container]), 120)
            print(
                f"{where}: {m.container} exists ({state.stdout.strip()}); start: exit {r.returncode} "
                f"{r.stderr.strip()[-300:]}"
            )
        else:
            if m.peer and where.startswith("node1") and m.sync == "rsync":
                for path in (fxconfig.REPO, fxconfig.WORKDIR):
                    _sh(m.ssh + [f"mkdir -p {path}"], 30)
            r = _sh(wrap(run), 1800)
            print(
                f"{where}: podman run {image}: exit {r.returncode} {(r.stderr or r.stdout).strip()[-300:]}"
            )
        if r.returncode:
            raise SystemExit(f"{where}: could not start {m.container}")


def cmd_install_mori(opts):
    """Build this repo's mori in node0's container and install the wheel in the
    container on every node; then preflight."""
    import shlex

    m = _machine(opts)
    wheels = fxconfig.WORKDIR / "wheels" / time.strftime("%Y%m%d_%H%M%S")
    wheels.mkdir(parents=True, exist_ok=True)
    build = (
        f"cd {shlex.quote(m.repo)} && MORI_GPU_ARCHS=gfx950 MORI_PYBUILD_DIR=build_ep_tuning "
        f"python3 -m pip wheel . --no-deps --no-build-isolation -w {shlex.quote(str(wheels))}"
    )
    log(f"building mori in {m.container} on node0 (a few minutes the first time)")
    r = _sh(["podman", "exec", m.container, "bash", "-c", build], 7200)
    found = sorted(wheels.glob("amd_mori-*.whl"))
    if r.returncode or not found:
        print((r.stdout + r.stderr)[-3000:])
        raise SystemExit("mori build failed")
    wheel = found[-1]
    log(f"built {wheel}")
    if m.peer and m.sync == "rsync":
        m.sync_to_peer([wheels])
    install = (
        f"python3 -m pip install --no-deps --force-reinstall {shlex.quote(str(wheel))}"
    )
    for where, wrap in _nodes(m):
        r = _sh(wrap(["podman", "exec", m.container, "bash", "-c", install]), 900)
        print(
            f"{where}: pip install: exit {r.returncode} {r.stderr.strip()[-300:] if r.returncode else ''}"
        )
        if r.returncode:
            raise SystemExit(f"{where}: install failed")
    cmd_preflight(opts)


def cmd_ci(opts):
    """Freeze the node0 runner between two jobs, or resume it. Freezing needs
    FX_CI_FREEZE=1 (scripts call ci-freeze unconditionally). A runner frozen
    here stays frozen across stages: Runner only resumes what it froze itself."""
    from fxrun import CiRunner

    m = Machine()
    ci = CiRunner(m)
    pid = ci.listener_pid()
    if pid is None:
        log("no CI runner (Runner.Listener) on this host; nothing to freeze or resume")
        return
    if opts.cmd == "ci-freeze":
        if not m.ci_freeze:
            log(
                f"Runner.Listener {pid} runs here but FX_CI_FREEZE is not 1: not freezing it "
                f"(its jobs would show up as foreign GPU users and pause the measurements)"
            )
            return
        froze = ci.freeze(opts.ci_wait)
        log(
            f"Runner.Listener {pid}: "
            + ("frozen now" if froze else "was already frozen")
        )
    elif opts.cmd == "ci-resume":
        ci.resume()
    print(
        f"listener {pid} state={ci.proc_state(pid) if pid else None} "
        f"worker_running={ci.worker_running()} last_job_state={ci.last_job_state()}"
    )


def cmd_status(opts):
    c = Campaign(opts.name)
    for stage in ("g1", "g2", "q", "ab", "cf", "vf"):
        try:
            entries = c.stage_entries(stage)
        except (FileNotFoundError, SystemExit, KeyError):
            print(f"{stage}: not ready")
            continue
        recs = [
            json.loads(Path(e["evidence"]).read_text())
            for e in entries
            if Path(e["evidence"]).exists()
        ]
        line = f"{stage}: {len(recs)}/{len(entries)} measured"
        real = [r for r in recs if r.get("host") != "sim" and "t_end" in r]
        if real:
            per_cell = statistics.fmean(r["t_end"] - r["t_start"] for r in real)
            span = max(r["t_end"] for r in real) - min(r["t_start"] for r in real)
            line += (
                f", {per_cell:.1f} s/cell inside the driver, {span / len(real):.1f} s/cell wall "
                f"(estimate assumes {SECONDS_PER_CELL})"
            )
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("name")
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--base", default=str(T.REPO_TABLES))
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("plan")
    p.add_argument("name")
    p.add_argument("stage", nargs="?")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("archive")
    p.add_argument("name")
    p.add_argument("stages", nargs="+", choices=["g", "q", "ab", "cf", "vf"])
    p.set_defaults(func=cmd_archive)

    p = sub.add_parser("rebase")
    p.add_argument("name")
    p.add_argument("--base", required=True)
    p.set_defaults(func=cmd_rebase)

    for name, func in (("run", cmd_run), ("chain", cmd_chain)):
        p = sub.add_parser(name)
        p.add_argument("name")
        if name == "run":
            p.add_argument("stage", choices=["g1", "g2", "q", "ab", "cf", "vf"])
        p.add_argument("--dry-run", action="store_true")
        p.add_argument(
            "--simulate",
            action="store_true",
            help="write synthetic evidence instead of measuring (offline test)",
        )
        p.add_argument("--chunk", type=int, default=200)
        p.add_argument(
            "--limit", type=int, help="measure only the first N entries (smoke test)"
        )
        p.add_argument(
            "--sample",
            type=int,
            help="measure N entries spread over the stage (smoke test)",
        )
        p.add_argument(
            "--quiet",
            type=int,
            default=120,
            help="seconds of idle GPUs before a launch",
        )
        p.add_argument("--ci-wait", type=int, default=3600)
        for flag in (
            "container",
            "peer",
            "peer-label",
            "peer-key-alias",
            "master-addr",
            "ifname",
        ):
            p.add_argument(f"--{flag}")
        p.set_defaults(func=func)

    p = sub.add_parser("analyze")
    p.add_argument("name")
    p.add_argument("stage", choices=sorted(ANALYSES))
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("land")
    p.add_argument("name")
    p.add_argument(
        "--into", help="a tables directory to update in place (e.g. the repo's)"
    )
    p.set_defaults(func=cmd_land)

    p = sub.add_parser("status")
    p.add_argument("name")
    p.set_defaults(func=cmd_status)

    for name in ("ci-freeze", "ci-resume", "ci-status"):
        p = sub.add_parser(name)
        p.add_argument("--ci-wait", type=int, default=3 * 3600)
        p.set_defaults(func=cmd_ci)

    for name, func in (
        ("preflight", cmd_preflight),
        ("where", cmd_where),
        ("make-container", cmd_make_container),
        ("install-mori", cmd_install_mori),
    ):
        p = sub.add_parser(name)
        for flag in (
            "container",
            "peer",
            "peer-label",
            "peer-key-alias",
            "master-addr",
            "ifname",
        ):
            p.add_argument(f"--{flag}")
        if name == "where":
            p.add_argument(
                "--get",
                choices=["repo", "workdir", "container"],
                help="print only this value",
            )
        if name == "make-container":
            p.add_argument(
                "--image", help="default FX_IMAGE, else localhost/rocm/mori:ci"
            )
        p.set_defaults(func=func)

    opts = ap.parse_args(argv)
    opts.func(opts)


if __name__ == "__main__":
    main()
```

#### `fxharness.py`

```python
#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Round-1 pre-pruning from harness geometry sweeps already on disk.

The 2026-10-12/13 MI355X campaigns (mori_v2_refresh_20261012, on /shared) swept
every candidate of the harness's quick grid at T64-4096 for several shapes,
each a paired median against the shipped geometry. Their conditions are not
the protocol's (2 QPs allocated, scale_dim 32, the kernel of that week), so
they never select anything here; they only drop candidates that were far from
the best of their sweep, before round 1 measures the rest. Optional: without
logs round 1 measures every candidate.

  python3 fxharness.py NAME [--window 0.10] [--min-keep 8] [--logs DIR ...]

The log directories default to FX_HARNESS_LOGS (colon-separated), e.g.
  .../mori_v2_refresh_20261012/logs_ll16:.../logs_ep8:.../logs_dsv41:.../logs_dsv4pro

writes the campaign's picks/r1_lists.json. A rule the logs do not cover keeps
every candidate in round 1. A covered rule keeps the harness candidates within
`window` of its sweep's best (the incumbent included), at least `min_keep` of
them, and every campaign candidate the quick grid does not contain.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fxconfig  # noqa: E402
from fxcells import Shape  # noqa: E402

HEADER = re.compile(r"# TUNING tok=(\d+) phase=(\w+) .*?candidates=(\d+)")
LINE = re.compile(
    r"#\s+\[\d+/\d+\] \((\d+), (\d+), (\d+)\) med=\s*([\d.]+)us vs incumbent \((\d+), (\d+), (\d+)\) med=\s*([\d.]+)us"
)
NAMED = re.compile(
    r"^sweep_(v2_ll|v2|intra)_(dispatch|combine)_(bf16|fp8_e4m3)_t(\d+)_a(\d+)(?:_node0)?\.log$"
)
PLAIN = re.compile(
    r"^sweep_(dispatch|combine)_(bf16|fp8_e4m3)_k(\d+)_t(\d+)_a(\d+)(?:_node0)?\.log$"
)

# Which family and shape each old campaign's log directory swept.
SOURCES = {
    "logs_ll16": ("v2_ll", lambda topk: Shape(256, topk, 7168)),
    "logs_ep8": ("intra", lambda topk: Shape(256, topk, 7168)),
    "logs_dsv41": (None, lambda topk: Shape(384, 6, 5120)),
    "logs_dsv4pro": (None, lambda topk: Shape(384, 6, 7168)),
}


def parse_sweep(path: Path):
    """{tokens, phase, values: {(b, r, w): median total us}} of one sweep log,
    the incumbent included; None if the log holds no complete sweep."""
    text = path.read_text(errors="replace")
    header = HEADER.search(text)
    if header is None:
        return None
    values = {}
    for m in LINE.finditer(text):
        values[tuple(map(int, m.group(1, 2, 3)))] = float(m.group(4))
        values.setdefault(tuple(map(int, m.group(5, 6, 7))), float(m.group(8)))
    if len(values) < 10:
        return None
    return {"tokens": int(header.group(1)), "phase": header.group(2), "values": values}


def collect(log_dirs):
    """{(family, shape tag, phase, dtype, tokens): sweep}, the latest attempt."""
    found = {}
    for directory in log_dirs:
        directory = Path(directory)
        fixed_family, shape_of = SOURCES.get(directory.name, (None, None))
        if shape_of is None:
            continue
        for path in sorted(directory.glob("sweep_*.log")):
            named, plain = NAMED.match(path.name), PLAIN.match(path.name)
            if named:
                family, phase, dtype, tokens, attempt = named.groups()
                shape = shape_of(None)
            elif plain and fixed_family:
                phase, dtype, topk, tokens, attempt = plain.groups()
                family, shape = fixed_family, shape_of(int(topk))
            else:
                continue
            sweep = parse_sweep(path)
            if (
                sweep is None
                or sweep["phase"] != phase
                or sweep["tokens"] != int(tokens)
            ):
                continue
            key = (family, shape.tag, phase, dtype, int(tokens))
            if key not in found or int(attempt) >= found[key][0]:
                found[key] = (int(attempt), sweep, str(path))
    return {k: (v[1], v[2]) for k, v in found.items()}


def r1_lists(campaign, sweeps, window=0.10, min_keep=8, probes=2, seed=7):
    """(lists, probes, covered). `probes` candidates the harness would have
    dropped are kept per covered rule, so round 1 can show whether the old
    sweeps still rank this kernel and protocol (fxcampaign analyze prune)."""
    import random

    rng = random.Random(seed)
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    probe_out = collections.defaultdict(lambda: collections.defaultdict(dict))
    covered = 0
    for group in campaign.groups:
        for phase in group.owns:
            pool = [tuple(g) for g in campaign.candidates[group.family][phase]]
            for cell in group.cells():
                # Combine was swept on the bf16-dispatch cell only.
                if (
                    phase == "combine"
                    and cell.dispatch_dtype != "bf16"
                    and "bf16" in group.dispatch_dtypes
                ):
                    continue
                key = (
                    group.family,
                    group.shape.tag,
                    phase,
                    cell.dispatch_dtype,
                    cell.tokens,
                )
                if key not in sweeps:
                    continue
                values = sweeps[key][0]["values"]
                best = min(values.values())
                ranked = sorted((v, g) for g, v in values.items())
                keep = {g for v, g in ranked if v <= best * (1 + window)}
                for _, g in ranked:
                    if len(keep) >= min_keep:
                        break
                    keep.add(g)
                # 3-field harness geometries; FP4 candidates carry a QP count.
                kept = [
                    g
                    for g in pool
                    if tuple(g[:3]) in keep or tuple(g[:3]) not in values
                ]
                dropped = [g for g in pool if g not in kept]
                chosen = rng.sample(dropped, min(probes, len(dropped)))
                sel = json.dumps(list(cell.selector(phase)))
                out[group.family][phase][sel] = [list(g) for g in kept + chosen]
                probe_out[group.family][phase][sel] = [list(g) for g in chosen]
                covered += 1
    return (
        {f: dict(p) for f, p in out.items()},
        {f: dict(p) for f, p in probe_out.items()},
        covered,
    )


def main(argv=None):
    import fxcampaign

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("name")
    ap.add_argument("--window", type=float, default=0.10)
    ap.add_argument("--min-keep", type=int, default=8)
    ap.add_argument(
        "--probes",
        type=int,
        default=2,
        help="dropped candidates kept per covered rule as a check on the logs",
    )
    ap.add_argument(
        "--logs", nargs="*", default=[str(p) for p in fxconfig.harness_log_dirs()]
    )
    opts = ap.parse_args(argv)
    campaign = fxcampaign.Campaign(opts.name)
    logs = [p for p in opts.logs if Path(p).is_dir()]
    if not logs:
        print(
            "no harness sweep logs (FX_HARNESS_LOGS); round 1 measures every candidate"
        )
        return
    sweeps = collect(logs)
    lists, probes, covered = r1_lists(
        campaign, sweeps, opts.window, opts.min_keep, opts.probes
    )
    campaign.save_picks("r1_lists", lists)
    campaign.save_picks("r1_probes", probes)
    before = after = 0
    for fam, phases in lists.items():
        for phase, rules in phases.items():
            for kept in rules.values():
                before += len(campaign.candidates[fam][phase])
                after += len(kept)
    print(
        f"{len(sweeps)} sweeps on disk; {covered} rules covered; "
        f"round-1 candidates on covered rules {after}/{before} ({after / max(before, 1):.0%})"
    )


if __name__ == "__main__":
    main()
```

#### `apply_existing.py`

```python
#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Write the EP16 results measured on 2026-11-02/03 that hold as they are into
the repo's MI355X tables (a working-tree change, nothing committed).

Per campaign the whole-table A/B (ab) decided which buckets take the new rules
and the confirm A/B (cf) measured both arms. What is written:

* an accepted bucket: its rules from the campaign's merged tables, metric
  columns from the cf 'merged' arm;
* any other bucket the campaign owns that has no rule in the repo tables: a
  rule pinning what it runs today (fxtables.pinned_rule), metric columns from
  the cf 'base' arm -- without it the bucket would run a neighbour's new rule;
* not written as accepted:
  - e16-new V2LL 896/16 T2048 and T4096, accepted on a dispatch geometry that
    gave wrong results on that shape (256 blocks x 16 warps); pinned instead;
  - an e16-fp4 / e16-fp8c bucket whose other phase (the bf16 combine rule for
    FP4 dispatch, the fp8 dispatch rule for the FP8 combine) differs here from
    what the campaign measured against; pinned instead.

  python3 apply_existing.py [--into DIR]      (default: the repo's tuning_configs)

Needs that run's campaigns in $FX_WORKDIR/campaigns.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fxanalyze as A  # noqa: E402
import fxcampaign as F  # noqa: E402
import fxtables as T  # noqa: E402

KIMI_WRONG = {("v2_ll", "896x16h3584", 2048), ("v2_ll", "896x16h3584", 4096)}
SPEC_FIELDS = (
    "family",
    "block_num",
    "rdma_block_num",
    "warp_per_block",
    "active_qps",
    "load_first",
)


def latest_archive_with(c, rel):
    found = (
        sorted(p for p in (c.root / "archive").glob("*") if (p / rel).exists())
        if (c.root / "archive").exists()
        else []
    )
    return found[-1] / rel if found else None


def sources(c):
    """(ab picks, merged tables, cf evidence dir) of the last complete A/B."""
    if (
        c.picks_path("ab").exists()
        and c.evidence_dir("cf").exists()
        and any(c.evidence_dir("cf").rglob("*.json"))
    ):
        return (
            c.load_picks("ab"),
            T.TableSet.load(c.tables_dir("merged")),
            c.evidence_dir("cf"),
        )
    import json

    picks = latest_archive_with(c, "picks/ab.json")
    picks = json.loads(picks.read_text()) if picks else c.load_picks("ab")
    merged = latest_archive_with(c, "tables/merged") or c.tables_dir("merged")
    cf = latest_archive_with(c, "evidence/cf")
    return picks, T.TableSet.load(merged), cf


def spec(e):
    return tuple(e.get(f) for f in SPEC_FIELDS)


def apply(c, target, head_dir, other_phase_check=False, excluded=frozenset()):
    picks, merged, cf_dir = sources(c)
    idx = A.index(A.load_records(cf_dir))
    accepted = {(r["family"], r["shape"], r["tokens"]) for r in picks if r.get("take")}
    head = T.TableSet.load(head_dir)
    if other_phase_check:
        tmp = Path(tempfile.mkdtemp())
        target.save(tmp / "now")
    taken = pinned = 0
    notes = []
    for g in c.groups:
        by_t = {}
        for cell in g.cells():
            by_t.setdefault(cell.tokens, []).append(cell)
        for tokens, cells in sorted(by_t.items()):
            bucket = (g.family, g.shape.tag, tokens)
            take = bucket in accepted and bucket not in excluded
            if take and other_phase_check:
                other = "combine" if g.owns == ("dispatch",) else "dispatch"
                for cell in cells:
                    then = T.expected(c.tables_dir("base"), cell)[other]
                    now = T.expected(tmp / "now", cell)[other]
                    if spec(then) != spec(now):
                        take = False
                        notes.append(
                            f"{c.root.name} {g.family} {g.shape.tag} T{tokens}: {other} rule changed "
                            f"since it was measured; pinned"
                        )
                        break
            if bucket in excluded and bucket in accepted:
                notes.append(
                    f"{c.root.name} {g.family} {g.shape.tag} T{tokens}: accepted on a wrong-result "
                    f"geometry; pinned"
                )
            for cell in cells:
                for phase in g.owns:
                    sel = cell.selector(phase)
                    metric_cell = A.metric_cell(g, phase, sel)
                    if take:
                        rule = {
                            k: v
                            for k, v in merged.get(g.family, sel).items()
                            if k not in T.METRIC_FIELDS
                        }
                        arm = "merged"
                        taken += 1
                    elif head.get(g.family, sel) is None:
                        rule = {
                            **T.rule_from_selector(sel),
                            **T.pinned_rule(head_dir, cell, phase),
                        }
                        arm = "base"
                        pinned += 1
                    else:
                        target.rules[g.family][sel] = dict(head.get(g.family, sel))
                        continue
                    metrics = A.landing_metrics(idx, metric_cell, arm, phase)
                    if metrics:
                        rule.update({k: v for k, v in metrics.items() if v is not None})
                    target.rules[g.family][sel] = rule
    return taken, pinned, notes


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--into", default=str(T.REPO_TABLES))
    opts = ap.parse_args(argv)
    # The tables every campaign of that run started from (git HEAD 7148de8bc).
    head_dir = F.Campaign("e16-dsr1").tables_dir("base")
    target = T.TableSet.load(head_dir)
    for name, check, excluded in (
        ("e16-dsr1", False, frozenset()),
        ("e16-rest", False, frozenset()),
        ("e16-new", False, KIMI_WRONG),
        ("e16-fp4", True, frozenset()),
        ("e16-fp8c", True, frozenset()),
    ):
        taken, pinned, notes = apply(
            F.Campaign(name), target, head_dir, check, excluded
        )
        print(
            f"{name}: {taken} rule writes from accepted buckets, {pinned} pinning today's configuration"
        )
        for note in notes:
            print("   ", note)
    target.write_into(opts.into)
    for fam in ("v2", "v2_ll", "intra"):
        print(f"{T.table_name(fam)}: {len(T.TableSet.load(opts.into).rules[fam])} rows")


if __name__ == "__main__":
    main()
```

#### `remaining.sh`

```bash
#!/bin/bash
# What is left of the 2026-11 MI355X EP16 campaign, and EP8
# (qp_status_mi355x.md). Needs the five e16-* campaigns of that run in
# $FX_WORKDIR/campaigns (copied from the old work directory).
#
#   [START_AT=21:00] ./remaining.sh
#
# Machine settings: $FX_WORKDIR/machine.env (machine.env.example).
#
# 1. e16-dsr1, e16-rest: confirm A/B + the 20-check correctness pass (vf) on
#    the re-merged tables.
# 2. e16-new: A/B, confirm A/B, vf on the re-picked tables.
# 3. $FX_WORKDIR/landed/: the original MI355X tables + those three campaigns.
# 4. e16-fp4, e16-fp8c re-based on it, then A/B, confirm A/B, vf, landed.
# 5. EP8 (preset ep8), landed into the same directory.
# 6. landed/'s three MI355X tables copied over the repo's (working tree).
set -u
cd "$(dirname "$0")"
REPO=$(python3 fxcampaign.py where --get repo) || exit 1
WORK=$(python3 fxcampaign.py where --get workdir) || exit 1
TABLES_REL=python/mori/ops/dispatch_combine_v2/tuning_configs
# The tables every campaign of that run started from (git HEAD 7148de8bc).
ORIG=$WORK/campaigns/e16-dsr1/tables/base
OUT=$WORK/landed
Q="--quiet 30"

for n in e16-dsr1 e16-rest e16-new e16-fp4 e16-fp8c; do
  [ -f "$WORK/campaigns/$n/campaign.json" ] || { echo "missing $WORK/campaigns/$n (copy the old campaigns/ there)"; exit 1; }
done
if [ -n "${START_AT:-}" ]; then
  echo "=== waiting until $START_AT"
  while [ "$(date +%s)" -lt "$(date -d "$START_AT" +%s)" ]; do sleep 30; done
fi
step() { echo "=== $(date '+%F %T') $*"; python3 fxcampaign.py "$@"; }

python3 fxcampaign.py preflight || exit 1
step ci-freeze --ci-wait 10800 || { echo "=== CI never idle; nothing run"; exit 1; }

for n in e16-dsr1 e16-rest; do
  step run "$n" cf $Q
  step analyze "$n" cf
  step run "$n" vf $Q
done
step run e16-new ab $Q
step analyze e16-new ab
step run e16-new cf $Q
step analyze e16-new cf
step run e16-new vf $Q

if [ -d "$OUT" ]; then mv "$OUT" "$WORK/archive_landed_$(date +%m%d_%H%M%S)"; fi
cp -r "$ORIG" "$OUT"
for n in e16-dsr1 e16-rest e16-new; do step land "$n" --into "$OUT"; done

for n in e16-fp4 e16-fp8c; do
  step rebase "$n" --base "$OUT"
  step analyze "$n" geo
  step analyze "$n" qp
  step run "$n" ab $Q
  step analyze "$n" ab
  step run "$n" cf $Q
  step analyze "$n" cf
  step run "$n" vf $Q
  step land "$n" --into "$OUT"
done

[ -d "$WORK/campaigns/e8" ] || step init e8 --preset ep8 --base "$ORIG"
[ -f "$WORK/campaigns/e8/picks/r1_lists.json" ] || python3 fxharness.py e8
step chain e8 $Q
step land e8 --into "$OUT"

cp "$OUT"/gfx950_mi355x_*.csv "$REPO/$TABLES_REL/"
git -C "$REPO" diff --stat -- "$TABLES_REL"
for n in e16-dsr1 e16-rest e16-new e16-fp4 e16-fp8c e8; do step status "$n"; done
step ci-resume
echo "=== $(date '+%F %T') remaining done"
```

#### `run_all.sh`

```bash
#!/bin/bash
# The whole MI355X campaign from scratch: EP16 (five campaigns) and EP8.
#
#   [TAG=full] [BASE=DIR] [START_AT=21:00] ./run_all.sh
#
# Campaigns are $FX_WORKDIR/campaigns/$TAG-{dsr1,rest,new,fp4,fp8c,ep8}; a
# campaign that exists is continued, not restarted. BASE is the tables
# directory to tune against (default: git HEAD's MI355X tables). dsr1, rest,
# new and ep8 start from BASE; they land into $FX_WORKDIR/landed-$TAG/, and
# fp4 / fp8c are tuned on top of that (FP4 dispatch against the new bf16
# combine rules, the FP8 combine against the new fp8 dispatch rules). At the
# end landed-$TAG/'s MI355X tables are copied into the repo (working tree).
#
# Each campaign first measures 24 sampled round-1 entries; if any is not
# valid the campaign is skipped. About 11-12 hours on 2 x 8 MI355X.
set -u
cd "$(dirname "$0")"
TAG=${TAG:-full}
REPO=$(python3 fxcampaign.py where --get repo) || exit 1
WORK=$(python3 fxcampaign.py where --get workdir) || exit 1
TABLES_REL=python/mori/ops/dispatch_combine_v2/tuning_configs
OUT=$WORK/landed-$TAG
Q="--quiet 30"

if [ -z "${BASE:-}" ]; then
  BASE=$WORK/base-$TAG
  if [ ! -d "$BASE" ]; then
    mkdir -p "$BASE"
    for f in gfx950_mi355x_InterNodeV2_ep16.csv gfx950_mi355x_InterNodeV2LL_ep16.csv gfx950_mi355x_IntraNodeV2_ep8.csv; do
      git -C "$REPO" show "HEAD:$TABLES_REL/$f" > "$BASE/$f" || exit 1
    done
  fi
fi
if [ -n "${START_AT:-}" ]; then
  echo "=== waiting until $START_AT"
  while [ "$(date +%s)" -lt "$(date -d "$START_AT" +%s)" ]; do sleep 30; done
fi

smoke_ok() {
  python3 fxcampaign.py run "$1" g1 --sample 24 --chunk 24 $Q
  python3 - "$1" <<'EOF'
import sys
import fxanalyze as A
import fxcampaign as F
c = F.Campaign(sys.argv[1])
recs = c.records("g")
bad = [(r["id"], A.problems(r), r.get("error")) for r in recs if A.problems(r)]
print(f"smoke {sys.argv[1]}: {len(recs)} records, {len(bad)} with problems")
for item in bad[:10]:
    print("   ", item)
sys.exit(1 if bad or not recs else 0)
EOF
}

campaign() {  # name preset base
  local name=$1 preset=$2 base=$3
  echo "=== $(date '+%F %T') $name ($preset) base=$base"
  [ -d "$WORK/campaigns/$name" ] || python3 fxcampaign.py init "$name" --preset "$preset" --base "$base" || return 1
  [ -f "$WORK/campaigns/$name/picks/r1_lists.json" ] || python3 fxharness.py "$name"
  if ! smoke_ok "$name"; then
    echo "=== $(date '+%F %T') $name: smoke failed, skipped"
    return 1
  fi
  python3 fxcampaign.py chain "$name" $Q
  python3 fxcampaign.py land "$name" --into "$OUT"
  python3 fxcampaign.py status "$name"
}

python3 fxcampaign.py preflight || exit 1
python3 fxcampaign.py ci-freeze --ci-wait 10800 || { echo "=== CI never idle; nothing run"; exit 1; }
echo "=== $(date '+%F %T') run_all $TAG start (base $BASE)"
[ -d "$OUT" ] || cp -r "$BASE" "$OUT"

campaign "$TAG-dsr1" ep16-dsr1 "$BASE" || true
campaign "$TAG-rest" ep16-existing-rest "$BASE" || true
campaign "$TAG-new" ep16-new "$BASE" || true
campaign "$TAG-fp4" ep16-fp4 "$OUT" || true
campaign "$TAG-fp8c" ep16-fp8c "$OUT" || true
campaign "$TAG-ep8" ep8 "$BASE" || true

cp "$OUT"/gfx950_mi355x_*.csv "$REPO/$TABLES_REL/"
git -C "$REPO" diff --stat -- "$TABLES_REL"
python3 fxcampaign.py ci-resume
echo "=== $(date '+%F %T') run_all $TAG done"
```

#### `fxsim.py`

```python
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""A synthetic stand-in for the driver, for exercising the campaign offline.

Writes evidence in the driver's schema from a made-up latency model of the
configuration a cell runs (its `expect`), with noise and the occasional stall.
The model only has to give the analysis something with structure: an optimum
geometry that moves with the token count, QP counts and load-first that help
at some sizes and hurt at others, and the (d < c) dispatch drain penalty.
Nothing here says anything about real MI355X numbers.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path


def _geometry_cost(tokens, block, rdma, warp, best_block, best_warp):
    shape = (
        1
        + 0.15 * abs(math.log2(block / best_block))
        + 0.08 * abs(math.log2(warp / best_warp))
    )
    if rdma:
        frac = rdma / block
        shape *= 1 + 0.3 * abs(frac - 0.6)
    return shape


def latency(cell, expect):
    t, h = cell["tokens"], cell["hidden"]
    size = t * h / 7168
    d, c = expect["dispatch"], expect["combine"]
    best_block = 32 if t <= 64 else 128 if t <= 512 else 256
    best_warp = 8 if t <= 256 else 16
    base_d = 60 + 0.35 * size * (
        0.6
        if cell["dispatch_dtype"] == "fp8_e4m3"
        else 0.35 if cell["dispatch_dtype"] == "fp4" else 1.0
    )
    base_c = 55 + 0.38 * size
    disp = base_d * _geometry_cost(
        t,
        d["block_num"],
        d["rdma_block_num"] or 0,
        d["warp_per_block"],
        best_block,
        best_warp,
    )
    comb = base_c * _geometry_cost(
        t,
        c["block_num"],
        c["rdma_block_num"] or 0,
        c["warp_per_block"],
        best_block,
        best_warp // 2 or 4,
    )
    dq, cq = d.get("active_qps") or 2, c.get("active_qps") or 2
    if cell["family"] != "intra":
        comb *= 1 - 0.06 * math.log2(cq / 2) * (1 if t >= 512 else -1)
        disp *= 1 + 0.02 * (cq > dq) * (cq - dq)
        if c.get("load_first"):
            comb *= 0.9 if t <= 256 else 1.03
    return disp, comb


# Test hook: a predicate on (entry) -> True makes that measurement a wrong result.
WRONG_IF = None


def write(entry, protocol, rng, stall=False):
    cell, expect = entry["cell"], entry["expect"]
    if WRONG_IF is not None and WRONG_IF(entry):
        _dump(
            entry,
            {
                "schema": 1,
                "id": entry["id"],
                "stage": entry["stage"],
                "round": entry["round"],
                "arm": entry.get("arm"),
                "variant": entry["variant"],
                "cell": cell,
                "cell_key": entry["cell_key"],
                "expect": expect,
                "effective": expect,
                "protocol": protocol,
                "host": "sim",
                "t_start": time.time(),
                "t_end": time.time(),
                "verified": False,
                "failed_ranks": 1,
                "status": "wrong",
            },
        )
        return
    if protocol.get("timing") is False:
        _dump(
            entry,
            {
                "schema": 1,
                "id": entry["id"],
                "stage": entry["stage"],
                "round": entry["round"],
                "arm": entry.get("arm"),
                "variant": entry["variant"],
                "cell": cell,
                "cell_key": entry["cell_key"],
                "expect": expect,
                "effective": expect,
                "protocol": protocol,
                "host": "sim",
                "t_start": time.time(),
                "t_end": time.time(),
                "verified": True,
                "failed_ranks": 0,
                "failed_checks": 0,
                "status": "ok",
            },
        )
        return
    disp, comb = latency(cell, expect)
    noise = lambda: math.exp(rng.gauss(0, 0.01))  # noqa: E731
    factor = rng.uniform(5, 30) if stall else 1.0
    d = disp * noise() * factor
    c = comb * noise() * factor
    per_pass = [(d + c) * noise() for _ in range(protocol["passes"])]

    def phase(m):
        return {
            "median": m,
            "mean": m * 1.01,
            "min": m * 0.97,
            "max": m * 1.3,
            "mean_over_median": 1.01,
            "max_over_median": 1.3,
        }

    gates = {
        "dispatch_mean_median": True,
        "combine_mean_median": True,
        "dispatch_max_median": True,
        "combine_max_median": True,
        "pass_spread": max(per_pass) / min(per_pass) <= 1.2,
    }

    def metric(m):
        return {
            "bandwidth_gbps": round(1e3 / m, 2),
            "avg_rdma_bandwidth_gbps": round(1e3 / m, 2),
            "avg_xgmi_bandwidth_gbps": round(3e3 / m, 2),
            "avg_ll_bandwidth_gbps": round(3e3 / m, 2),
            "avg_latency_us": round(m * 1.01, 2),
            "bandwidth_metric": "grand_mean",
        }

    now = time.time()
    record = {
        "schema": 1,
        "id": entry["id"],
        "stage": entry["stage"],
        "round": entry["round"],
        "arm": entry.get("arm"),
        "variant": entry["variant"],
        "cell": cell,
        "cell_key": entry["cell_key"],
        "table_dir": entry["table_dir"],
        "expect": expect,
        "effective": expect,
        "launch": "sim",
        "host": "sim",
        "protocol": protocol,
        "t_start": now,
        "t_end": now,
        "verified": True,
        "status": "ok",
        "failed_ranks": 0,
        "metrics": {"dispatch": metric(d), "combine": metric(c)},
        "stats": {
            "dispatch": phase(d),
            "combine": phase(c),
            "total_median": d + c,
            "total_mean": (d + c) * 1.01,
            "per_pass_total_mean": per_pass,
            "pass_spread": max(per_pass) / min(per_pass),
            "gates": gates,
            "gates_ok": all(gates.values()),
            "samples_per_phase": 1392,
        },
    }
    _dump(entry, record)


def _dump(entry, record):
    path = Path(entry["evidence"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def simulate(entries, protocol, seed=0, stall_rate=0.01):
    """Evidence for every entry that has none; stalls come in runs of 3-5
    consecutive entries, as on the real nodes."""
    stage = entries[0]["stage"] if entries else ""
    rng = random.Random(f"{seed}-{stage}-{os.getpid()}-{time.time()}")
    stall_left = 0
    for entry in entries:
        if Path(entry["evidence"]).exists():
            continue
        if stall_left == 0 and rng.random() < stall_rate:
            stall_left = rng.randint(3, 5)
        write(entry, protocol, rng, stall=stall_left > 0)
        stall_left = max(0, stall_left - 1)
```

#### `tests/test_fx.py`

```python
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Offline tests: no torch, no GPU. python3 -m unittest discover -s tests -v"""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import fxanalyze as A  # noqa: E402
import fxcampaign  # noqa: E402
import fxconfig  # noqa: E402
import fxdriver  # noqa: E402
import fxharness  # noqa: E402
import fxtables as T  # noqa: E402
from fxcells import Cell, Group, Shape  # noqa: E402

SHAPE = Shape(256, 8, 7168)
# The MI355X tables as they were before the 2026-11 campaign (77 rows each).
BASE = HERE / "tests" / "fixtures" / "base_tables"


class TablesTest(unittest.TestCase):
    def test_selector_is_the_repo_table_key(self):
        for cell in (
            Cell("v2", SHAPE, "bf16", "bf16", 64),
            Cell("v2", SHAPE, "fp4", "bf16", 4),
            Cell("intra", Shape(264, 9, 7168, "replica"), "fp8_e4m3", "bf16", 4096),
        ):
            for phase in ("dispatch", "combine"):
                rule = T.rule_from_selector(cell.selector(phase))
                self.assertEqual(T.rule_key(rule), cell.selector(phase))
        self.assertEqual(
            Cell("v2", SHAPE, "fp4", "bf16", 4).selector("dispatch")[2], 3584
        )

    def test_repo_tables_round_trip(self):
        base = T.TableSet.load(T.REPO_TABLES)
        with tempfile.TemporaryDirectory() as tmp:
            base.save(Path(tmp) / "v")
            for fam in ("v2", "v2_ll", "intra"):
                name = T.table_name(fam)
                self.assertEqual(
                    (Path(tmp) / "v" / name).read_text(),
                    (T.REPO_TABLES / name).read_text(),
                )

    def test_seed_materializes_the_geometry_each_bucket_runs(self):
        base = T.TableSet.load(BASE)
        group = Group("v2", SHAPE, ("bf16", "fp8_e4m3"))
        with tempfile.TemporaryDirectory() as tmp:
            base.save(Path(tmp) / "base")
            seed = T.seed(base, [group])
            seed.save(Path(tmp) / "seed")
            for tokens in (4, 8, 16, 32, 64, 4096):
                cell = Cell("v2", SHAPE, "fp8_e4m3", "bf16", tokens)
                e_base = T.expected(Path(tmp) / "base", cell)
                e_seed = T.expected(Path(tmp) / "seed", cell)
                for phase in ("dispatch", "combine"):

                    def geom(e):
                        return tuple(
                            e[phase][k]
                            for k in ("block_num", "rdma_block_num", "warp_per_block")
                        )

                    self.assertEqual(geom(e_base), geom(e_seed), (tokens, phase))
                    self.assertFalse(e_seed["combine"]["load_first"])
                    self.assertIsNotNone(seed.get("v2", cell.selector(phase)))

    def test_qp_variant_needs_rules_and_sets_them(self):
        base = T.TableSet.load(BASE)
        group = Group("v2", SHAPE, ("bf16",))
        with self.assertRaises(KeyError):
            T.with_qp(base, [group], "v2", 1, 4, True)  # no T4 rules yet
        seed = T.seed(base, [group])
        q = T.with_qp(seed, [group], "v2", 1, 4, True)
        with tempfile.TemporaryDirectory() as tmp:
            q.save(Path(tmp) / "q")
            e = T.expected(Path(tmp) / "q", Cell("v2", SHAPE, "bf16", "bf16", 4))
            self.assertEqual(
                (
                    e["dispatch"]["active_qps"],
                    e["combine"]["active_qps"],
                    e["combine"]["load_first"],
                ),
                (1, 4, True),
            )
        with self.assertRaises(ValueError):
            T.with_qp(seed, [Group("v2_ll", SHAPE, ("bf16",))], "v2_ll", 2, 2, True)


class MergeTest(unittest.TestCase):
    def test_unaccepted_bucket_without_a_rule_keeps_todays_configuration(self):
        # FP4 dispatch has no rules today: every bucket runs the default. Take
        # only T512; T64 must not start running T512's rule.
        base = T.TableSet.load(BASE)
        group = Group("v2_ll", SHAPE, ("fp4",), "bf16", (64, 512), ("dispatch",))
        final = T.with_geometry(
            T.seed(base, [group]), [group], "v2_ll", "dispatch", (256, 170, 8, 4)
        )
        with tempfile.TemporaryDirectory() as tmp:
            base.save(Path(tmp) / "base")
            accepted = [("v2_ll", SHAPE.tag, 512)]
            for pin, want in ((False, (256, 170, 8, 4)), (True, (96, 64, 8, 2))):
                merged = T.merge_buckets(
                    base,
                    final,
                    [group],
                    accepted,
                    base_dir=(Path(tmp) / "base") if pin else None,
                )
                merged.save(Path(tmp) / f"m{pin}")
                e = T.expected(
                    Path(tmp) / f"m{pin}", Cell("v2_ll", SHAPE, "fp4", "bf16", 64)
                )["dispatch"]
                self.assertEqual(
                    (
                        e["block_num"],
                        e["rdma_block_num"],
                        e["warp_per_block"],
                        e["active_qps"],
                    ),
                    want,
                )
            e = T.expected(
                Path(tmp) / "mTrue", Cell("v2_ll", SHAPE, "fp4", "bf16", 512)
            )["dispatch"]
            self.assertEqual((e["block_num"], e["active_qps"]), (256, 4))


class DriverStatsTest(unittest.TestCase):
    def rows(self, d, c, spike=None):
        passes, rounds = 3, 30
        out = []
        for rank in range(4):
            row = [100, 50, 14336, 14336]
            for p in range(passes):
                for r in range(rounds):
                    dv = d * (1 + 0.001 * ((rank + p + r) % 5))
                    cv = c * (1 + 0.001 * ((rank * 3 + r) % 7))
                    if spike and (rank, p, r) == spike:
                        dv *= 50
                    row += [dv, cv]
            out.append(row)
        return out

    def test_gates(self):
        s = fxdriver.summarize(self.rows(100.0, 120.0), 3, 30, 1)
        self.assertTrue(s["gates_ok"])
        self.assertAlmostEqual(s["dispatch"]["median"], 100.1, delta=0.5)
        self.assertEqual(s["samples_per_phase"], 4 * 3 * 29)
        spiked = fxdriver.summarize(self.rows(100.0, 120.0, spike=(0, 1, 5)), 3, 30, 1)
        self.assertFalse(spiked["gates"]["dispatch_max_median"])
        # The dropped first round of a pass does not count.
        dropped = fxdriver.summarize(self.rows(100.0, 120.0, spike=(0, 1, 0)), 3, 30, 1)
        self.assertTrue(dropped["gates_ok"])


def record(stage, rnd, variant, cell, d, c, t=0.0, **extra):
    exp = {
        ph: {
            "family": cell.family,
            "block_num": 1,
            "rdma_block_num": 0,
            "warp_per_block": 1,
            "active_qps": 2,
            "load_first": False,
        }
        for ph in ("dispatch", "combine")
    }
    rec = {
        "id": f"{stage}/r{rnd}/{variant}/{cell.key}",
        "stage": stage,
        "round": str(rnd),
        "variant": variant,
        "cell_key": cell.key,
        "status": "ok",
        "verified": True,
        "expect": exp,
        "effective": json.loads(json.dumps(exp)),
        "t_start": t,
        "stats": {
            "dispatch": {"median": d},
            "combine": {"median": c},
            "total_median": d + c,
            "gates_ok": True,
        },
        "_path": f"/tmp/{stage}-{rnd}-{variant}-{cell.key}",
    }
    rec.update(extra)
    return rec


class AnalysisTest(unittest.TestCase):
    group = Group("v2", SHAPE, ("bf16", "fp8_e4m3"), tokens=(64,))
    cands = {
        "v2": {
            "dispatch": [(32, 16, 8), (64, 32, 8), (128, 64, 8)],
            "combine": [(32, 16, 4), (64, 32, 4)],
        }
    }

    def geometry_records(self, best_gain=0.05, rounds=("1", "2"), flaky=False):
        recs = []
        for rnd in rounds:
            for cell in self.group.cells():
                recs.append(record("g", rnd, "seed", cell, 100, 100))
                recs.append(record("g", rnd, "seed-end", cell, 100, 100))
                for g in self.cands["v2"]["dispatch"]:
                    d = 100 * (1 - best_gain) if g == (64, 32, 8) else 110
                    if flaky and rnd == "2" and g == (64, 32, 8):
                        d = 101
                    recs.append(
                        record(
                            "g",
                            rnd,
                            A.geometry_variant_id("v2", "dispatch", g),
                            cell,
                            d,
                            100,
                        )
                    )
                for g in self.cands["v2"]["combine"]:
                    c = 97 if g == (32, 16, 4) else 105
                    recs.append(
                        record(
                            "g",
                            rnd,
                            A.geometry_variant_id("v2", "combine", g),
                            cell,
                            100,
                            c,
                        )
                    )
        return recs

    def test_prune_and_select(self):
        idx = A.index(self.geometry_records())
        lists = A.prune(idx, [self.group], self.cands, window=0.03, min_keep=1)
        sel = json.dumps(list(self.group.cells()[0].selector("dispatch")))
        self.assertEqual(lists["v2"]["dispatch"][sel], [[64, 32, 8]])
        picks = A.select_geometry(idx, [self.group], lists)
        by_phase = {(p["phase"], p["selector"][4]): p for p in picks}
        self.assertEqual(by_phase[("dispatch", "bf16")]["geometry"], [64, 32, 8])
        self.assertEqual(by_phase[("combine", "bf16")]["geometry"], [32, 16, 4])

    def test_seed_kept_unless_faster_every_round(self):
        idx = A.index(self.geometry_records(flaky=True))
        lists = A.prune(idx, [self.group], self.cands, window=0.03, min_keep=1)
        picks = A.select_geometry(idx, [self.group], lists)
        d = [p for p in picks if p["phase"] == "dispatch"][0]
        self.assertIsNone(d["geometry"])  # 5% faster in round 1, 1% slower in round 2
        idx = A.index(self.geometry_records(best_gain=0.005))
        picks = A.select_geometry(
            idx, [self.group], A.prune(idx, [self.group], self.cands)
        )
        self.assertIsNone(
            [p for p in picks if p["phase"] == "dispatch"][0]["geometry"]
        )  # < 1%

    def test_a_geometry_that_was_wrong_once_is_never_picked(self):
        recs = self.geometry_records()
        best = A.geometry_variant_id("v2", "dispatch", (64, 32, 8))
        bad = dict(recs[0])
        bad.update(
            variant=best,
            status="wrong",
            cell=self.group.cells()[0].to_json(),
            cell_key=self.group.cells()[0].key,
            _path="/tmp/wrong",
        )
        recs.append(bad)
        wrong = A.wrong_configs(recs)
        self.assertIn(("v2", SHAPE.tag, best), wrong)
        idx = A.index(recs)
        lists = A.prune(
            idx, [self.group], self.cands, window=0.5, min_keep=1, wrong=wrong
        )
        for kept in lists["v2"]["dispatch"].values():
            self.assertNotIn([64, 32, 8], kept)
        picks = A.select_geometry(
            idx, [self.group], A.prune(idx, [self.group], self.cands), wrong=wrong
        )
        self.assertTrue(all(p.get("geometry") != [64, 32, 8] for p in picks))

    def test_verify_only_records_have_no_gates(self):
        cell = self.group.cells()[0]
        rec = record("vf", "1", "merged", cell, 1, 1)
        rec.pop("stats")
        rec["protocol"] = {"timing": False}
        self.assertTrue(A.is_valid(rec))
        rec["status"] = "wrong"
        self.assertEqual(A.wrong_cells([rec], "merged"), {cell.key})

    def test_invalid_runs_are_ignored(self):
        recs = self.geometry_records()
        for r in recs:
            if r["variant"] == A.geometry_variant_id("v2", "dispatch", (64, 32, 8)):
                r["effective"]["dispatch"]["block_num"] = 999  # ran something else
        idx = A.index(recs)
        picks = A.select_geometry(
            idx, [self.group], A.prune(idx, [self.group], self.cands)
        )
        self.assertIsNone([p for p in picks if p["phase"] == "dispatch"][0]["geometry"])

    def test_stalls(self):
        cell = self.group.cells()[0]
        recs = [
            record("g", "1", "v", cell, 100, 100),
            record("g", "2", "v", cell, 900, 100),
        ]
        self.assertEqual([r["round"] for r in A.find_stalls(recs)], ["2"])
        block = [
            record("g", "1", "seed", cell, 100, 100, t=0),
            record("g", "1", "seed-end", cell, 100, 100, t=9),
        ]
        block += [
            record(
                "g", "1", f"v{i}", cell, 100 if i not in (3, 4, 5) else 2000, 100, t=i
            )
            for i in range(1, 8)
        ]
        self.assertEqual(
            sorted(r["variant"] for r in A.find_block_stalls(block)), ["v3", "v4", "v5"]
        )
        block[1]["stats"]["dispatch"]["median"] = 400
        self.assertEqual(len(A.find_block_stalls(block)), len(block))

    def test_qp_selection(self):
        group = Group("v2", SHAPE, ("bf16",), tokens=(4,))
        cell = group.cells()[0]
        recs = []
        for rnd in ("1", "2"):
            for d, c, lf in A.qp_grid("v2", [1, 2, 4]):
                dv = 50 + (2 if d < c else 0)
                cv = 60 - (6 if c == 1 else 0) - (8 if lf else 0)
                recs.append(
                    record("q", rnd, A.qp_variant_id("v2", d, c, lf), cell, dv, cv)
                )
        idx = A.index(recs)
        picks = A.select_qp(idx, [group], [1, 2, 4], base_lf=lambda c: False)
        self.assertEqual(picks[0]["choice"]["combine"], 1)
        self.assertTrue(picks[0]["choice"]["load_first"])
        self.assertEqual(picks[0]["choice"]["dispatch"][cell.key], 1)

    def test_ab(self):
        cells = self.group.cells()
        recs = []
        for rnd in ("1", "2", "3"):
            for cell in cells:
                recs.append(record("ab", rnd, "base", cell, 100, 100))
                recs.append(
                    record("ab", rnd, "final", cell, 95 if rnd != "2" else 101, 100)
                )
        rows = A.ab_buckets(A.index(recs), [self.group], "base", "final")
        self.assertFalse(rows[0]["take"])  # not faster in round 2
        recs = [r for r in recs if not (r["variant"] == "final" and r["round"] == "2")]
        for cell in cells:
            recs.append(record("ab", "2", "final", cell, 96, 100))
        self.assertTrue(
            A.ab_buckets(A.index(recs), [self.group], "base", "final")[0]["take"]
        )


class HarnessTest(unittest.TestCase):
    def test_parse_old_sweeps(self):
        logs = next(
            (p for p in fxconfig.harness_log_dirs() if p.name == "logs_ll16"), None
        )
        if logs is None or not logs.exists():
            self.skipTest("old harness sweep logs not configured (FX_HARNESS_LOGS)")
        sweeps = fxharness.collect([logs])
        self.assertGreater(len(sweeps), 20)
        sweep, path = sweeps[("v2_ll", "256x8h7168", "dispatch", "bf16", 512)]
        self.assertEqual(sweep["tokens"], 512)
        self.assertGreaterEqual(len(sweep["values"]), 36)
        self.assertLess(min(sweep["values"].values()), 530)


class EndToEndTest(unittest.TestCase):
    def test_simulated_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            fxcampaign.CAMPAIGNS = Path(tmp)
            fxcampaign.main(
                ["init", "e2e", "--preset", "test-ep16", "--base", str(BASE)]
            )
            fxcampaign.main(["chain", "e2e", "--simulate"])
            fxcampaign.main(["land", "e2e"])
            landed = T.TableSet.load(Path(tmp) / "e2e" / "landed")
            base = T.TableSet.load(Path(tmp) / "e2e" / "tables" / "base")
            self.assertGreater(len(landed.rules["v2"]), len(base.rules["v2"]))
            t4 = Cell("v2", SHAPE, "bf16", "bf16", 4)
            rule = landed.get("v2", t4.selector("combine"))
            self.assertIsNotNone(rule)
            self.assertIn("avg_latency_us", rule)


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
```

#### `tests/fixtures/base_tables/gfx950_mi355x_InterNodeV2LL_ep16.csv`

```csv
num_experts,topk,phase,dtype,hidden_dim,zero_copy,quant_type,num_tokens,block_num,rdma_block_num,warp_per_block,active_qps,combine_load_first,avg_latency_us,bandwidth_gbps,avg_rdma_bandwidth_gbps,avg_xgmi_bandwidth_gbps,avg_ll_bandwidth_gbps,bandwidth_metric
256,6,dispatch,bf16,7168,,,64,96,64,8,,,84.22,65.27,21.9,57.11,65.27,grand_mean
256,6,dispatch,bf16,7168,,,128,96,64,8,,,105.86,100.67,35.27,91.5,100.67,grand_mean
256,6,dispatch,bf16,7168,,,256,256,128,4,,,149.84,141,49.51,128.7,141,grand_mean
256,6,dispatch,bf16,7168,,,512,96,64,8,,,232.14,196.37,63.67,164.86,196.37,grand_mean
256,6,dispatch,bf16,7168,,,1024,128,85,16,,,404.41,216.62,72.84,189.12,216.62,grand_mean
256,6,dispatch,bf16,7168,,,2048,128,85,16,,,809.39,220.64,72.81,188.54,220.64,grand_mean
256,6,dispatch,bf16,7168,,,4096,128,85,16,,,1606.81,220.72,73.39,190.01,220.72,grand_mean
256,6,combine,bf16,7168,0,none,64,96,64,8,,,86.89,63.39,21.27,55.47,63.39,grand_mean
256,6,combine,bf16,7168,0,none,128,96,64,8,,,105.38,101.35,35.49,92.11,101.35,grand_mean
256,6,combine,bf16,7168,0,none,256,96,64,8,,,157.11,134.08,47.08,122.38,134.08,grand_mean
256,6,combine,bf16,7168,0,none,512,96,64,8,,,257.83,176.86,57.33,148.48,176.86,grand_mean
256,6,combine,bf16,7168,0,none,1024,96,64,8,,,451.45,193.86,65.19,169.25,193.86,grand_mean
256,6,combine,bf16,7168,0,none,2048,96,64,8,,,826.44,216.12,71.31,184.68,216.12,grand_mean
256,6,combine,bf16,7168,0,none,4096,96,64,8,,,1571.2,225.53,75,194.15,225.53,grand_mean
256,8,dispatch,bf16,7168,,,64,96,64,8,,,86.33,82.29,21.41,70.07,82.29,grand_mean
256,8,dispatch,bf16,7168,,,128,96,64,8,,,107.53,131.21,34.58,112.89,131.21,grand_mean
256,8,dispatch,bf16,7168,,,256,96,64,8,,,158.96,177.25,46.32,151.37,177.25,grand_mean
256,8,dispatch,bf16,7168,,,512,96,64,8,,,247.14,242.83,59.49,194.27,242.83,grand_mean
256,8,dispatch,bf16,7168,,,1024,256,128,16,,,427.35,274.93,68.93,225.29,274.93,grand_mean
256,8,dispatch,bf16,7168,,,2048,256,128,16,,,857.42,279.53,69.02,224.78,279.53,grand_mean
256,8,dispatch,bf16,7168,,,4096,32,16,16,,,1742.21,270.31,67.61,220.55,270.31,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,64,96,64,8,,,63.85,55.51,14.44,47.27,55.51,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,128,96,64,8,,,77.87,90.46,23.86,77.83,90.46,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,256,96,64,8,,,106.46,132.66,34.67,113.3,132.66,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,512,128,85,8,,,159.22,191.23,46.85,152.99,191.23,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,1024,256,128,16,,,280.23,209.77,52.6,171.9,209.77,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,2048,256,128,16,,,471.7,253.11,62.49,203.54,253.11,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,4096,256,128,16,,,955.64,246.9,61.75,201.45,246.9,grand_mean
256,8,combine,bf16,7168,0,none,64,96,64,8,,,89.87,78.96,20.53,67.24,78.96,grand_mean
256,8,combine,bf16,7168,0,none,128,96,64,8,,,110.02,127.32,33.58,109.54,127.32,grand_mean
256,8,combine,bf16,7168,0,none,256,96,64,8,,,166.52,169.03,44.17,144.35,169.03,grand_mean
256,8,combine,bf16,7168,0,none,512,96,64,8,,,270.31,222.05,54.4,177.65,222.05,grand_mean
256,8,combine,bf16,7168,0,none,1024,96,64,8,,,467.63,251.05,62.95,205.73,251.05,grand_mean
256,8,combine,bf16,7168,0,none,2048,96,64,8,,,862.26,277.38,68.48,223.05,277.38,grand_mean
256,8,combine,bf16,7168,0,none,4096,96,64,8,,,1633.65,289.01,72.29,235.81,289.01,grand_mean
384,6,dispatch,bf16,5120,,,64,96,64,8,,,71.98,55.71,18.37,47.59,55.71,grand_mean
384,6,dispatch,bf16,5120,,,128,96,64,8,,,88.67,89.68,29.99,77.77,89.68,grand_mean
384,6,dispatch,bf16,5120,,,256,96,64,8,,,120.84,133.21,44.02,113.87,133.21,grand_mean
384,6,dispatch,bf16,5120,,,512,96,64,8,,,182.35,171.74,57.77,149.21,171.74,grand_mean
384,6,dispatch,bf16,5120,,,1024,96,64,8,,,337.17,186.52,62.63,161.75,186.52,grand_mean
384,6,dispatch,bf16,5120,,,2048,256,170,16,,,626.04,200.93,67.13,173.36,200.93,grand_mean
384,6,dispatch,bf16,5120,,,4096,256,170,16,,,1220.63,205.94,68.98,178.26,205.94,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,64,96,64,8,,,57.02,35.03,11.56,29.92,35.03,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,128,96,64,8,,,68.4,57.87,19.36,50.19,57.87,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,256,96,64,8,,,89.92,89.28,29.5,76.32,89.28,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,512,96,64,8,,,122.05,128.58,43.26,111.71,128.58,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,1024,96,64,8,,,212.02,149.44,50.17,129.59,149.44,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,2048,256,128,16,,,376.3,167.46,55.95,144.48,167.46,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,4096,256,128,16,,,769.15,163.76,54.86,141.75,163.76,grand_mean
384,6,combine,bf16,5120,0,none,64,96,64,8,,,71.26,56.06,18.52,47.88,56.06,grand_mean
384,6,combine,bf16,5120,0,none,128,96,64,8,,,85.3,94.19,31.5,81.68,94.19,grand_mean
384,6,combine,bf16,5120,0,none,256,96,64,8,,,125.09,128.12,42.35,109.52,128.12,grand_mean
384,6,combine,bf16,5120,0,none,512,64,42,8,,,193.36,162.37,54.61,141.07,162.37,grand_mean
384,6,combine,bf16,5120,0,none,1024,64,42,8,,,333.02,189.1,63.49,163.98,189.1,grand_mean
384,6,combine,bf16,5120,0,none,2048,96,64,8,,,612.26,205.42,68.63,177.23,205.42,grand_mean
384,6,combine,bf16,5120,0,none,4096,96,64,8,,,1146.79,219.29,73.46,189.81,219.29,grand_mean
384,6,dispatch,bf16,7168,,,64,96,64,8,,,84.56,66.09,21.81,56.45,66.09,grand_mean
384,6,dispatch,bf16,7168,,,128,96,64,8,,,105.74,104.65,35,90.75,104.65,grand_mean
384,6,dispatch,bf16,7168,,,256,96,64,8,,,152.48,148.24,48.98,126.72,148.24,grand_mean
384,6,dispatch,bf16,7168,,,512,96,64,8,,,234.04,187.56,63.1,162.95,187.56,grand_mean
384,6,dispatch,bf16,7168,,,1024,128,85,16,,,404.4,220.18,72.86,188.11,220.18,grand_mean
384,6,dispatch,bf16,7168,,,2048,128,85,16,,,806.52,222.07,73.01,188.49,222.07,grand_mean
384,6,dispatch,bf16,7168,,,4096,128,85,16,,,1603.2,222.35,73.5,189.76,222.35,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,64,96,64,8,,,63.37,44.26,14.6,37.81,44.26,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,128,96,64,8,,,76.34,72.55,24.26,62.91,72.55,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,256,96,64,8,,,104.04,108.39,35.82,92.65,108.39,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,512,96,64,8,,,149.41,147.23,49.54,127.91,147.23,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,1024,256,170,8,,,252.22,176.49,58.4,150.78,176.49,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,2048,256,128,16,,,446.38,200.45,65.91,170.14,200.45,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,4096,256,128,16,,,911.77,195.34,64.57,166.71,195.34,grand_mean
384,6,combine,bf16,7168,0,none,64,96,64,8,,,86.37,64.74,21.38,55.3,64.74,grand_mean
384,6,combine,bf16,7168,0,none,128,96,64,8,,,103.74,107.15,35.83,92.92,107.15,grand_mean
384,6,combine,bf16,7168,0,none,256,96,64,8,,,157.21,142.62,47.14,121.91,142.62,grand_mean
384,6,combine,bf16,7168,0,none,512,96,64,8,,,257.79,170.14,57.24,147.82,170.14,grand_mean
384,6,combine,bf16,7168,0,none,1024,96,64,8,,,449.97,197.56,65.38,168.78,197.56,grand_mean
384,6,combine,bf16,7168,0,none,2048,96,64,8,,,827.31,216.7,71.25,183.93,216.7,grand_mean
384,6,combine,bf16,7168,0,none,4096,96,64,8,,,1563.21,227.86,75.31,194.47,227.86,grand_mean
```

#### `tests/fixtures/base_tables/gfx950_mi355x_InterNodeV2_ep16.csv`

```csv
num_experts,topk,phase,dtype,hidden_dim,zero_copy,quant_type,num_tokens,block_num,rdma_block_num,warp_per_block,active_qps,combine_load_first,avg_latency_us,bandwidth_gbps,avg_rdma_bandwidth_gbps,avg_xgmi_bandwidth_gbps,avg_ll_bandwidth_gbps,bandwidth_metric
256,6,dispatch,bf16,7168,,,64,16,8,8,,,110.44,16.48,16.48,43.62,49.85,grand_mean
256,6,dispatch,bf16,7168,,,128,32,16,8,,,119.65,30.31,30.31,79.96,87.98,grand_mean
256,6,dispatch,bf16,7168,,,256,96,64,8,,,160.05,45.52,45.52,120.01,131.48,grand_mean
256,6,dispatch,bf16,7168,,,512,96,64,8,,,233.47,62.47,62.47,164.09,195.46,grand_mean
256,6,dispatch,bf16,7168,,,1024,256,128,8,,,433.46,66.84,66.84,176.03,201.62,grand_mean
256,6,dispatch,bf16,7168,,,2048,256,128,8,,,810.53,71.52,71.52,187.86,219.85,grand_mean
256,6,dispatch,bf16,7168,,,4096,256,128,8,,,1599.36,72.55,72.55,190.62,221.42,grand_mean
256,6,combine,bf16,7168,0,none,64,16,10,8,,1,104.47,17.48,17.48,46.26,52.87,grand_mean
256,6,combine,bf16,7168,0,none,128,96,64,8,,1,110.99,32.73,32.73,86.34,95,grand_mean
256,6,combine,bf16,7168,0,none,256,64,32,8,,1,166.9,43.88,43.88,115.7,126.75,grand_mean
256,6,combine,bf16,7168,0,none,512,64,32,8,,,260.43,55.81,55.81,146.63,174.67,grand_mean
256,6,combine,bf16,7168,0,none,1024,64,32,8,,,440.84,65.72,65.72,173.07,198.23,grand_mean
256,6,combine,bf16,7168,0,none,2048,64,32,8,,,811.21,71.46,71.46,187.71,219.68,grand_mean
256,6,combine,bf16,7168,0,none,4096,64,32,8,,,1562.45,74.31,74.31,195.25,226.81,grand_mean
256,8,dispatch,bf16,7168,,,64,64,32,8,,,118.28,15.59,15.59,51.22,60.15,grand_mean
256,8,dispatch,bf16,7168,,,128,32,16,8,,,129.78,28.17,28.17,92.36,107.35,grand_mean
256,8,dispatch,bf16,7168,,,256,96,64,8,,,169.87,43.91,43.91,144.18,168.82,grand_mean
256,8,dispatch,bf16,7168,,,512,96,64,8,,,243.79,60.33,60.33,197.74,247.16,grand_mean
256,8,dispatch,bf16,7168,,,1024,256,128,8,,,437,67.14,67.14,220.19,268.7,grand_mean
256,8,dispatch,bf16,7168,,,2048,256,128,8,,,808.08,72.52,72.52,236.92,294.63,grand_mean
256,8,dispatch,bf16,7168,,,4096,256,128,8,,,1571.11,74.58,74.58,244.16,299.25,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,64,64,32,8,,,90.2,10.49,10.49,34.43,40.43,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,128,32,16,8,,,97.07,18.91,18.91,61.98,72.04,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,256,96,64,8,,,120.01,31.34,31.34,102.92,120.51,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,512,96,64,8,,,160.35,45.72,45.72,149.87,187.33,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,1024,256,128,8,,,252.58,58.05,58.05,190.38,232.33,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,2048,256,128,8,,,475.7,61.67,61.67,201.46,250.53,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,4096,256,128,8,,,907.79,64.61,64.61,211.52,259.25,grand_mean
256,8,combine,bf16,7168,0,none,64,16,10,8,,1,108.72,17.02,17.02,55.9,65.65,grand_mean
256,8,combine,bf16,7168,0,none,128,64,32,8,,1,112.56,32.51,32.51,106.62,123.92,grand_mean
256,8,combine,bf16,7168,0,none,256,64,42,8,,1,170.53,43.54,43.54,142.94,167.37,grand_mean
256,8,combine,bf16,7168,0,none,512,64,32,8,,,275.58,53.23,53.23,174.46,218.06,grand_mean
256,8,combine,bf16,7168,0,none,1024,64,32,8,,,457.71,64.04,64.04,210.01,256.28,grand_mean
256,8,combine,bf16,7168,0,none,2048,64,32,8,,,827.69,70.78,70.78,231.23,287.55,grand_mean
256,8,combine,bf16,7168,0,none,4096,64,32,8,,,1592.91,73.61,73.61,241,295.37,grand_mean
384,6,dispatch,bf16,5120,,,64,16,8,8,,,92.41,13.99,13.99,36.85,43.14,grand_mean
384,6,dispatch,bf16,5120,,,128,32,16,8,,,101.22,25.78,25.78,68.02,78.44,grand_mean
384,6,dispatch,bf16,5120,,,256,96,64,8,,,131.15,40.7,40.7,106.85,124.99,grand_mean
384,6,dispatch,bf16,5120,,,512,96,64,8,,,186.18,55.82,55.82,146.33,168.43,grand_mean
384,6,dispatch,bf16,5120,,,1024,128,85,16,,,324.21,63.91,63.91,167.61,193.28,grand_mean
384,6,dispatch,bf16,5120,,,2048,256,128,8,,,627.25,65.97,65.97,173,200.51,grand_mean
384,6,dispatch,bf16,5120,,,4096,256,128,8,,,1224.79,67.72,67.72,177.58,205.15,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,64,32,16,8,,,68.52,9.56,9.56,25.21,29.51,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,128,32,16,8,,,73.87,17.75,17.75,46.85,54.02,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,256,96,64,8,,,98.79,26.76,26.76,70.23,82.16,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,512,96,64,8,,,125.37,41.28,41.28,108.25,124.6,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,1024,256,170,8,,,211.82,48.79,48.79,127.94,147.54,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,2048,96,64,8,,,422.3,49.33,49.33,129.37,149.94,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,4096,96,64,8,,,838.29,50.13,50.13,131.43,151.84,grand_mean
384,6,combine,bf16,5120,0,none,64,96,64,8,,1,82.88,15.61,15.61,41.18,48.21,grand_mean
384,6,combine,bf16,5120,0,none,128,96,64,8,,1,87.39,29.9,29.9,78.86,90.94,grand_mean
384,6,combine,bf16,5120,0,none,256,32,16,8,,1,122.92,43.69,43.69,114.57,134.03,grand_mean
384,6,combine,bf16,5120,0,none,512,64,32,8,,,195.9,52.93,52.93,138.78,159.74,grand_mean
384,6,combine,bf16,5120,0,none,1024,64,32,8,,,326.3,63.5,63.5,166.53,192.03,grand_mean
384,6,combine,bf16,5120,0,none,2048,96,64,8,,,611.7,67.62,67.62,177.33,205.53,grand_mean
384,6,combine,bf16,5120,0,none,4096,96,64,8,,,1162.86,71.43,71.43,187.32,216.4,grand_mean
384,6,dispatch,bf16,7168,,,64,16,8,8,,,107.96,16.73,16.73,44.09,51.61,grand_mean
384,6,dispatch,bf16,7168,,,128,32,16,8,,,120.28,30.42,30.42,80.26,92.55,grand_mean
384,6,dispatch,bf16,7168,,,256,96,64,8,,,159.86,46.02,46.02,120.79,141.3,grand_mean
384,6,dispatch,bf16,7168,,,512,96,64,8,,,234.58,62.01,62.01,162.57,187.12,grand_mean
384,6,dispatch,bf16,7168,,,1024,128,85,16,,,416.44,69.69,69.69,182.72,213.88,grand_mean
384,6,dispatch,bf16,7168,,,2048,96,64,8,,,842.98,69,69,180.75,212.95,grand_mean
384,6,dispatch,bf16,7168,,,4096,128,85,16,,,1616.79,71.78,71.78,188.05,220.34,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,64,16,8,8,,,83.61,10.91,10.91,28.75,33.65,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,128,32,16,8,,,89.63,20.52,20.52,54.15,62.44,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,256,96,64,8,,,116.39,32.04,32.04,84.1,98.38,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,512,96,64,8,,,156.13,46.56,46.56,122.07,140.5,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,1024,256,128,8,,,251.69,57.47,57.47,150.68,176.37,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,2048,256,128,8,,,491.45,59.04,59.04,154.66,182.21,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,4096,256,128,8,,,953.64,60.99,60.99,159.78,187.22,grand_mean
384,6,combine,bf16,7168,0,none,64,96,64,8,,1,101.9,17.73,17.73,46.75,54.73,grand_mean
384,6,combine,bf16,7168,0,none,128,96,64,8,,1,108,33.89,33.89,89.39,103.08,grand_mean
384,6,combine,bf16,7168,0,none,256,32,21,8,,1,155.03,48,48,125.86,147.24,grand_mean
384,6,combine,bf16,7168,0,none,512,64,42,8,,,261.26,55.71,55.71,146.13,168.2,grand_mean
384,6,combine,bf16,7168,0,none,1024,64,42,8,,,440.58,65.79,65.79,172.52,201.94,grand_mean
384,6,combine,bf16,7168,0,none,2048,96,64,8,,,845.74,68.84,68.84,180.35,212.48,grand_mean
384,6,combine,bf16,7168,0,none,4096,96,64,8,,,1581.76,73.41,73.41,192.35,225.39,grand_mean
```

#### `tests/fixtures/base_tables/gfx950_mi355x_IntraNodeV2_ep8.csv`

```csv
num_experts,topk,phase,dtype,hidden_dim,zero_copy,quant_type,num_tokens,block_num,rdma_block_num,warp_per_block,active_qps,combine_load_first,avg_latency_us,bandwidth_gbps,avg_rdma_bandwidth_gbps,avg_xgmi_bandwidth_gbps,avg_ll_bandwidth_gbps,bandwidth_metric
256,6,dispatch,bf16,7168,,,64,64,0,8,,,29.77,137.29,,137.29,,grand_mean
256,6,dispatch,bf16,7168,,,128,64,0,4,,,43.09,189.1,,189.1,,grand_mean
256,6,dispatch,bf16,7168,,,256,128,0,4,,,64.37,254.61,,254.61,,grand_mean
256,6,dispatch,bf16,7168,,,512,128,0,4,,,109.77,297.83,,297.83,,grand_mean
256,6,dispatch,bf16,7168,,,1024,64,0,8,,,214.48,305.43,,305.43,,grand_mean
256,6,dispatch,bf16,7168,,,2048,128,0,4,,,384.24,339.63,,339.63,,grand_mean
256,6,dispatch,bf16,7168,,,4096,256,0,4,,,720.92,361.48,,361.48,,grand_mean
256,6,combine,bf16,7168,0,none,64,64,0,8,,,27.58,148.2,,148.2,,grand_mean
256,6,combine,bf16,7168,0,none,128,64,0,8,,,39.64,205.77,,205.77,,grand_mean
256,6,combine,bf16,7168,0,none,256,64,0,8,,,61.89,264.77,,264.77,,grand_mean
256,6,combine,bf16,7168,0,none,512,64,0,8,,,105.3,310.42,,310.42,,grand_mean
256,6,combine,bf16,7168,0,none,1024,64,0,8,,,197.1,332.31,,332.31,,grand_mean
256,6,combine,bf16,7168,0,none,2048,64,0,8,,,370.45,352.25,,352.25,,grand_mean
256,6,combine,bf16,7168,0,none,4096,128,0,4,,,726.46,358.68,,358.68,,grand_mean
256,8,dispatch,bf16,7168,,,64,64,0,8,,,31.8,152.06,,152.06,,grand_mean
256,8,dispatch,bf16,7168,,,128,64,0,8,,,46.98,206.15,,206.15,,grand_mean
256,8,dispatch,bf16,7168,,,256,64,0,8,,,75.77,257.99,,257.99,,grand_mean
256,8,dispatch,bf16,7168,,,512,128,0,8,,,128.4,302.68,,302.68,,grand_mean
256,8,dispatch,bf16,7168,,,1024,128,0,8,,,234.96,332.24,,332.24,,grand_mean
256,8,dispatch,bf16,7168,,,2048,128,0,8,,,445.18,349.19,,349.19,,grand_mean
256,8,dispatch,bf16,7168,,,4096,128,0,8,,,866.29,358.99,,358.99,,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,64,64,0,8,,,24.57,98.49,,98.49,,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,128,64,0,8,,,33.15,146.16,,146.16,,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,256,64,0,8,,,49.55,197.32,,197.32,,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,512,256,0,4,,,73.78,263.44,,263.44,,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,1024,256,0,4,,,130.61,298.94,,298.94,,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,2048,256,0,4,,,240.94,322.68,,322.68,,grand_mean
256,8,dispatch,fp8_e4m3,7168,,,4096,256,0,4,,,455.29,341.64,,341.64,,grand_mean
256,8,combine,bf16,7168,0,none,64,64,0,4,,,28.05,172.57,,172.57,,grand_mean
256,8,combine,bf16,7168,0,none,128,64,0,8,,,43.2,224.38,,224.38,,grand_mean
256,8,combine,bf16,7168,0,none,256,64,0,8,,,70.16,278.64,,278.64,,grand_mean
256,8,combine,bf16,7168,0,none,512,64,0,8,,,121.62,319.55,,319.55,,grand_mean
256,8,combine,bf16,7168,0,none,1024,64,0,8,,,228.51,341.58,,341.58,,grand_mean
256,8,combine,bf16,7168,0,none,2048,64,0,8,,,434.56,357.69,,357.69,,grand_mean
256,8,combine,bf16,7168,0,none,4096,128,0,4,,,844.86,368.08,,368.08,,grand_mean
384,6,dispatch,bf16,5120,,,64,64,0,16,,,27.93,104.64,,104.64,,grand_mean
384,6,dispatch,bf16,5120,,,128,64,0,16,,,38.27,153.29,,153.29,,grand_mean
384,6,dispatch,bf16,5120,,,256,128,0,4,,,53.05,218.76,,218.76,,grand_mean
384,6,dispatch,bf16,5120,,,512,256,0,4,,,85.89,270.7,,270.7,,grand_mean
384,6,dispatch,bf16,5120,,,1024,256,0,4,,,149.45,311.66,,311.66,,grand_mean
384,6,dispatch,bf16,5120,,,2048,256,0,4,,,279.69,332.05,,332.05,,grand_mean
384,6,dispatch,bf16,5120,,,4096,256,0,4,,,539.44,345.06,,345.06,,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,64,64,0,16,,,22.93,63.73,,63.73,,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,128,64,0,16,,,29.11,100,,100,,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,256,256,0,4,,,39.29,147.42,,147.42,,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,512,256,0,4,,,54.49,213.39,,213.39,,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,1024,256,0,4,,,89.38,260.68,,260.68,,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,2048,64,0,16,,,162.62,285.6,,285.6,,grand_mean
384,6,dispatch,fp8_e4m3,5120,,,4096,64,0,16,,,296.36,314.07,,314.07,,grand_mean
384,6,combine,bf16,5120,0,none,64,64,0,8,,,24.7,120.46,,120.46,,grand_mean
384,6,combine,bf16,5120,0,none,128,64,0,8,,,33.41,174.28,,174.28,,grand_mean
384,6,combine,bf16,5120,0,none,256,64,0,8,,,49.23,235.27,,235.27,,grand_mean
384,6,combine,bf16,5120,0,none,512,64,0,8,,,79.88,291.17,,291.17,,grand_mean
384,6,combine,bf16,5120,0,none,1024,64,0,8,,,145.35,320.46,,320.46,,grand_mean
384,6,combine,bf16,5120,0,none,2048,64,0,8,,,270.76,343.01,,343.01,,grand_mean
384,6,combine,bf16,5120,0,none,4096,64,0,8,,,523.77,355.38,,355.38,,grand_mean
384,6,dispatch,bf16,7168,,,64,64,0,8,,,30.56,134.95,,134.95,,grand_mean
384,6,dispatch,bf16,7168,,,128,64,0,8,,,45.01,180.94,,180.94,,grand_mean
384,6,dispatch,bf16,7168,,,256,64,0,8,,,66.89,242.36,,242.36,,grand_mean
384,6,dispatch,bf16,7168,,,512,256,0,4,,,110.38,294.93,,294.93,,grand_mean
384,6,dispatch,bf16,7168,,,1024,256,0,4,,,195.95,332.3,,332.3,,grand_mean
384,6,dispatch,bf16,7168,,,2048,256,0,4,,,372.68,348,,348,,grand_mean
384,6,dispatch,bf16,7168,,,4096,256,0,4,,,717.22,362.67,,362.67,,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,64,64,0,8,,,23.09,88.74,,88.74,,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,128,64,0,8,,,33.77,120.66,,120.66,,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,256,64,0,8,,,45.87,176.83,,176.83,,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,512,64,0,8,,,70.51,230.93,,230.93,,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,1024,64,0,8,,,119.73,271.99,,271.99,,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,2048,64,0,8,,,220.46,294.2,,294.2,,grand_mean
384,6,dispatch,fp8_e4m3,7168,,,4096,64,0,8,,,404.97,321.17,,321.17,,grand_mean
384,6,combine,bf16,7168,0,none,64,64,0,8,,,27.29,150.03,,150.03,,grand_mean
384,6,combine,bf16,7168,0,none,128,64,0,8,,,39.9,204.19,,204.19,,grand_mean
384,6,combine,bf16,7168,0,none,256,64,0,8,,,61.9,261.89,,261.89,,grand_mean
384,6,combine,bf16,7168,0,none,512,64,0,8,,,105.26,309.29,,309.29,,grand_mean
384,6,combine,bf16,7168,0,none,1024,64,0,8,,,194.44,334.89,,334.89,,grand_mean
384,6,combine,bf16,7168,0,none,2048,64,0,8,,,372.65,348.06,,348.06,,grand_mean
384,6,combine,bf16,7168,0,none,4096,128,0,4,,,721.42,360.54,,360.54,,grand_mean
```
