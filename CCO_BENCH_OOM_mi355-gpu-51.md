# CCO ops-v2 bench 在 mi355-gpu-51 上报 OOM（ROCm 10.1）：交接说明

给在 mi355-gpu-51 上接手的 agent。目标：查清 PR #871 的 CCO CI 为什么在这台机器上
报 `hipMemCreate failed: 2 (out of memory)`，并给出修复方向。下面的结论和数据都来自
pit2-p03-g17（另一台 MI355X），在那台机器上复现不了。

## 1. 背景

- PR [ROCm/mori#871](https://github.com/ROCm/mori/pull/871)，分支 `dev/migrate_rocm10`，
  head `4a4df301`。它把 CI 从 ROCm 7.14 迁到 ROCm 10.1，镜像是
  `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.12.0`。
- 这是 pull_request 事件，CI checkout 的是 merge commit `cc973cdd`
  （`4a4df301` 合进 main 的 `f3cebc3a`），不是 PR head。
- 失败的 job 是 `ci_cco.yml` 的 "CCO unit test (MI355X_AINIC)"，runner 是
  `mi355-gpu-51`。这台 runner 用 rootless podman（workflow 里 `CT: podman`），
  容器用 `docker/ci_run.sh` 起，默认带 `--privileged`。
- CI 镜像 tag `rocm/mori:ci-cco` 是共用的：main 的 7.14 构建和 PR 的 10.1 构建都写它。

## 2. 现象

- 失败的步骤是 "Run ops-v2 dispatch/combine bench (EP8 smoke, both backends)"：

  ```
  cd tests/python/ops/dispatch_combine_v2 && \
  BACKENDS=flydsl,hip COMB_MODE=pull MODES=graph SWEEP=128,512,4096,8192 \
  ITERS=20 WARMUP=10 torchrun --standalone --nproc_per_node=8 bench_ep.py
  ```

- torchrun 启动约 2.5 秒后，rank 2 在第一次分配上失败：
  `[shmem] [error] ccoMemAlloc: hipMemCreate failed: 2 (out of memory)`。调用链是
  `bench_ep.py:308 build('flydsl')`、`flydsl_backend.py:162 SymmArena(comm, regions)`、
  `symm_arena.py:58 comm.alloc_mem(self._total)`、`communicator.py:429 alloc_mem`。
  这块 arena 是 1.76 GB，GPU 有 288 GB。截图里只看到 rank 2 的 traceback，其他 rank
  是否也失败不清楚。
- 同一个 job 里前面的 ops-v2 步骤（test、HIP backend、scales、parity、AsyncLL、
  routing snapshot）都通过，它们的 arena 都小于 1 GB。
- 失败记录（时间都是 UTC）：

  | run | 代码 | 镜像 | bench 步骤 | 结果 |
  |---|---|---|---|---|
  | [41017647570](https://github.com/ROCm/mori/actions/runs/41017647570/job/123937501153) | PR `4806fda7` | 10.1 | 2026-10-30 07:46:45 至 07:46:49 | 失败 |
  | [41023155247](https://github.com/ROCm/mori/actions/runs/41023155247/job/123954966536) | PR `4a4df301` | 10.1 | 2026-10-30 09:47:04 至 09:47:08 | 失败 |
  | main 和其他分支，10-30 07:56 之后 | 各自的 | 7.14 | 跑到这一步 26 次，都在 mi355-gpu-51 上 | 26 次全部通过（另有 6 次被取消或没跑到这一步） |

  PR 失败之后紧接着的 main（10-30 08:51，7.14）也通过了，所以这台机器没有一直
  处在坏状态。问题只在 10.1 镜像跑在 mi355-gpu-51 上时出现。

## 3. 在 pit2-p03-g17 上已经确认的（全部正常）

- 裸 HIP VMM，属性和 `ccoMemAlloc` 一样（uncached、POSIX fd、gpuDirectRDMACapable）。
  每次 32 GB 反复 alloc/free：`hipMemRelease` 之后显存立即回来；子进程退出后，
  `waitpid` 返回时就全部回收。10.1 和 7.14 表现一样。
- 走 MORI：2 rank，每次 16 GB，`alloc_mem`、`register_window`（P2P 导出和导入 fd）、
  关闭、free，跑 10 轮。最慢 8 ms 回来，通信器销毁后回到起点。
- 原样跑 CI 的 bench：PR head `4a4df301` 和 CI 实际测的 merge commit `cc973cdd` 都通过，
  两个后端 8/8 个点都验证通过。每卡峰值约 12 GB，每个 rank 两块 1.76 GB 的 arena。
- CI 的 8 个 ops-v2 步骤连着跑也都通过。每一步开始时显存都回到空闲水平
  （0.3 到 0.5 GB），步骤之间没有残留进程。AsyncLL sdma 第一次失败在
  `hsaKmtCreateQueueExt`（SDMA 队列不够，同机的 CI job 在用），单独重跑就过了。
- 用下面附录 A 的 `LD_PRELOAD` 拦截 KFD ioctl：
  - 10.1 和 7.14 发给驱动的 VRAM 分配请求完全一样：uncached 是 `flags=0xb5000001`，
    pinned 是 `0xb0000001`，没有 contiguous 位（bit 23）。
  - bench 进程里没有任何 SVM 注册（`AMDKFD_IOC_SVM`），加 `HSA_XNACK=1` 也没有。
  - 每个 rank 会有一条 `KFD ioctl nr=0x8 FAILED errno=28`，这是 CREATE_EVENT
    的事件槽用完了，runtime 会回退，可以忽略。
- 本机环境：kernel `6.8.0-60-generic`，amdgpu-dkms `1:6.16.13.30300100-2303411.22.04`
  （带下面说的 93aa919 修复），`amdgpu.noretry=-1`，docker，非 privileged。

## 4. 怀疑方向

**H1：amdgpu 的 SVM 重叠检查缺陷（`docker/Dockerfile.dev` 里有记录）。** 不带 VA 的
VRAM 分配（VMM 的 `hipMemCreate` 就是这种）在 `kfd_ioctl_alloc_memory_of_gpu` 里
会把 `[0, size)` 当成 SVM 重叠检查范围。只要有 SVM 注册的区间低于 `size`，内核就
返回 EADDRINUSE，HIP 报成 out of memory。镜像里的 python3.12 不是 PIE，brk 堆在
0.9 到 1 GB 附近，所以只有大于约 1 GB 的分配会撞上，这和"小 arena 通过、1.76 GB
的 bench 失败"对得上。上游修复是 93aa919（"drm/amdkfd: fix vram allocation failure
for a special case"），在 6.18 和 stable 6.17.8、6.12.58、6.6.117、6.1.159 里。
DKMS 模块会覆盖发行版内核自带的 amdgpu，所以要看 amdgpu-dkms 的版本。
疑点：pit2 上 10.1 的进程根本没有 SVM 注册，所以如果是 H1，51 上一定有别的东西在
注册 SVM。

**H2：GPU 2 当时真的没有显存了。** 可能是同一个 job 里前面的步骤在 51 上留下了显存
（比如 RDMA/dma-buf 引用在进程退出后没放），也可能是机器上别的进程。job 结束时
容器被删，下一个 job 就恢复正常，这也能解释 main 紧接着能过。

两者用 errno 就能分开：H1 是 `EADDRINUSE`（98），H2 是 `ENOMEM`（12）。

## 5. 在 51 上要做的

### 5.0 注意

这是共享的 CI runner。动 GPU 前先确认没有 CI job 在跑：
`ps aux | grep '[R]unner.Worker'`。不要覆盖 `rocm/mori:ci-cco` 这个 tag，也不要用
`mori_cco_ci_*` 的容器名，下面用 `localhost/mori-oom:*` 和 `mori_oom_*`。不要在 CI 的
workspace（`/shared/amdgpu/home/jiahao_zhou_qle/actions-runner/_work/mori/mori`）里
操作，另外 clone 一份。

### 5.1 不用 GPU 的检查（先做，可能直接定案）

```bash
uname -r
cat /sys/module/amdgpu/version
dpkg -l | grep -E 'amdgpu-dkms ' || rpm -qa | grep amdgpu-dkms
cat /sys/module/amdgpu/parameters/noretry
# 93aa919 在不在。在的话这一行能找到：
grep -n "Skip a special case that allocates VRAM without VA" /usr/src/amdgpu-*/amd/amdkfd/kfd_chardev.c
# 两次失败时内核有没有报 SVM 冲突（H1 的直接证据）。时间带了 UTC 后缀，按 UTC 解释；
# 需要 persistent journal，如果机器之后重启过又没有，就只能看下面的 dmesg：
journalctl -k --since "2026-10-30 07:40 UTC" --until "2026-10-30 07:50 UTC" | grep -E "already allocated by SVM|amdgpu|kfd"
journalctl -k --since "2026-10-30 09:40 UTC" --until "2026-10-30 09:50 UTC" | grep -E "already allocated by SVM|amdgpu|kfd"
# 没有 journal 的话：
sudo dmesg -T | grep -E "already allocated by SVM" | tail
podman info --format '{{.Host.Security.Rootless}}'; ulimit -l
```

如果在那两个时间点看到 `Address: 0x0 already allocated by SVM`，就是 H1。

### 5.2 用 10.1 镜像原样跑 bench，并挂 spy

```bash
git clone https://github.com/ROCm/mori.git mori-oom && cd mori-oom
git fetch origin pull/871/head:pr871 && git checkout pr871   # 4a4df301
podman build --network=host \
  --build-arg BASE_IMAGE=rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.12.0 \
  --build-arg GPU_ARCHS="gfx950 gfx1250" --build-arg WITH_FLYDSL=1 \
  -t localhost/mori-oom:rocm101 -f docker/Dockerfile.dev .
mkdir -p /tmp/oom   # 把附录 A 存成 /tmp/oom/kfd_svm_spy.c，附录 B 存成 /tmp/oom/replay_ops_v2.sh
CONTAINER_RUNTIME=podman ./docker/ci_run.sh --name mori_oom_101 \
  -e MORI_RDMA_DEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
  -e MORI_RDMA_SL=3 -e MORI_RDMA_TC=104 \
  -v $PWD:$PWD -v /tmp/oom:/w localhost/mori-oom:rocm101 sleep infinity
podman exec mori_oom_101 git config --global --add safe.directory $PWD
podman exec mori_oom_101 bash -c "cd $PWD && BUILD_TESTS=ON BUILD_EXAMPLES=ON BUILD_BENCHMARK=ON \
  MORI_WITH_MPI=ON BUILD_UMBP=OFF BUILD_CCO_SDMA=ON pip install ."
podman exec mori_oom_101 gcc -shared -fPIC -O2 -o /w/kfd_svm_spy.so /w/kfd_svm_spy.c -ldl

amd-smi metric -m      # 先记一下每张卡当前用了多少
podman exec -e LD_PRELOAD=/w/kfd_svm_spy.so -e SPY_BT=5 mori_oom_101 bash -c \
  "cd $PWD/tests/python/ops/dispatch_combine_v2 && BACKENDS=flydsl,hip COMB_MODE=pull MODES=graph \
   SWEEP=128,512,4096,8192 ITERS=20 WARMUP=10 timeout 900 torchrun --standalone --nproc_per_node=8 bench_ep.py" \
  > /tmp/oom/bench_spy.log 2>&1; echo rc=$?
grep -E '^\[spy' /tmp/oom/bench_spy.log | sed -E 's/\[spy [0-9]+\]/[spy]/' | sort | uniq -c
```

pit2 上正常的输出只有这两种行：16 条
`VRAM alloc va=0 size=1.76 GB flags=0xb5000001 ok, no SVM range below it`，和 8 条
`KFD ioctl nr=0x8 FAILED errno=28`（可以忽略）。

### 5.3 spy 输出怎么看

| 失败 rank 上的 spy 输出 | 含义 | 下一步 |
|---|---|---|
| `VRAM alloc ... FAILED errno=98 (Address already in use)`，并且列出了一个低于 size 的 SVM 区间 | H1 | 升级 51 的 amdgpu-dkms 到带 93aa919 的版本。同时看 `SPY_BT` 打出的 backtrace，找出是谁注册的 SVM |
| `SVM op=0 start=...` 行，并且有 `WOULD FAIL on an amdgpu without 93aa919` | 有 SVM 注册（pit2 上没有），分配本身成功了 | 看 backtrace 和 7.14 的对比（5.5） |
| `VRAM alloc ... FAILED errno=12 (Cannot allocate memory)` | H2，显存真的不够 | `amd-smi process`、`amd-smi metric -m`、`podman ps -a`，找是谁占着 GPU 2 |
| `KFD reports only X GB available` | runtime 查到可用显存不够，在进内核前就拒绝了 | 同上 |
| 失败的那次分配没有任何 spy 输出 | 在 HIP/ROCr 里就被拒了，没到内核 | 加 `AMD_LOG_LEVEL=4` 重跑，看 HIP 那边的原因 |

### 5.4 如果单独跑 bench 能过

CI 里 bench 前面还有很多步骤（CCO 单测、SDMA、benchmark、examples、allocator、
GEMM+AR、ops-v2）。按 `ci_cco.yml` 里 cco-unit-test 的顺序在同一个容器里逐步跑，
同时在 host 上开附录 C 的显存采样，看哪一步之后 GPU 2 的显存不回到空闲水平。
附录 B 只覆盖最后 8 个 ops-v2 步骤，可以先跑它；不行再把前面的步骤按 workflow
补上。

```bash
podman exec mori_oom_101 bash /w/replay_ops_v2.sh $PWD /w/replay_logs > /tmp/oom/replay_marks.log 2>&1
```

### 5.5 对照：7.14 镜像

用 main 的 base 再建一个镜像，同样挂 spy 跑 bench，比较两边的 SVM 注册：

```bash
podman build --network=host \
  --build-arg BASE_IMAGE=rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0 \
  --build-arg GPU_ARCHS="gfx950 gfx1250" --build-arg WITH_FLYDSL=1 \
  -t localhost/mori-oom:rocm714 -f docker/Dockerfile.dev .
```

然后和 5.2 一样起容器 `mori_oom_714`、装 MORI、跑 bench。7.14 的镜像里
`/opt/rocm/lib` 不在默认搜索路径上，单独跑的二进制需要 `LD_LIBRARY_PATH=/opt/rocm/lib`。

### 5.6 A/B 开关（在 10.1 上能复现之后再用）

- `CCO_UNCACHED_WINDOW=0`：arena 用 pinned 而不是 uncached（flags 变成 `0xb0000001`）。
- `CCO_GDR_CAPABLE=0`：分配不带 gpuDirectRDMACapable。
- `SWEEP=128,512,4096`：最大 token 数从 8192 降到 4096，arena 大约减半。如果这样就能
  过，说明和分配大小有关，支持 H1。

## 附录 A：kfd_svm_spy.c

```c
// LD_PRELOAD shim: logs KFD SVM registrations, VA-less VRAM allocations and failed
// KFD ioctls (with errno). Flags every VA-less VRAM allocation that an amdgpu
// without 93aa919 would reject with EADDRINUSE: its SVM-overlap check runs over
// [0, size) when va_addr == 0.
//   gcc -shared -fPIC -O2 -o kfd_svm_spy.so kfd_svm_spy.c -ldl
//   LD_PRELOAD=./kfd_svm_spy.so SPY_BT=3 python ...
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

struct svm_args {
  uint64_t start_addr, size;
  uint32_t op, nattr;
};
struct alloc_args {
  uint64_t va_addr, size, handle, mmap_offset;
  uint32_t gpu_id, flags;
};
struct avail_args {
  uint64_t available;
  uint32_t gpu_id, pad;
};
#define NR_ALLOC 0x16
#define NR_SVM 0x20
#define NR_AVAIL 0x23
#define VRAM_FLAG 1u

#define MAX_RANGES 4096
static uint64_t g_start[MAX_RANGES], g_len[MAX_RANGES];
static int g_n, g_bt_left = -1, g_other_fail_left = 20;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static void Backtrace(void) {
  if (g_bt_left < 0) g_bt_left = getenv("SPY_BT") ? atoi(getenv("SPY_BT")) : 0;
  if (g_bt_left <= 0) return;
  g_bt_left--;
  void* frames[24];
  int n = backtrace(frames, 24);
  backtrace_symbols_fd(frames + 2, n - 2, 2);
}

static int LowestSvmBelow(uint64_t size) {
  int hit = -1;
  for (int i = 0; i < g_n; i++)
    if (g_start[i] < size && (hit < 0 || g_start[i] < g_start[hit])) hit = i;
  return hit;
}

int ioctl(int fd, unsigned long req, ...) {
  static int (*real)(int, unsigned long, void*);
  if (!real) real = (int (*)(int, unsigned long, void*))dlsym(RTLD_NEXT, "ioctl");
  va_list ap;
  va_start(ap, req);
  void* arg = va_arg(ap, void*);
  va_end(ap);
  int ret = real(fd, req, arg);
  int err = errno;
  if (_IOC_TYPE(req) != 'K') return ret;
  pthread_mutex_lock(&g_mu);
  if (_IOC_NR(req) == NR_SVM) {
    struct svm_args* a = (struct svm_args*)arg;
    fprintf(stderr, "[spy %d] SVM op=%u start=0x%lx size=0x%lx nattr=%u ret=%d\n", getpid(), a->op,
            a->start_addr, a->size, a->nattr, ret);
    if (a->op == 0 && ret == 0 && g_n < MAX_RANGES) {
      g_start[g_n] = a->start_addr;
      g_len[g_n++] = a->size;
      Backtrace();
    }
  } else if (_IOC_NR(req) == NR_ALLOC) {
    struct alloc_args* a = (struct alloc_args*)arg;
    if (a->va_addr == 0 && (a->flags & VRAM_FLAG)) {
      int hit = LowestSvmBelow(a->size);
      if (ret != 0)
        fprintf(stderr,
                "[spy %d] VRAM alloc va=0 size=%.2f GB flags=0x%08x FAILED errno=%d (%s); lowest "
                "SVM range below the size: %s0x%lx\n",
                getpid(), a->size / 1073741824.0, a->flags, err, strerror(err),
                hit >= 0 ? "" : "none, ", hit >= 0 ? g_start[hit] : 0);
      else if (hit >= 0)
        fprintf(stderr,
                "[spy %d] VRAM alloc va=0 size=%.2f GB flags=0x%08x ok here, WOULD FAIL on an "
                "amdgpu without 93aa919: SVM range [0x%lx, +0x%lx) is below the size\n",
                getpid(), a->size / 1073741824.0, a->flags, g_start[hit], g_len[hit]);
      else if (a->size >= (64ul << 20))
        fprintf(stderr, "[spy %d] VRAM alloc va=0 size=%.2f GB flags=0x%08x ok, no SVM range below it\n",
                getpid(), a->size / 1073741824.0, a->flags);
    } else if (ret != 0) {
      fprintf(stderr, "[spy %d] alloc va=0x%lx size=0x%lx flags=0x%08x FAILED errno=%d (%s)\n",
              getpid(), a->va_addr, a->size, a->flags, err, strerror(err));
    }
  } else if (_IOC_NR(req) == NR_AVAIL) {
    struct avail_args* a = (struct avail_args*)arg;
    if (ret == 0 && a->available < (16ul << 30))
      fprintf(stderr, "[spy %d] KFD reports only %.2f GB available on gpu_id %u\n", getpid(),
              a->available / 1073741824.0, a->gpu_id);
  } else if (ret != 0 && g_other_fail_left > 0) {
    g_other_fail_left--;
    fprintf(stderr, "[spy %d] KFD ioctl nr=0x%x FAILED errno=%d (%s)\n", getpid(),
            (unsigned)_IOC_NR(req), err, strerror(err));
  }
  pthread_mutex_unlock(&g_mu);
  errno = err;
  return ret;
}

int munmap(void* addr, size_t len) {
  static int (*real)(void*, size_t);
  if (!real) real = (int (*)(void*, size_t))dlsym(RTLD_NEXT, "munmap");
  int ret = real(addr, len);
  int err = errno;
  if (ret == 0) {
    uint64_t lo = (uint64_t)addr, hi = lo + len;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_n; i++)
      if (g_start[i] >= lo && g_start[i] + g_len[i] <= hi) {
        g_start[i] = g_start[--g_n];
        g_len[i] = g_len[g_n];
        i--;
      }
    pthread_mutex_unlock(&g_mu);
  }
  errno = err;
  return ret;
}
```

## 附录 B：replay_ops_v2.sh（在容器里跑）

```bash
#!/bin/bash
# Replays ci_cco.yml's ops-v2 steps (dispatch/combine test .. bench) back to back,
# inside the CI container. Usage: replay_ops_v2.sh <workspace> <logdir>
W=$1
LOGS=$2
D=$W/tests/python/ops/dispatch_combine_v2
mark() { echo "@@ $(date +%T.%N | cut -c1-10) $*"; }
leftover() { echo "@@ leftover python procs: $(pgrep -c -f '[p]ython|[t]orchrun' || true)"; }
step() {
  local name=$1; shift
  mark "BEGIN $name"
  (cd "$D" && env "$@") > "$LOGS/$name.log" 2>&1
  local rc=$?
  mark "END $name rc=$rc"
  leftover
}
mkdir -p "$LOGS"
step 24-pytest-intranode bash -c "cd $W && python -m pytest tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_intranode.py -q"
step 25-hip-test-op MORI_V2_KERNEL_BACKEND=hip timeout 900 torchrun --standalone --nproc_per_node=8 test_op.py
step 26-hip-scales MORI_V2_KERNEL_BACKEND=hip SCALE_DIM=32 timeout 900 torchrun --standalone --nproc_per_node=8 test_op.py
for disp in fp8 fp4; do
  step 27-asym-$disp MORI_V2_KERNEL_BACKEND=hip SCALE_DIM=224 HIDDEN=7168 DISP=$disp timeout 900 torchrun --standalone --nproc_per_node=8 test_asym_dtype.py
done
step 28-parity timeout 900 torchrun --standalone --nproc_per_node=8 test_ep_backend_parity.py
for transport in sdma lsa; do
  step 29-asyncll-$transport MORI_ENABLE_SDMA=1 TRANSPORT=$transport timeout 900 torchrun --standalone --nproc_per_node=8 test_asyncll.py
done
step 30-routing-snapshot timeout 900 torchrun --standalone --nproc_per_node=8 test_routing_snapshot.py
step 31-bench BACKENDS=flydsl,hip COMB_MODE=pull MODES=graph SWEEP=128,512,4096,8192 ITERS=20 WARMUP=10 timeout 900 torchrun --standalone --nproc_per_node=8 bench_ep.py
mark DONE
```

## 附录 C：host 上的显存采样（每 0.5 秒一行，单位 MB，gpu0 到 gpu7）

```bash
( while true; do echo "$(date +%T.%N | cut -c1-10) $(amd-smi metric -m --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin); d=d.get('gpu_data',d) if isinstance(d,dict) else d
print(' '.join(str(g['mem_usage']['used_vram']['value']) for g in d))")" >> /tmp/oom/vram.log; sleep 0.5; done ) &
SAMPLER=$!
# ... 跑测试 ...
kill $SAMPLER
```

把 `replay_marks.log` 里每一步的 BEGIN/END 时间和 `vram.log` 对上，就能看出每一步
开始时的显存和峰值。

## 附录 D：pit2 上用过的其他测试

在 pit2-p03-g17 的 `/var/zqz/zqz_vmm_churn/` 下，需要的话可以拷过来：
`vmm_churn.cpp`（裸 HIP VMM 反复 alloc/free，有同进程和跨进程两种模式）和
`cco_churn.py`（MORI 的 `alloc_mem`、`register_window`、free 循环，用 mpirun 跑）。
在 pit2 上这两个都显示释放及时。H2 的情况下可以在 51 上跑来对照。
