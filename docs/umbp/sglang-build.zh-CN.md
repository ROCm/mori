# 从公开基础镜像构建 SGLang + Mori UMBP demo

**不需要 `yangwu/umbp-resume:20261015`。** 本仓库提供 Dockerfile、启动入口、
环境检查、HTTP smoke、自动生成输入的 DRAM fill/replay，以及接入源码导读。
同事只需要获取本 demo 分支、公开基础镜像，并准备四卡 MI455 和 DeepSeek V4 Pro 模型。
镜像构建和 CPU 检查不需要 GPU 或模型。

先学习接口可阅读 [UMBP 使用手册](user-guide.zh-CN.md) 和
[SGLang 接入代码导读](sglang-linker-walkthrough.zh-CN.md)。
[历史实验记录](sglang-demo.zh-CN.md) 保留原来的配置、输入移交要求和 A/B 结果。

## 1. 基础镜像与版本

已核实历史 resume 镜像的直接父镜像对应以下公开镜像：

```text
tag:      lmsysorg/sglang-rocm:v0.5.21-rocm10-mi45x-20261008
manifest: lmsysorg/sglang-rocm@sha256:9b82c0b334618afe6182a542fe4adf3694b9b059edfeb41b997a22f2fd4c0a86
Image ID: sha256:392ff204760b8179d49b99daa593835fee8d2250d4268d0bd137093103de2ef6
```

tag 与固定 manifest 的 config digest 均匹配这个父镜像的 Image ID。
manifest digest 用于从 registry 拉取；Image ID 是镜像配置的 digest，两者是不同对象。
Dockerfile 默认使用固定 manifest，避免 tag 更新影响软件组合。

| 组件 | 固定内容 |
|---|---|
| SGLang Git HEAD | `943621c3364de6948fc02ee13c55b4f1cdf97f3a`，保留基础镜像内 vendor 修改 |
| Mori Git HEAD | `879983bdbd8c65c52e9f79ad836a61cbffef98b6` |
| AITER Git HEAD | `a6d2b564fd671724a3720b8edf70e8d674e4d694`，保留基础镜像内 vendor 修改 |
| PyTorch / HIP | `2.11.0+rocm10.0.0` / `7.15.26333` |
| UMBP standalone | 复用基础镜像已编译程序，SHA256 `7a1eaff3d0c9ce4c4b8d1c853b888151d38666acac93a9d0b4f86f686dbc4c62` |
| 新增模型修正 | DeepSeek V4 mHC input normalization 的 `fn_transpose=False` 改为 `True` |
| supervisor / launcher | 本仓库内 `recipe/`，固定来源提交 `d26286ef2bf8b6cc14d1624aa19df550e7e28a66` |

基础镜像已经提供 `UMBPDirectLinker`、Mori Python 扩展与 standalone；默认构建不再编译 Mori，
也不需要 `pip install` 最新 SGLang。基础镜像中存在的 vendor 修改属于固定运行环境，
只 clone 上述 Git HEAD 再重新安装无法保证得到相同软件。

构建程序严格检查版本和 hash，应用模型修正，复制完整 SGLang Python 包到
`/opt/umbp-demo/integrated/`，生成新的逐文件 source manifest 和构建来源记录。
修正前后文件 SHA256 与 Linker hash 见
[runtime-lock.json](../../examples/umbp/sglang/runtime-lock.json)。
这个新 manifest 标记为 `new-demo-snapshot`；历史 `integrated/` 的完整 manifest 未找回，
因此新镜像用于重新验证接入，历史性能数字仍只归属于原来的测试环境。

## 2. 获取代码、构建和 CPU 检查

在 Linux 主机上准备 Git、Bash、tar 和可用的 Docker daemon。基础镜像展开约 54 GB，
建议 Docker 存储预留至少 100 GiB；模型和运行日志另外计入空间。

```bash
git clone --branch demo/umbp-sglang-tp4 --single-branch \
  https://github.com/ROCm/mori.git mori-umbp-demo
cd mori-umbp-demo

export TEST_IMAGE=mori-umbp-sglang:tp4-20261008
bash examples/umbp/sglang/run_demo.sh build
bash examples/umbp/sglang/run_demo.sh check
bash examples/umbp/sglang/run_demo.sh preview
```

`build` 从固定公开基础镜像构建，使用小 tar context，支持 BuildKit 和 legacy Docker builder；
不会把整个 Mori checkout 或本地 build 目录送给 Docker。
构建时会完成全部源码/hash 校验、4 MiB DRAM KV 字节校验、11 项原始 recipe CPU 检查，
以及本仓库新增的 workload/HTTP 失败处理检查。

`check` 不暴露 GPU，确认 Python UMBP 能加载、PyTorch/HIP 版本一致、
standalone hash 正确，且源码快照未变；它执行小容量进程内 DRAM put/get。
`preview` 只打印完整服务命令、环境和 128 GiB payload 配置，不创建运行目录，不启动服务。

若使用 BuildKit，也可以直接在仓库根目录执行：

```bash
docker build -t "$TEST_IMAGE" -f examples/umbp/sglang/Dockerfile .
```

### 使用已有基础镜像或离线移交

如果主机已经有原始基础镜像，可先核对 Image ID，再使用本地 tag 构建：

```bash
docker image inspect --format '{{.Id}}' lmsysorg/sglang-rocm:v0.5.21-rocm10-mi45x-20261008
# 必须匹配上面的基础 Image ID。
export BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.21-rocm10-mi45x-20261008
bash examples/umbp/sglang/run_demo.sh build
unset BASE_IMAGE
```

`BASE_IMAGE` 用于指向同一个已核实基础镜像；源码和二进制校验仍会执行。
若没有这个镜像，默认公开 manifest 会由 Docker 自动拉取。

离线同事可接收构建后的镜像和代码，不需要个人 resume 镜像：

```bash
docker save "$TEST_IMAGE" | gzip > mori-umbp-sglang-tp4.tar.gz
sha256sum mori-umbp-sglang-tp4.tar.gz > mori-umbp-sglang-tp4.tar.gz.sha256
# 在接收端核对校验文件，再加载：
sha256sum -c mori-umbp-sglang-tp4.tar.gz.sha256
gzip -dc mori-umbp-sglang-tp4.tar.gz | docker load
```

默认流程复用公开镜像的底层工具链。若要从空白 Ubuntu 开始重编 Torch/AITER/SGLang/Mori，
那需要另一套工具链构建与模型验证；本 demo 的 Dockerfile 没有声称提供这一过程。

## 3. 模型与硬件准备

实际推理沿用四卡 MI455/gfx1250、DeepSeek V4 Pro、TP4/DP1/EP1、BF16 KV、
65,536 GPU KV tokens、page 256、chunked prefill 512、单个 128 GiB standalone DRAM 池。
主机 DRAM 要覆盖 128 GiB payload、模型加载、临时缓冲区和系统开销。
主机 ROCm 驱动需与镜像兼容；CPU 检查不能替代 GPU 探测或模型加载。

模型权重不包含在 Git 或镜像内，需要准备完整模型目录：

```text
DeepSeek-V4-Pro/
  config.json
  tokenizer_config.json
  tokenizer.json                 # 或该 tokenizer 所需的其他完整文件
  model.safetensors.index.json
  model-00001-of-....safetensors
  ...                            # index 中引用的全部 shard
```

根据同事自己的主机修改目录；这些路径不依赖作者账号：

```bash
export MODEL_HOST_PATH=/data/models/DeepSeek-V4-Pro
export RUN_HOST_PATH=/data/umbp-sglang-runs
mkdir -p "$RUN_HOST_PATH"

bash examples/umbp/sglang/run_demo.sh check
bash examples/umbp/sglang/run_demo.sh check-hardware
```

设置 `MODEL_HOST_PATH` 后，`check` 会额外检查模型类型、weight index 引用的 shard 是否存在且非空、
以及 tokenizer 能否本地加载；不下载模型，也不计算所有权重 hash。
此检查不能证明模型来源或完整权重正确性，仍需使用指定 Pro 模型。
其他 DeepSeek 型号需要重新核对布局、容量和启动配置。

`check-hardware` 检查 `/dev/kfd`、`/dev/dri`、四个 AMD GPU 的 sysfs 内存计数器、
`amd-smi` JSON schema 和 dmesg 权限；只查询状态，不运行 GPU kernel。
其结果显示当前 GPU 进程，不会停止它们。实际服务由 supervisor 等待空闲窗口后运行。

运行入口使用 host network/IPC/PID、GPU 设备、`--privileged` 和诊断权限，
以满足固定 recipe 的进程及 kernel 日志检查。默认 HTTP 只绑定 `127.0.0.1`。

## 4. 执行 smoke、DRAM 恢复或交互式服务

先执行两次相同 prompt 的 HTTP smoke：

```bash
bash examples/umbp/sglang/run_demo.sh smoke smoke-v1
```

它启动私有 standalone 和 SGLang，检查四个 rank 的 Direct Linker attachment 和实际池容量，
然后运行 [sglang_linker_smoke.py](../../examples/umbp/sglang_linker_smoke.py)。
结果保存在 `$RUN_HOST_PATH/smoke-v1/`，client 完成后 supervisor 清理本次服务，容器自动退出。
两次短请求可以命中 GPU KV；验证 DRAM 恢复需执行下面的 fill/replay：

```bash
bash examples/umbp/sglang/run_demo.sh fill-replay dram-v1
```

[fill_replay.py](../../examples/umbp/sglang/fill_replay.py) 从模型 tokenizer 生成固定 seed 的
24 个不同前缀，每个 4,096 input tokens、32 output tokens、温度 0、并发 2。
先 fill，再回放同一组输入，总共 48 请求，stage 之间不 flush。
它保存 input IDs/hash、完整成功响应、阶段耗时和 device/host/storage cache counts。
**replay 至少一次 `cached_tokens_details.host > 0` 才通过**，纯 GPU hit 会报错。
这是新生成的功能 workload，不含历史 retrieval 答案检查，也不是原来的 168 请求性能基准。
默认总输入量超过 65,536-token GPU 预算，但是否发生 DRAM 恢复以实际 metadata 为准。

需要交互式服务时，在终端 A 执行：

```bash
bash examples/umbp/sglang/run_demo.sh serve interactive-v1
```

等待 `$RUN_HOST_PATH/interactive-v1/ready.json`，在同一主机终端 B 发送请求：

```bash
python3 examples/umbp/sglang_linker_smoke.py \
  --url http://127.0.0.1:18000 \
  --out "$RUN_HOST_PATH/interactive-http-v1"
```

终端 A 的 Ctrl-C 或 supervisor timeout 会清理本次子进程。默认模型运行上限 3,600 秒，
从空闲检查结束后计时；此前最多等待 30 分钟，要求连续 300 秒空闲。
新运行必须使用新目录名，避免覆盖已有结果。可调整端口与时间：

```bash
PORT=18001 NCCL_PORT=19501 IDLE_SECONDS=300 RUN_TIMEOUT=3600 \
  bash examples/umbp/sglang/run_demo.sh smoke smoke-v2
```

默认保持 `kernel_sync`：`AMD_SERIALIZE_KERNEL=3` 和 `HIP_LAUNCH_BLOCKING=1`。
学习和首次验证使用此配置；调整异步、模型、布局或容量后，需要重新做稳定性和性能测试。

## 5. 验收、来源记录和排查

| 文件 | 验收内容 |
|---|---|
| `model.log` | TP0..3 均出现 `UMBPDirectLinker topology=StandaloneProcess+... ranged_io=yes` 与 `hicache_attached=False` |
| `standalone.log`、`capacity.json` | 一个 DRAM payload pool，`total_bytes=137438953472` |
| `ready.json`、`server-info.json` | 服务地址与实际模型参数 |
| `client/complete.json` | smoke 的两次请求通过，或 fill/replay 的 DRAM hit 验收通过 |
| `client/workload.json`、`client/summary.json` | 新 fill/replay 的完整输入、hash 与缓存命中统计 |
| `state.json` | supervisor `error=null`，且进程正常清理；结合 ECC/kernel snapshots 查看设备状态 |

镜像内的 `/opt/umbp-demo/build-provenance.json` 和 `source-manifest.json` 可导出给同事：

```bash
docker image inspect --format '{{.Id}}' "$TEST_IMAGE" > "$RUN_HOST_PATH/demo-image-id.txt"
docker run --rm --network none --entrypoint cat "$TEST_IMAGE" \
  /opt/umbp-demo/build-provenance.json > "$RUN_HOST_PATH/build-provenance.json"
docker run --rm --network none --entrypoint cat "$TEST_IMAGE" \
  /opt/umbp-demo/source-manifest.json > "$RUN_HOST_PATH/source-manifest.json"
```

| 问题 | 处理方式 |
|---|---|
| 构建时版本/hash 不匹配 | 核对基础 manifest 和 Image ID；保留失败日志，使用指定基础镜像重新构建 |
| registry 无法访问 | 从有网络的环境移交已校验镜像；使用 `BASE_IMAGE` 指定同一个本地基础镜像 |
| 模型目录不存在 / 缺 shard / tokenizer 无法加载 | 修正宿主机目录并补齐模型文件，再执行 `check` |
| `amd-smi` schema、dmesg 或 sysfs 检查失败 | 核对四卡设备、主机驱动和容器诊断权限，查看固定 recipe 的预期格式 |
| 无空闲窗口、端口被占用 | 等待其他任务结束或选择空闲端口；supervisor 不结束其他任务 |
| replay 无 host hit | 查看 attachment、offload/get 错误和实际 workload；此结果不满足 DRAM 恢复验收 |
| GPU fault / timeout | 保留 model、probe、ECC、kernel 和 state 日志，先排查运行环境及 kernel |

2026-11-10 已验证：从原始公开基础镜像内容构建、严格 source/hash 检查、CPU DRAM roundtrip、
原始 11 项 recipe 检查、新 workload/HTTP 检查及配置预览。
本轮未运行完整模型或四卡 fill/replay；真实 GPU 验收仍需上述模型与空闲硬件。
