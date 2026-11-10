# Mori UMBP 文档与 SGLang demo

本组文档介绍 Mori UMBP 的本机使用，以及 MI455 单机 TP4 实验中的 SGLang Mori Direct Linker 对接。

文档与示例保存在独立分支
[`demo/umbp-sglang-tp4`](https://github.com/ROCm/mori/tree/demo/umbp-sglang-tp4)，
以 Mori 主线为基础，只包含本组文档和 demo。

| 文档或示例 | 内容 |
|---|---|
| [UMBP 使用手册](user-guide.zh-CN.md) | 安装、进程内使用、standalone 共享、DRAM/SSD 配置、指针与范围 API、内存生命周期、性能调优与排查 |
| [SGLang 构建与运行](sglang-build.zh-CN.md) | 从公开基础镜像构建，CPU/模型/硬件检查，启动、smoke、自动生成输入的 DRAM fill/replay、离线移交和排查 |
| [SGLang 接入代码导读](sglang-linker-walkthrough.zh-CN.md) | Linker/Store 接口、初始化、page/range 映射、offload/load、完成通知和性能参数 |
| [SGLang 历史实验记录](sglang-demo.zh-CN.md) | 固定实验版本、Linker 结构、原始附件、168 请求回放与历史结果 |
| [本机 KV 示例](../../examples/umbp/local_kv_demo.py) | 小容量 DRAM、SSD 和 standalone 指针读写，逐字节校验 |
| [SGLang 请求示例](../../examples/umbp/sglang_linker_smoke.py) | 检查服务端 Linker 参数，发送两次相同请求并记录 cache metadata |
| [SGLang demo 入口](../../examples/umbp/sglang/run_demo.sh) | 一处入口完成 build/check/preview/serve/smoke/fill-replay |
| [固定接入源码](../../examples/umbp/sglang/reference/umbp_direct_linker.py) | 来自公开基础镜像的完整 UMBPDirectLinker，配套 Store 和接口也保存在 reference/ |

建议同事按“使用手册 → 接入代码导读 → 构建与运行”的顺序学习。
新 demo 不需要个人 resume 镜像或历史输入附件，构建和 CPU 检查不需要 GPU；
真实 TP4 推理需要另行准备四卡 MI455 和完整 DeepSeek V4 Pro 模型。

历史记录以
[`repro/tp4-umbp-direct`](https://github.com/inkcherry/MI455_inferencemax/tree/d26286ef2bf8b6cc14d1624aa19df550e7e28a66/repro/tp4/umbp)
分支的固定提交为依据。原始 supervisor/launcher 已固定收录到本仓库；
精确重跑历史 A/B 仍需当时的集成源码 manifest 和输入附件。

使用手册核对的是 Mori 当前代码接口；SGLang 镜像固定兼容的 Mori/SGLang/AITER 组合。
先确认版本，再执行对应文档的命令。
