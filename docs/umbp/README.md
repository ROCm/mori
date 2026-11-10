# Mori UMBP 文档与 SGLang demo

本组文档介绍 Mori UMBP 的本机使用，以及 MI455 单机 TP4 实验中的 SGLang Mori Direct Linker 对接。

| 文档或示例 | 内容 |
|---|---|
| [UMBP 使用手册](user-guide.zh-CN.md) | 安装、进程内使用、standalone 共享、DRAM/SSD 配置、指针与范围 API、内存生命周期、性能调优与排查 |
| [SGLang 对接 demo](sglang-demo.zh-CN.md) | 固定实验版本、Linker 结构、附件准备、服务启动、两次请求、168 请求回放与历史结果 |
| [本机 KV 示例](../../examples/umbp/local_kv_demo.py) | 小容量 DRAM、SSD 和 standalone 指针读写，逐字节校验 |
| [SGLang 请求示例](../../examples/umbp/sglang_linker_smoke.py) | 检查服务端 Linker 参数，发送两次相同请求并记录 cache metadata |

建议先完成本机 KV 示例，再准备 SGLang demo 的运行环境。SGLang demo 以
[`repro/tp4-umbp-direct`](https://github.com/inkcherry/MI455_inferencemax/tree/d26286ef2bf8b6cc14d1624aa19df550e7e28a66/repro/tp4/umbp)
分支的固定提交为依据。该仓库提供复现脚本；集成源码快照、模型、历史输入和本地镜像需要另行移交。

使用手册核对的是 Mori 当前代码接口；SGLang 历史回放使用的是实验时的固定镜像和源码组合。
先确认版本，再执行对应文档的命令。
