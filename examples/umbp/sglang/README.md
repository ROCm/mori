# SGLang + Mori UMBP TP4 demo

中文入口：[从公开基础镜像构建与运行](../../../docs/umbp/sglang-build.zh-CN.md)，
[接入源码导读](../../../docs/umbp/sglang-linker-walkthrough.zh-CN.md)。

```bash
# 在 Mori 仓库根目录执行；构建/检查不需要 GPU 或模型。
bash examples/umbp/sglang/run_demo.sh build
bash examples/umbp/sglang/run_demo.sh check
bash examples/umbp/sglang/run_demo.sh preview
```

准备好四卡 MI455 和 DeepSeek V4 Pro 后：

```bash
export MODEL_HOST_PATH=/data/models/DeepSeek-V4-Pro
export RUN_HOST_PATH=/data/umbp-sglang-runs
bash examples/umbp/sglang/run_demo.sh check
bash examples/umbp/sglang/run_demo.sh check-hardware
bash examples/umbp/sglang/run_demo.sh smoke smoke-v1
bash examples/umbp/sglang/run_demo.sh fill-replay dram-v1
```

| 文件 | 内容 |
|---|---|
| `Dockerfile`、`runtime-lock.json`、`prepare_image.py` | 固定公开基础镜像与版本，校验和应用模型修正，生成新的 SGLang 源码快照 |
| `run_demo.sh`、`container_entry.py` | 宿主机/容器入口；模型只读挂载、结果保存、固定 supervisor 参数 |
| `preflight.py` | source/hash/Python KV 检查，可选模型文件与设备可见性检查 |
| `fill_replay.py` | 新生成的 24-prefix/48-request DRAM restore 功能验证 |
| `test_demo.py` | 无 GPU 的 workload、模拟 HTTP 验收和失败处理检查 |
| `reference/` | 原始 SGLang Linker、Store、接口源码及 Apache 2.0 license；从固定公开镜像提取，用于学习 |
| `recipe/` | 固定原始启动、探测、历史回放脚本；`source.json` 记录来源和每文件 hash |

`recipe/` 原样来自
[`inkcherry/MI455_inferencemax@d26286ef`](https://github.com/inkcherry/MI455_inferencemax/tree/d26286ef2bf8b6cc14d1624aa19df550e7e28a66/repro/tp4/umbp)。
`reference/` 的来源信息见其 `source.json`；它不作为一个独立 SGLang 安装包。
构建使用基础镜像内完整依赖，保留 vendor 修改，不要求个人 resume 镜像。
新 source manifest/workload 不代表历史 A/B 附件已恢复。

在不安装模型依赖的主机上检查新的客户端逻辑：

```bash
python3 examples/umbp/sglang/test_demo.py
```

真实模型和 GPU fill/replay 仍需按构建文档执行验收。
