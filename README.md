# Inkling 2 Mini → Ascend

面向 **Atlas A2 910B1 64 GB × 8** 与 **Atlas A5 950PR / 950DT × 8** 的 Inkling 2 Mini 移植工程资料，目标引擎 **vLLM Ascend**。先完成纯文本，再接入图像。

**当前状态：工程设计、公开源码核验和辅助脚本；尚未实现完整 NPU 模型后端，尚未在 A2 / A5 上执行模型推理或性能测试。** 部署脚本默认仅打印命令。模型权重需从上游单独获取。

资料核验日期：**2026-11-24**。具体证据、固定源码快照、已知缺口见 [来源清单](docs/sources.md)。

## 工程结论

适配具备明确的工程路径。主要工作集中在 **loop / AttnRes 调度、逐 pass 独立缓存、相对位置注意力与卷积状态、混合量化权重加载、MoE 并行与通信**。不能仅靠注册模型名或替换 CUDA 设备字符串完成。

| 条件 | 首期决策 |
|---|---|
| A2：910B1，单卡 64 GB，单机八卡 | 以 BF16 解量化权重、eager、小上下文开始；完整模型优先 TP=2 或 TP=4；逐步验证 TP=8 |
| A5：950PR / 950DT，单卡显存未确认 | 先采集具体设备、驱动、固件、CANN 与显存；分别建立验证记录 |
| 已有 CANN，版本未提供 | 锁定实际软件组合后建基线，参考组合不能替代本机版本确认 |
| 目标引擎 vLLM Ascend | 新增模型实现和 backend，优先复用公共调度器与缓存管理接口 |
| 先文本、后图像 | 首期显式关闭 image 输入；图像阶段单独验收处理器、编码器、特征映射与调度 |

### 已核实且影响设计的细节

- 固定 checkpoint 为 **24 个物理层、48 次层访问**：`6 + 12 × 3 + 6`。同一核心层复用权重，pass 的状态缓存仍需独立。
- 隐藏维度 `2304`，Q / KV heads 为 `24 / 8`，**head_dim=128**；投影宽度不等于隐藏维度除以头数。
- 注意力同时涉及 relative logits、L2 clipping、value norm、短卷积、gate / sink。通用 FlashAttention 或 CANN fused attention 不能在未核验语义时直接替换。
- checkpoint 专家权重混用 **NVFP4 / MXFP8**。上游 NVIDIA 专家实现不直接支持 BF16 专家权重；BF16 基线需要转换流程、专用加载器与计算路径。
- MoE 使用专门的 **router 与 SiTU 激活**：选择分数为 sigmoid(logit)+bias，权重由所选原始 logits 计算；routing weight 在 down projection 前应用，不能直接换成普通 SwiGLU 路径。
- 低精度解量化后的 BF16 是对已发布权重的重建，并非原始训练 BF16 权重。关键 FP32 参数与计算精度必须保留。
- 较早的 Inkling RFC / MTP 草稿不是 Inkling 2 Mini 的完整 Ascend 支持；Mini、Small、旧 Inkling 的接口不能混用。

依据：[架构与参数](docs/architecture.md)、[上游来源与版本](docs/sources.md)、[量化与加载](docs/quantization.md)。

## 阅读路径

| 要做的工作 | 文档 |
|---|---|
| 理解模型、循环与权重共享 | [architecture.md](docs/architecture.md) |
| 按仓库文件拆改造任务 | [vllm-ascend-integration.md](docs/vllm-ascend-integration.md) |
| 实现算子、核对张量形状和精度 | [operator-design.md](docs/operator-design.md) |
| 设计 KV / conv 状态与多请求生命周期 | [cache-and-loop.md](docs/cache-and-loop.md) |
| 处理 NVFP4 / MXFP8、BF16 基线和权重索引 | [quantization.md](docs/quantization.md) |
| 确认硬件、内存和软件矩阵 | [ascend-hardware.md](docs/ascend-hardware.md) |
| 决定 TP、专家分片与通信 | [parallel-strategy.md](design/parallel-strategy.md) |
| 环境检查、启动与服务验收 | [deployment.md](docs/deployment.md) |
| 数值、缓存、系统与性能测试 | [benchmark-plan.md](docs/benchmark-plan.md) |
| 第二阶段图像接入 | [multimodal.md](docs/multimodal.md) |
| 里程碑与可直接拆分的任务 | [ROADMAP.md](ROADMAP.md) |

## 现在可执行的操作

这些命令只依赖本仓库辅助脚本；不代表模型适配已经完成。

```bash
git clone https://github.com/lgw2023/inkling2-ascend.git
cd inkling2-ascend

# 只使用 Python 标准库，核对示例配置的层访问和逻辑 KV 预算
python3 scripts/inspect_model_config.py configs/model_shape.example.json --tp 4 --tokens 8192

# 本地收集白名单环境版本；先阅读脚本和部署文档
bash scripts/collect_env.sh

# 查看部署模板的参数；默认 dry-run，不启动模型
bash scripts/serve_a2.sh --help
bash scripts/serve_a5.sh --help

# 校验纯 CPU 规划工具
python3 -m unittest discover -s tests -v
```

`configs/model_shape.example.json` 是从指定上游版本提取的**结构子集**，用于规划工具，不能作为模型完整 `config.json` 加载。KV 预算不包含卷积状态、消息 bank、分页对齐、权重、通信区、激活、运行时和图内存。

## 验证状态

| 项目 | 状态 |
|---|---|
| 公开源码路径、模型配置与关键算子语义核验 | 已整理，证据见各文档 |
| 本仓库规划工具与部署模板检查 | 见 [tests/README.md](tests/README.md) |
| Ascend 模型注册、BF16 MoE、注意力 / conv backend | 待开发 |
| A2 / A5 模型数值对齐和推理 | 待执行 |
| 吞吐、TTFT、TPOT、显存与扩展效率 | 待测量，无预填性能值 |

首个验收目标是 **A2 上固定 checkpoint、固定软件版本、固定 loop 次数的纯文本正确性基线**，通过后再优化算子、图模式、批处理和多卡扩展。

## 协作

提交方式与证据要求见 [CONTRIBUTING.md](CONTRIBUTING.md)，任务模板在 `.github/ISSUE_TEMPLATE/`。源码引用保留上游许可证；本仓库未包含模型权重或完整上游代码副本。授权状态见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。


## 上游推理源码

已固定 vLLM、vLLM Ascend、SGLang、Transformers、llama.cpp 和 Maya 的上游 commit。使用 `git clone --recurse-submodules` 下载完整参考代码。版本锁、权重 revision 和许可证说明见 [references/README.md](references/README.md)。
