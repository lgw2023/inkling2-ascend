# 来源、版本与核验记录

核验日期：**2026-11-24（北京时间）**。本文件区分公开事实、工程推导和待测方案。仓库没有发布 Ascend 性能成绩，也没有将上游 NVIDIA 测试标成本项目测试。

## 1. 固定的研究基准

| 对象 | 固定值 | 用途与说明 |
|---|---|---|
| vLLM Mini 引入提交 | `ab18334a4f60da74db7de89fe95594e5ed0cda36` | 由 PR #69028 的 `merge_commit_sha` 取得；核心源码已按该 ref 读取 |
| Mini 发布模型初始上传 | `1fb4134b9a74f9681054f4ef82e4bf1f458a2337` | HF commit 页面提供 `config.json` 及量化 metadata；生产应固定实际下载 revision |
| vLLM Ascend 当前阅读快照 | `1999fa582aba279dc4214de7ea3035535917d4c4` | 用于核对 registry、MoE 和当前代码结构；该快照走 `worker/v2/model_runner.py`，**不作为本项目执行器基底** | 
| vLLM Ascend 历史 V1 runner 参考 | `f16e5e26801f86b071bca6454c9b9b2b63362e0a` | 包含 `worker/v1/model_runner.py` 和 V1/V2 选择逻辑；作为 V1 回移参考，不是与当前 vLLM Mini 已验证的版本组合 |
| 本机软件和 A5 显存 | 尚未采集 | 用户已确认存在 CANN 环境，未提供具体版本与 A5 容量 |

注意：GitHub 文件接口的 `sha` 通常是 blob SHA，不能当作仓库 commit pin。本表的 vLLM SHA 来自 PR 合并元数据，Ascend SHA 来自仓库提交记录并已据此复读文件。**两个源码快照不是经过验证的兼容软件组合。** 开始实现时应核对双方接口及配套依赖，完成版本资格验证。

## 2. 发布方资料

| 编号 | 来源 | 本仓采用的事实 |
|---|---|---|
| S1 | [Thinking Machines：Inkling 2 Mini Model Card](https://thinkingmachines.ai/model-card/inkling-2-mini/) | 26.4B 参数、3.4B 物理激活/约 5.4B 有效激活、24 层共享核心三轮、输入文本/图像输出文本、1M context、混合 NVFP4/MXFP8 |
| S2 | [Hugging Face 模型](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | 发布权重入口及模型身份 |
| S3 | [HF 初始上传提交](https://huggingface.co/thinkingmachines/Inkling-2-Mini/commit/1fb4134b9a74f9681054f4ef82e4bf1f458a2337) | 已读取 commit diff 中完整 config：H2304、24Q/8KV、head128、core[6,18)×3、rank64、vision23 等 |
| S4 | [hf_quant_config.json](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/hf_quant_config.json) | 已读取各精度组与尺度规则；当前页面的上传提交为上述 S3 |

读取情况：本次检索未能直接打开 HF `config.json` 的 raw/blob 页面，改由已成功读取的上传 commit diff 核验内容。没有据此声称已下载全部权重或完成转换。

## 3. vLLM 源码

以下源码链接固定到同一 vLLM commit。

| 编号 | 源码/记录 | 核验内容 |
|---|---|---|
| V1 | [PR #69028](https://github.com/vllm-project/vllm/pull/69028) | 标题 `[Model] Add Inkling 2 Mini`；北京时间 2026-11-22 合并（UTC 11-21）；同时包含 variable-loop research checkpoint；验证硬件列为 NVIDIA |
| V2 | [configs.py](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/configs.py) | loop 字段校验、visit 数计算、固定/可变 checkpoint 限制、DSpark visit 约束 |
| V3 | [nvidia/loop.py](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/loop.py) | Prelude 保存/恢复、上一轮 bank 克隆、四流 controller、FP32 消息融合、独立 pass owner |
| V4 | [nvidia/model.py](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/model.py) | vision 按 multimodal limit 实例化、权重名称/切分映射、dtype 严格检查、拒绝 BF16 expert |
| V5 | [nvidia/attention.py](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/attention.py) | q/r/gate/sink/k/v、动态/静态 K/V conv、相对位置 logits、pass cache、LSE、TP partial 输出 |
| V6 | [quant_config.py](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/quant_config.py) | 每 tensor 精度 metadata；expert 层格式选择 |
| V7 | [nvidia/moe.py](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/moe.py) | 路由参数、量化 expert 参数及 loader、输出归约契约 |

算子和 cache 的更细引用见 [operator-design.md](operator-design.md)、[cache-and-loop.md](cache-and-loop.md) 及 [vllm-ascend-integration.md](vllm-ascend-integration.md)。Ascend 阅读快照上的关键入口如下：

- [模型注册入口 `vllm_ascend/models/__init__.py`](https://github.com/vllm-project/vllm-ascend/blob/1999fa582aba279dc4214de7ea3035535917d4c4/vllm_ascend/models/__init__.py)。
- [当前 Ascend Model Runner V2（仅用于识别并排除）](https://github.com/vllm-project/vllm-ascend/blob/1999fa582aba279dc4214de7ea3035535917d4c4/vllm_ascend/worker/v2/model_runner.py)。
- [历史 Ascend V1 Model Runner](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/v1/model_runner.py)。
- [历史 worker V1/V2 选择逻辑](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/worker.py)。
- [Ascend fused MoE](https://github.com/vllm-project/vllm-ascend/blob/1999fa582aba279dc4214de7ea3035535917d4c4/vllm_ascend/ops/fused_moe/fused_moe.py)。

## 4. Ascend 与旧 Inkling 记录

| 来源 | 核验当日状态 | 能说明什么 |
|---|---|---|
| [Ascend RFC #20812](https://github.com/vllm-project/vllm-ascend/issues/20812) | Open；明确实现与 Ascend 验证待完成 | 较早 Inkling/MTP 适配意向；不是 Mini 完成公告 |
| [Ascend PR #20831](https://github.com/vllm-project/vllm-ascend/pull/20831) | Open、Draft；`head_sha=1905c66adeca4f1e8bba7d2824d8809e4444cec0` | 两文件、24 行新增；PR 说明 helper 未接入 runtime、NPU 未测 |
| [旧 Inkling PR #48799](https://github.com/vllm-project/vllm/pull/48799) | RFC 引用的较早模型来源 | 不作为 Mini 代码基准 |
| [旧 MTP PR #48869](https://github.com/vllm-project/vllm/pull/48869) | RFC 引用的 single-token MTP 来源 | 不等同 Mini 的 DSpark 接入 |

本次核验了 RFC 正文、#20831 元数据和完整变更文件名列表；旧 PR 在此仅用于识别引用关系，未声称逐文件重新审查。#20831 的测试说明只有 lint/语法/空白检查，且完整仓库检查曾在环境初始化阶段停止；不能将其描述为 NPU 可运行实现。

## 5. 工程结论的证据等级

| 结论 | 类型 | 根据 |
|---|---|---|
| 24 物理层对应 48 次访问 | 配置事实 + 直接计算 | S3、V2、V3 |
| 固定 Mini 每轮独立 KV/conv owner | 上游实现事实 | V3、V5 |
| Ascend 可按自定义模型/后端路线适配 | 工程判断 | 模型计算可拆为通用线性、归一化、路由、卷积、attention；具体 kernel 尚待实现 |
| BF16 路线需转换器和新 expert/loader | 上游约束 + 工程设计 | S4、V4、V6、V7 |
| A2 完整模型先 TP2/TP4 | 工程建议 | 用户 64 GB 条件与粗略容量预算，尚无实测吞吐 |
| A5 可用显存、原生低精度支持和最佳 TP | 待确认 | 必须采集硬件与配套版本，并测试目标算子 |

## 6. 开始实现前需要补齐的证据

- 模型实际 revision、所有 shard SHA256、license 文件和 tensor header 清单。
- 选定 vLLM 与 vLLM Ascend commit、相应安装说明/支持矩阵，以及 driver/firmware/CANN/torch/torch_npu 版本。
- 两类 A5 的 `npu-smi` 设备识别、显存与实际 CANN 算子支持；不要由产品简称推断配置。
- 同一权重、输入、dtype 的参考 trace；先按 physical layer/pass/visit 记录，再做端到端质量对比。
- 每项「支持」声明对应的测试配置和结果文件。未产生测试结果前保持计划状态。


## 7. 推理代码仓库与新增参考（2026-11-24）

- [Hugging Face Inkling 2 Mini 专题](https://huggingface.co/blog/inkling2mini)：比旧的 `thinkingmachines-inkling.md` 更贴近 Mini，覆盖 Transformers、SGLang、llama.cpp 等部署。
- [vLLM Inkling 2 Mini Recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini)：官方部署配方，记录 DSpark 与 NVIDIA 验证环境。
- [vLLM Inkling2 模型实现](https://github.com/vllm-project/vllm/tree/28fb1365a90e9929e4596a72906e309dd7ac701f/vllm/models/inkling2) 和 [模型测试](https://github.com/vllm-project/vllm/tree/28fb1365a90e9929e4596a72906e309dd7ac701f/tests/models/inkling2)：完整源码通过 submodule 固定在本仓库。
- [SGLang](https://github.com/sgl-project/sglang) 与 [Transformers](https://github.com/huggingface/transformers)：官方支持的 serving 与 HF processor/model 参考。
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：GGUF loop 架构与量化推理参考，需匹配支持 Inkling2 的版本。
- [Maya Inkling Mini](https://github.com/alesha-pro/maya-inkling-mini)：MIT 代码的 CUDA/HIP 独立实现，适合研究专家计算、KV/conv 缓存、loop、AttnRes 和 DSpark；不是昇腾后端代码。
- [NVIDIA Dynamo Mini recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/README.md)：以 SGLang 为执行引擎的集群 serving 编排参考，不是新的模型算子实现。

官方模型卡还列出 TokenSpeed、Unsloth、Llama App，以及 Apple Silicon 上的 Ollama / Exo，可作为补充兼容性参考。昇腾移植优先研究 vLLM、vLLM Ascend、SGLang、Transformers 与 llama.cpp。

官方 HF checkpoint 标注 Apache-2.0；Maya 根目录 LICENSE 为 MIT，仓库中还存在其他第三方许可。约 20.2 GB 的官方权重不提交 Git，使用固定 HF revision 单独下载。复制上游代码时保留各自 SPDX、LICENSE 和 NOTICE。


### 官方模型卡列出的完整生态（补充核验）

模型卡还明确列出以下框架/服务入口，均值得纳入后续兼容性矩阵：

- [TokenSpeed](https://lightseek.org/tokenspeed/recipes/models#inkling-2-mini)：模型卡明确列为支持的推理框架，需单独核对其模型实现与昇腾支持范围。
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)：分布式 serving 编排；官方 recipe 以 SGLang 为执行引擎，适合参考多实例与调度部署。
- [Unsloth](https://unsloth.ai/docs/inkling-2-mini)：量化权重和 GGUF 生成/部署入口。
- [Ollama](https://ollama.com/library/inkling-2-mini) 与 [Exo Labs](https://github.com/exo-explore/exo)：Apple Silicon 推理路线，不是昇腾算子参考。
- [Miles](https://miles.radixark.com/docs/models/thinkingmachines/inkling-2-mini)、[SkyRL](https://github.com/SkyRL-Team/SkyRL)：后训练/强化学习生态，不是首期推理后端。
- [SGLang 官方 Mini cookbook](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini)：应与其上游源码一起作为 serving 对照。

官方模型卡与发布公告将 Hugging Face、SGLang、vLLM、TokenSpeed、NVIDIA Dynamo、Unsloth、llama.cpp、Llama App 及 Apple Silicon 上的 Ollama/Exo 列为部署选择。可参见 [官方模型卡](https://huggingface.co/thinkingmachines/Inkling-2-Mini) 与 [Thinking Machines 发布公告](https://thinkingmachines.ai/news/inkling-2-mini/)。这不是对每个设备/版本组合的验证承诺；本项目仍需单独资格验证。


## 8. 新增推理实现仓库（2026-11-24）

- [TokenSpeed](https://github.com/lightseekorg/tokenspeed)：已找到 Mini 专用 `runtime/models/inkling.py`、GFX950 Inkling2 monokernel 的 attention/conv/mix/experts/route/quant/head/layer/plan/dims，以及专用测试。其 GFX950 实现针对 AMD GPU，不是 Ascend Kernel；可借鉴算子融合、调度、分块与数值测试，不可直接在昇腾执行。
- [Unsloth](https://github.com/unslothai/unsloth)：包含 `unsloth/models/inkling2.py`、`inkling2_flex.py`、loader 适配。适合研究模型加载、FlexAttention、LoRA/训练适配和权重处理。注意模型相关源码文件带有 AGPL-3.0-only SPDX，和仓库其他 Apache-2.0 组件分开处理。
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)：包含 Inkling 2 Mini + SGLang 的 B200/VR200 聚合与 disaggregated Kubernetes recipes、模型缓存和下载 YAML。它是部署编排参考，不是新的模型算子实现。
- [Hugging Face Mini 专题](https://github.com/huggingface/blog/blob/6229fabe5f27f7934d825940853bcc1369528c2f/inkling2mini.md) 与 [原 Inkling 文章](https://github.com/huggingface/blog/blob/6229fabe5f27f7934d825940853bcc1369528c2f/thinkingmachines-inkling.md)：保留链接和来源信息，不复制整篇受版权保护的文章。

上述三套新代码仓库已经作为固定 commit 的 submodule 加入；模型专用文件快照及许可证说明见 `reference-snapshots/`。完整仓库体积较大，推荐先按需初始化 submodule，不必一次下载全部框架。

## 9. 模型专用源码快照与参考索引

本仓库新增 `reference-snapshots/tokenspeed/`、`reference-snapshots/unsloth/` 和 `reference-snapshots/dynamo/`。内容包括 TokenSpeed 的 Inkling runtime/GFX950 kernel/test，Unsloth 的 Inkling2/FlexAttention/loader，以及 Dynamo 的 Mini SGLang deployment manifests。许可证文本随快照保存；完整上游版本由 submodule 固定。Hugging Face 博客文章只保留链接，不复制全文。统一索引见 [references/articles/README.md](../references/articles/README.md)，快照说明见 [reference-snapshots/README.md](../reference-snapshots/README.md)。


## 10. Model Runner 选择约束（用户决策）

本项目只采用 Ascend Model Runner V1，明确不适配 MRV2。当前 vLLM Ascend 阅读快照 `1999fa...` 已不含 `worker/v1/model_runner.py`；历史 commit `f16e5e26801f86b071bca6454c9b9b2b63362e0a` 含 V1 runner 与 worker 选择逻辑。优先选定仍保留 V1 runner 的匹配 vLLM / Ascend 基线，再移植 Mini 及所需缓存接口；若采用较新基线，则在项目分支维护完整 V1 兼容执行链。不能将旧文件不经适配直接与当前 main 拼装。

本次复核的配置选择证据：

- [历史 `mrv2_utils.py`](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/mrv2_utils.py)：`VLLM_USE_V2_MODEL_RUNNER=0` 优先选择 V1。
- [当前 submodule 的 `mrv2_utils.py`](https://github.com/vllm-project/vllm-ascend/blob/79ee8d9bac80585816bacf6932222fc7cc30a00c/vllm_ascend/mrv2_utils.py)：选择器无条件返回 V2，并对请求 V1 的 `=0` 报错。该 submodule 只作为源码参考，不是 V1 部署基线。

新旧 runner 父类、cache API 与 worker 构造的逐项核验见 [runner-selection.md](runner-selection.md)。
