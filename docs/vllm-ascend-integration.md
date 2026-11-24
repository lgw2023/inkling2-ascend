# vLLM Ascend 文件级适配方案

本页是待实施设计；本仓库尚未提供可加载完整模型的 NPU 实现。源码核验日期：2026-11-24。vLLM 阅读快照为 [`ab18334a4f60da74db7de89fe95594e5ed0cda36`](https://github.com/vllm-project/vllm/commit/ab18334a4f60da74db7de89fe95594e5ed0cda36)，vLLM Ascend 阅读快照为 [`1999fa582aba279dc4214de7ea3035535917d4c4`](https://github.com/vllm-project/vllm-ascend/commit/1999fa582aba279dc4214de7ea3035535917d4c4)。这两个 SHA 固定阅读依据，不代表已经验证兼容的安装组合。

## 1. 适配边界与开发顺序

**执行器决策（用户明确要求）：只适配 Ascend Model Runner V1，禁止采用 Model Runner V2（MRV2）。** 这不是只把配置开关设为 false：当前 vLLM Ascend main 阅读快照已采用 `worker/v2/model_runner.py`，不再包含旧的 `worker/v1/model_runner.py`。因此实施前必须选定与 vLLM V1 engine API 兼容的版本组合，并将历史 V1 runner 回移/适配到项目分支；不得把当前 main 的 V2 runner 当作实现基底。

历史参考：vLLM Ascend commit [`f16e5e26801f86b071bca6454c9b9b2b63362e0a`](https://github.com/vllm-project/vllm-ascend/commit/f16e5e26801f86b071bca6454c9b9b2b63362e0a) 中，V1 runner 位于 `vllm_ascend/worker/v1/model_runner.py`，`worker.py` 在 `use_v2_model_runner=False` 时实例化该类。该提交只作为 V1 代码结构参考，不能视为与当前 vLLM Mini 实现已经兼容。

首期范围为 A2 910B1、纯文本、固定三次循环、BF16 计算基线、eager 执行、单机 TP2/TP4，随后 TP8。图像、ACL Graph、低精度 NPU 原生专家、DSpark/MTP、LoRA 和服务侧高级缓存功能分别验收。A5 用同一语义基线建立独立的设备与软件验证记录。

上游 Mini PR [#69028](https://github.com/vllm-project/vllm/pull/69028) 已合并并提供 NVIDIA 路径。Ascend [#20831](https://github.com/vllm-project/vllm-ascend/pull/20831) 是旧 Inkling/MTP 的草稿，只包含初始辅助函数且尚未接入运行时；它不能作为 Mini 已获 NPU 支持的依据。

建议在 vLLM Ascend 分支新增专用模型实现，复用上游配置、处理器和经过确认的平台无关逻辑。避免导入 `nvidia/model.py` 后才替换 CUDA 函数：该导入链涉及专用 kernel、workspace 和 GPU 能力检测。先做模块导入测试，确认 CPU/NPU 环境无需 CUDA 扩展即可完成模型注册和配置解析。

## 2. 仓库文件级清单

下表的 `新增` 路径是**拟在 vLLM Ascend 中创建**的文件，不表示本工程指南已经包含对应运行时代码。现有路径均以阅读快照为准。

| 优先级 | 路径（相对 vLLM Ascend 根目录） | 操作与职责 | 完成判据 |
|---|---|---|---|
| P0 | `vllm_ascend/models/__init__.py` | 修改现有 `register_model()`，将 `Inkling2LoopForConditionalGeneration` 注册到 Ascend 类 | 配置 architecture 精确命中；纯导入不触发 CUDA |
| P0 | `vllm_ascend/models/inkling2/__init__.py` | 新增最小导出；保持懒加载 | NPU worker 可解析类 |
| P0 | `vllm_ascend/models/inkling2/model.py` | 新增 embedding、24 个物理层、输出归一化、LM head、文本前向和严格 loader 入口 | 小模型完整前向；正确处理 padded vocabulary 和 logits softcap |
| P0 | `vllm_ascend/models/inkling2/loop.py` | 新增 visit 列表、prelude 状态恢复、前一 pass message bank、四流 controller；参数按物理层共享 | 48 次访问；核心层参数无三份复制；每个 pass 状态独立 |
| P0 | `vllm_ascend/models/inkling2/attention.py` | 新增 q/r/gate/sink/k/v 投影、相对位置偏置、L2 clipping、value norm、pass cache owner | local/full 层与参考逐项对齐 |
| P0 | `vllm_ascend/models/inkling2/sconv.py` | 新增四流 short-conv 包装、动态 K/V taps、conv 后残差更新 | chunked prefill 与逐 token decode 等价 |
| P0 | `vllm_ascend/models/inkling2/moe.py` | 新增 BF16 专家、Inkling2 router、SiTU 激活与加权位置；明确输出是否已归约 | 稳定路由、专家输出和 TP 归约对齐 |
| P0 | `vllm_ascend/models/inkling2/weight_loader.py` | 新增源权重到 NPU 权重的名称、分片、dtype、量化元数据契约；严格报 missing/unexpected | 每个目标参数恰好加载；controller 视图仍共享底层存储 |
| P0 | `vllm_ascend/attention/inkling2.py` | 新增 NPU attention backend / metadata builder；定义物理 KV view 和 layout | 所有 pass 的元数据、block table、cache view 绑定正确 |
| P0 | `vllm_ascend/attention/inkling2_sconv.py` | 新增 conv state backend / metadata builder；接入短窗口管理 | 窗口回收、padding、请求结束及块复用正确 |
| P0 | `vllm_ascend/ops/inkling2/reference.py` | 新增 torch 参考：AttnRes、pair/blend norm、SiTU、短卷积、短序列 attention | 小规模可作为 NPU 单元测试 oracle |
| P1 | `vllm_ascend/ops/inkling2/attn_res.py` | 新增优化实现及分派 | 数值通过后分别测 prefill/decode 延迟 |
| P1 | `vllm_ascend/ops/inkling2/loop.py` | 新增 pair norm、BMM、FP32 blend、RMSNorm | 保留指定舍入点，无错误 stream 广播 |
| P1 | `vllm_ascend/ops/inkling2/attention.py` | 新增相对偏置 paged attention，返回输出与 LSE；可拆 prologue / gate | SWA/full、跨块、混合 batch 通过 |
| P1 | `vllm_ascend/ops/inkling2/sconv.py` | 新增静态/动态 conv kernel 及 cache 更新 | 非连续布局和块边界通过 |
| P0 | `vllm_ascend/worker/v1/model_runner.py` | **目标执行器**：从历史 V1 runner 回移并适配当前选定 vLLM V1 API；负责 `get_kv_cache_spec()`、`initialize_kv_cache()`、layer binding、forward metadata 和 `execute_model()` | 纯文本请求全程只经过 V1 runner；不导入/实例化 MRV2 |
| P0 | `vllm_ascend/worker/worker.py` | 选择并实例化 V1 `NPUModelRunner`；启动时检查 `use_v2_model_runner=False`，拒绝误入 V2 | 启动日志明确报告 V1；测试确认 V2 构造器未被调用 |
| 按需 | `vllm_ascend/worker/v1/kv_block_zeroer.py` | 只在 V1 Cache 生命周期确有需要时接入块清零；遵循 V1 的 block reuse 语义 | 请求回收、块复用无状态残留 |
| 按需 | `vllm_ascend/ops/fused_moe/fused_moe.py` | 优先使用适配层；确认通用接口无法表达 SiTU/路由后再扩展 | 输出归约语义明确，其他模型回归通过 |
| P2 | `vllm_ascend/models/inkling2/vision.py` | 新增图像 encoder、projector 的 NPU 运算替换；复用经确认的 processor 契约 | 文图 token 拼接、尺寸与图像样本对齐 |
| P0/P1 | `tests/ut/models/test_inkling2_*.py`、`tests/e2e/inkling2/` | 拟新增单元、缓存集成、分布式与服务测试 | 按 [验收计划](benchmark-plan.md) 分层放行 |

若采用 Ascend C 自定义算子，应按所选 vLLM Ascend 版本的实际算子构建目录和注册机制新增工程，不预先假设上表 Python 文件就是二进制注册入口。先以 torch 参考实现和设备端组合算子打通，再按 profiler 数据替换热点。

## 3. 模型注册与权重加载

建议注册映射如下；这是修改规格，不是已经提交的运行时代码：

```python
ModelRegistry.register_model(
    "Inkling2LoopForConditionalGeneration",
    "vllm_ascend.models.inkling2.model:AscendInkling2LoopForConditionalGeneration",
)
```

加载器需要覆盖以下规则：

1. 读取 checkpoint 的 `text_config`、`loop_config` 和逐 tensor 的量化声明；保留固定循环约束，拒绝把研究 checkpoint 的可变 loop 开关套到 Mini。
2. 复用经核验的名称映射：embedding / LM head、q/r 缩放、相对投影、q/r/gate/sink/k/v 融合分片及多输入 norm。融合顺序必须与投影切片一致，不能按传统 q/k/v 三段拼接。
3. 把量化专家权重离线解码或逐 shard 流式解码到 BF16，生成转换清单；保留需要 FP32 的 pseudo-query、bias、数值中间量和量化辅助参数。转换说明见 [quantization.md](quantization.md)。
4. 原始 NVIDIA loader 对 BF16 专家会抛出异常，不能只修改 `--dtype bfloat16`。需要同时替换专家模块、参数 shape/dtype 和 loader。
5. 每个 controller 的 Parameter 与堆叠 buffer 需要保持别名。检查地址/存储一致性及参数计数；不把核心层为每个 pass 复制一份模型权重。

## 4. 并行与通信：先保持可解释的基线

上游 `Inkling2Experts` 的专家分片直接使用 TP rank / world size：每个 rank 持有完整专家的一个子集，输入 token 复制，专家局部输出再合并。**这是模型自身的映射，不能直接等同于通用 vLLM 的 EP CLI 开关。** 第一条 BF16 路径可复刻该映射，方便对照上游；后续接入 Ascend 通用 FusedMoE / EP 时必须重新核对专家分片、token dispatch、combine 和 reduction 契约。

| 输出边界 | 第一版契约 | 调用者动作 |
|---|---|---|
| attention `wo` | 每 TP rank 的局部 BF16 `[T,D]` | 一次 HCCL sum，再进入 attention-output conv |
| 自定义 BF16 experts | 局部专家合并后 `[T,D]`，或 wrapper 明确标记已完成 sum | 恰好归约一次，再进入 MLP-output conv |
| Ascend FusedMoE | 是否已归约取决于实际通信路径 | 尊重 `_fused_output_is_reduced()`；不能无条件再 all-reduce |

当前 Ascend `AscendMoERunner` 对 ALLTOALL / MC2 / FUSED_MC2 等路径标明已包含归约。这是适配时必须检查的边界。初版不搬运 CUDA `allreduce_conv_mix`；将 HCCL、conv、AttnRes 拆开验证后再评估融合。PP=1；loop 模型不先启用 sequence parallel。TP2/4/8 均要求 Q/KV heads 可整除，且检查每 rank 缓存宽度。

## 5. 提交拆分建议

- **提交 A：配置与导入。** 注册、纯文本路径、支持矩阵、未知版本显式报错。
- **提交 B：参考算子与 loader。** 可审查的权重转换、算子 oracle、合成小模型前向。
- **提交 C：loop 与缓存。** 48 个 visit、每 pass cache owner、chunk/prefix/回收测试。
- **提交 D：A2 eager 完整模型。** TP2/4 精度和服务烟测；TP8 作为后续容量与吞吐对比。
- **提交 E：优化。** 一次只引入一种 kernel、通信或图优化，保持可切回 reference 的路径。
- **提交 F：图像。** vision / processor / token 拼接的独立验收。

## 源码依据

- [模型与 loader](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/model.py)
- [循环驱动](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/loop.py)
- [专家与路由包装](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/moe.py)
- [Ascend 模型注册](https://github.com/vllm-project/vllm-ascend/blob/1999fa582aba279dc4214de7ea3035535917d4c4/vllm_ascend/models/__init__.py)
- [历史 Ascend V1 runner](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/v1/model_runner.py)
- [历史 Ascend worker 的 V1/V2 选择逻辑](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/worker.py)
- [Ascend MoE runner](https://github.com/vllm-project/vllm-ascend/blob/1999fa582aba279dc4214de7ea3035535917d4c4/vllm_ascend/ops/fused_moe/fused_moe.py)
