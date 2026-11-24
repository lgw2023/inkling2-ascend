# 量化、权重转换与 BF16 基线

状态：工程方案；本仓尚未实现转换器或 Ascend MoE。核验日期：2026-11-24。

## 1. 发布格式与首期选择

Inkling 2 Mini 发布权重混用 NVFP4、MXFP8、BF16 和 FP32。`config.json` 的 `torch_dtype=bfloat16` 不能代表全部 expert 权重已是 BF16；上游 [`quant_config.py`](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/quant_config.py) 读取 `hf_quant_config.json` 的逐 tensor 精度信息，再选择 expert 格式。

本仓建议先建立 **反量化权重 + Ascend BF16 expert 计算** 的正确性基线，同时保留需要 FP32 的参数和中间值。该路线降低首期低比特 kernel 工作量，代价是权重和中间激活占用增加；它不会恢复量化前的原始训练权重，也不保证与发布时的量化激活计算逐 bit 相同。

## 2. 已核验的 metadata

以下是上传 revision `1fb4134b9a74f9681054f4ef82e4bf1f458a2337` 所对应 metadata 的实例；转换器应解析 tensor 列表，不能把层号范围写成永久规则。

| tensor 组 | 发布 compute dtype | metadata 要点 |
|---|---|---|
| decoder 物理层 0–19 的 `experts.w13_weight / w2_weight` | NVFP4 | weight block `[1,16]`；packed uint8；含权重 scale 规则 |
| decoder 物理层 20–23 的上述 expert 张量 | MXFP8 | weight block `[32,32]`；E4M3；activation block 32、UE8M0 scale |
| embedding、attention/conv、controller 权重、vision 等 | BF16 | 不应被误当作低比特 expert 解包 |
| AttnRes 查询、跨轮查询、gate bias 等 | FP32 | 按各自 tensor 元数据保留 |
| unembedding | BF16 权重，FP32 forward output | 区分权重 dtype 与输出 dtype |

[量化 metadata](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/hf_quant_config.json)；[初始上传提交](https://huggingface.co/thinkingmachines/Inkling-2-Mini/commit/1fb4134b9a74f9681054f4ef82e4bf1f458a2337)。

NVFP4 与 MXFP4 不是可互换的文件格式。即使目标 NPU 支持某种 4-bit 或 8-bit 矩阵计算，也需核对编码、分组尺度、outer scale、布局及激活量化规则；不能把权重 reinterpret 成另一个格式就调用。

## 3. BF16 路线必须改的代码

上游 [`nvidia/model.py::load_weights`](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/model.py) 对不匹配的 BF16 expert 权重有显式报错；[`nvidia/moe.py`](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/moe.py) 按 NVFP4/MXFP8 构建参数和硬件执行路径。因此，转换 checkpoint 和设置 `--dtype bfloat16` 之后，还必须完成以下实现。

| 组件 | 建议落点（待新增） | 必需行为 |
|---|---|---|
| metadata 解析 | `vllm_ascend/models/inkling2/weight_loader.py` | 逐 tensor 建立 dtype、shape、scale、变换规则表；拒绝未识别格式 |
| 离线转换 | `tools/convert_inkling2_weights.py` | 按 expert/分片流式解包、反量化、校验、写 safetensors |
| BF16 expert 模块 | `vllm_ascend/models/inkling2/moe.py` | BF16 GEMM + 模型专用激活和路由；保留 FP32 累加边界 |
| 模型构造 | `vllm_ascend/models/inkling2/model.py` | 明确选择 Ascend BF16 实现，不实例化 NVIDIA quant expert |
| loader 接线 | 同上及 `weight_loader.py` | 支持转换后的格式版本、张量命名和切分；严格 missing/unexpected 检查 |
| 格式 manifest | 转换输出目录中的 `conversion_manifest.json` | 固定来源 revision、文件 hash、转换器 commit、输出 dtype 与误差摘要 |

模型注册入口仍保留上游 architecture 身份，由 Ascend 分派到对应实现；是否扩展已有量化选择钩子应按固定 vLLM/vLLM Ascend 版本确定。不要通过删除元数据来绕开 loader 的精度校验。

## 4. 转换流水线及验收

1. **锁输入。** 下载模型的固定 revision，保存 `config.json`、`hf_quant_config.json`、safetensors index 与全部 shard 的校验值；先读取张量 header 做清单。
2. **解析布局。** 分清源 checkpoint 的逻辑权重布局与 GPU kernel 的运行时 swizzle；后者不应盲目照搬进 Ascend checkpoint。
3. **反量化。** NVFP4 解包后结合块尺度和实际 outer scale/denominator；MXFP8 解码 E4M3 与尺度。尺度方向、broadcast 轴、transpose 顺序由 metadata 与上游加载路径逐项验证。
4. **分片落盘。** 以一个 expert 或受控 chunk 为工作单元，避免全量量化权重、全量 BF16 权重和中间 FP32 副本同时驻留。转换后的文件与原始模型分开存放。
5. **核验。** 张量总数/shape/dtype/有限值/缺失项先过关，再比较解码后的权重、单个 expert、MoE 层和整模输出。
6. **记录差异。** 对「仅权重反量化」和「去掉激活量化」造成的误差分别报告，避免把算法变化归因于 NPU kernel。

本仓暂不提供猜测 scale 公式的转换脚本。正确实现需读取实际 safetensors 的 scale tensors，并与发布方或上游可运行参考逐 tensor 对齐。

## 5. MoE 数值与并行契约

BF16 expert 路径不能简单替换成默认 SwiGLU。发布配置含 gate softcap=4、up softcap=25、route scale=8 和 gate bias；路由选择、权重使用位置、激活变换和 combine 的顺序均须沿上游实现建立 oracle。expert 选择至少对齐 top-8 索引集合及权重，再比较层输出。

上游 fallback 的 MoE 返回值已经完成对应分组归约；Ascend fused MoE 的不同通信模式也可能返回已归约输出。新接口要显式声明 `output_is_reduced`，让外层 loop/conv 混合路径只归约一次。不要按模块名字猜测通信是否发生。

首期优先 TP2/TP4/TP8、EP 关闭的可解释基线。之后接入 EP 时，以固定版本真实建立的 TP/DP/EP 分组为准；不能把「TP4 × EP2」当作独立的八卡维度配置直接填写。设备数、专家分布、每个 collective 的 group 必须能在日志里核对。

## 6. 容量与后续优化

总参数按两字节估算约 49.2 GiB，未扣除纯文本时不加载的视觉/草稿部分，也未计入 FP32 参数、缓存、通信和运行时。准确预算应由转换后的实际 tensor 清单汇总。A2 单卡 64 GB 上，先做算子和 tiny 模型；完整模型从 TP2/TP4 起步更适合建立工程基线。

低精度优化按「BF16 层输出基线 → 单 expert 量化 kernel → 路由/激活/scale 联调 → 整层精度 → 整模质量和性能」推进。A5 950PR 与 950DT 分别记录芯片识别、显存、驱动/CANN/torch_npu 配套和实际算子覆盖；在设备证据齐全之前不宣称 NVFP4/MXFP8 原生直跑。

## 7. 最小质量报告

每个转换器或量化实现版本至少留下：来源 revision、目标格式版本、逐 tensor 清单、几个覆盖低/高尺度及边界值的权重样本误差、expert 输出误差、路由索引一致率、48 次 layer visit 的隐藏状态误差曲线、短文本 greedy 输出与固定任务集指标。阈值由同源参考在目标 dtype 下的误差基线确定，并在报告中固定；不要仅凭一段可读文本认定转换正确。
