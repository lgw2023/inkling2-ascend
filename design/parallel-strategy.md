# 单机八卡并行策略

状态：建议方案；没有上机性能结果。

## 首期配置

首期采用 PP=1、DP=1，不启用通用 vLLM 的 `--enable-expert-parallel` 开关，TP 从 2 开始，随后 4、8。**这不表示专家权重不分片：**上游 Inkling2 的专用实现直接使用 TP rank / world size 划分完整专家，第一版 BF16 路径可复刻这一映射。根据模型 attention 的 Q/KV head 划分约束检查每个 TP，而不是只看 hidden size 是否整除。当前设计以模型配置中 Q=24、KV=8、head_dim=128 为核验项，若目标 checkpoint 不同须重新验证。

| 配置 | 主要目的 | 验收重点 |
|---|---|---|
| TP1，缩小模型或局部层 | 建立单设备参照 | 算子语义、状态更新、dtype |
| TP2，完整模型 | 首个文本正确性基线 | 权重切分、HCCL、峰值内存 |
| TP4，完整模型 | 扩大内存余量并测吞吐 | TP2/TP4 logits 一致性、通信占比 |
| TP8，完整模型 | 八卡单实例对照 | 收益是否被同步开销抵消 |
| 4 个 TP2 或 2 个 TP4 实例 | 整机服务吞吐对照 | 每实例设备隔离、并发与延迟分布 |

这些是候选实验，没有预设 TP8 一定最快。实例级吞吐和整机吞吐分开报告；保持 token 数、并发、采样参数与输出长度可比。

## 不把 TP 与 EP 当独立设备乘数

vLLM 的 EP 通信组由所选版本的并行配置决定。2026-11-24 核验的官方 EP 部署文档中，标准单机配置按 `EP_SIZE = TP_SIZE × DP_SIZE` 自动计算。EP 不是再额外乘上的一组设备，不能仅凭“TP4 × EP2 = 8 卡”构造配置。这个通用配置公式也不能替代 Inkling2 专用实现中“专家范围直接映射到 TP 组”的代码契约。首期不启用通用 EP 调度、EPLB、sequence/context parallel 或 prefill/decode 分离，待 BF16 MoE 路径正确后再核实实际 group membership。

打开 EP 前应记录：每 rank 的物理设备、TP rank、DP rank、EP rank、专家分片范围、路由 token ownership、dispatch/combine 通信方式、输出是 partial 还是 already-reduced。把这个表与固定版本提交一起保存。官方概念入口：[vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html)。接口行为以最终采用的源代码提交为准。

## 每个张量的并行契约

| 数据/模块 | 建议首期划分 | 必须检查 |
|---|---|---|
| Q/R/gate/sink 和 K/V projections | 与上游 head 划分一致 | checkpoint packed 顺序与本地 head offset |
| K/V convolution 和 KV cache | 按 KV head 划分 | 每个 pass 独立 owner，不能串层/串请求 |
| attention 输出投影 | row parallel partial | 在正确位置 HCCL sum，且只做一次 |
| AttnRes/message controller | 起步保留 full hidden 语义 | 源张量是 partial 还是已聚合；FP32 计算边界 |
| attention/MLP 输出卷积 | 按上游 replicate_hidden 语义 | 不能把 full-hidden 状态误切分成 TP-local |
| MoE | 第一版可按 TP rank 持有完整专家子集，并复制输入 token | sigmoid+bias 选择、SiTU、down 前 routing weight、局部专家合并及恰好一次归约 |
| embedding / LM head | 复用 vLLM 并行层并核验加载 | vocab padding、logits gathering、softcap |

Ascend MoE runner 的某些通信方法已经返回聚合结果。通用层外再做 HCCL sum 会产生倍乘错误；接口须明确 `output_is_reduced`，并测试不同 TP 大小，不根据类名猜行为。参考实现入口：[Ascend fused_moe.py](https://github.com/vllm-project/vllm-ascend/blob/1999fa582aba279dc4214de7ea3035535917d4c4/vllm_ascend/ops/fused_moe/fused_moe.py)、[Inkling2 专家分片](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/moe.py)。

## 后续优化顺序

先分离测量 attention、MoE、loop controller、conv、HCCL 和 CPU 调度占比；按实际瓶颈排序。图模式、prefill chunking、prefix cache、EP、量化和 speculative decoding 分别作为独立变更验收。图像阶段同时计入 vision encoder、token 扩展和 processor 的内存/并行策略，不能沿用文本阶段的 batch 上限。
