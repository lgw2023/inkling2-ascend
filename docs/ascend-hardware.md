# 硬件边界与显存预算

状态：工程规划，尚未在本项目的 A2/A5 设备上实测。资料核验日期：2026-11-24。

## 1. 当前输入与决策

| 项目 | 已知条件 | 工程动作 |
|---|---|---|
| A2 | 用户提供：910B1，单卡 64 GB，单机 8 卡 | 作为首期文本正确性平台；完整模型从 TP2/TP4 开始 |
| A5 950PR | 单机 8 卡，单卡显存未确认 | 单独记录设备型号、总内存字节数、驱动/固件和软件构建 |
| A5 950DT | 单机 8 卡，单卡显存未确认 | 单独建立验证矩阵；不复用未经确认的 PR 二进制和性能结果 |
| 运行时 | 已有 CANN，版本未知 | 先采集，再选择完整兼容组合；不覆盖现有安装 |
| 引擎 | vLLM Ascend | 先 eager、BF16 计算路径、小上下文；图模式和量化后续独立验证 |

vLLM Ascend 官方快速入门已列出 A2 和 950PR/950DT 产品类别；这表明引擎存在相应硬件路线，**不等于 Inkling2 Mini 已获得支持**。当前安装说明对部分构建和镜像单独区分 950DT。每种卡的 Ops 包、编译目标及固件仍应按实际产品匹配。[官方快速入门](https://docs.vllm.ai/projects/ascend/en/main/getting_started/quick_start.html)、[官方安装说明](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html)

## 2. 64 GB 应如何预算

按 26.4B 参数、全部以 2 字节存储作粗估，静态权重约为：

```text
26.4 × 10^9 × 2 = 52.8 × 10^9 bytes ≈ 49.17 GiB
64 × 10^9 bytes ≈ 59.60 GiB
64 GiB = 68.72 × 10^9 bytes
```

这是数量级估计，实际模型参数与精度分布以 checkpoint 为准。用户标称“64 GB”不能直接当作运行时报告的 `64 GiB`。`--probe-npu` 会按可用 TorchNPU 接口返回 `memory_total_bytes`，无该接口或设备不可用时保留未知值。

工程预算至少分开记录：权重常驻、权重加载峰值、激活/临时 buffer、KV cache、卷积状态、通信 workspace、运行时开销。量化权重解码到 BF16 过程中，原始 packed 张量和解码后张量可能同时存在，应做分片流式转换和加载。controller 的 down/up 权重是 BF16，而 message、gain/forget 投影输出和混合计算保留 FP32；此外还有 FP32 pseudo-query、bias 及复制组件，不能用“总参数除 TP”替代实际统计。具体精度边界见 [算子契约](operator-design.md)。

每 rank 的简化预算为：

```text
W_rank = W_sharded / TP + W_replicated
M_peak = W_rank + KV + convolution_state + activation_workspace
         + communication_workspace + runtime + loading_temporary
```

因此本项目不以“49.17 GiB < 64 GB”承诺单卡完整推理。TP1 用于算子、局部层和缩小模型测试；完整真实模型优先 TP2，再比较 TP4/TP8。八卡不必始终组成一个实例，正确性稳定后可比较多个 TP2/TP4 副本的总吞吐。

## 3. 循环模型缓存预算

权重按物理层共享，缓存按**物理层与 loop pass 的组合**隔离。必须从实际 `config.json` 生成访问序列，再统计 full/local attention owner 数。若配置为 6 层 prelude、12 层 core 重复 3 次、6 层 coda，总访问数为 48；不能只按 24 个物理层估算缓存。

对标准 GQA 的 KV 主体，可用下式作初始估算：

```text
KV_bytes ≈ 2 × head_dim × local_kv_heads × element_bytes
           × Σ_cache_owners(retained_tokens_for_that_owner)
```

`local_kv_heads` 是 TP 分片后的数目；full attention 保留全上下文，local attention 的逻辑保留长度依窗口而定。还必须加上分页对齐、batch 需求、cache manager 分组和卷积缓存。分配器实际页尺寸可能因混合状态对齐而增大，不能把逻辑卷积窗口 4 直接当物理 block size。消息 bank 是 forward 内中间状态，不应再按整个历史序列重复计算一套 KV。

具体状态与测试见 [cache-and-loop.md](cache-and-loop.md)。1M 模型上下文能力也不能直接换算成八卡下可服务的上下文或并发数。

## 4. 设备采集清单

先运行：

```bash
bash scripts/collect_env.sh
bash scripts/collect_env.sh --probe-npu > /tmp/inkling2-env.json
```

默认只输出 Python/系统架构和包版本白名单；显式 `--probe-npu` 仅输出设备索引、型号、内存字节数。报告不包含 hostname、IP、序列号、账号、token 或完整环境变量。CANN Toolkit、对应芯片 Ops、NNAL、driver、firmware 版本需要从安装元数据人工补录；工具不猜版本。

在私有机器记录中补充卡间拓扑、NUMA 对应关系、主存余量和 HCCL 基础互联测试结果。公开仓库仅提交脱敏的测试摘要及必要版本信息。

## 5. A2 → A5 的迁移顺序

1. A2 上完成文本算子、cache、TP 与端到端正确性。
2. 在 PR 与 DT 各自匹配的软件栈中重跑相同输入和参照输出。
3. 分别测量 BF16 GEMM、路由/专家、relative attention、conv 和 HCCL 的占比。
4. 按实测热点开发 A5 特定融合/量化实现，保留 BF16 参照路径。

低精度硬件能力不说明 checkpoint 的 NVFP4/MXFP8 编码可直接映射到 CANN 算子。格式、scale 粒度、打包顺序、累加精度和舍入规则均需明确，见 [quantization.md](quantization.md)。
