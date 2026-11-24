# 算子接口与数值契约

本页定义建议的 NPU 适配接口，尚未实现或上机验证。以下形状与计算顺序依据 vLLM Mini 合并快照 `ab18334a4f60da74db7de89fe95594e5ed0cda36` 核验（2026-11-24）。接口中的 `D` 是 hidden size，`d` 是 attention head dimension，`T` 是打包后的 token 数，`S` 是流数，`Q` 是 pseudo-query 数，`N` 是完整 residual block 数。

## 1. 通用约定

模型 baseline 使用 BF16 权重与主要激活；FP32 计算边界逐项保留。所有 `out=` 接口明确是否支持别名；默认不允许输出覆盖仍被本次计算读取的输入。支持 `T=0`，忽略 padding token 的 cache 写入。先定义逻辑连续布局，再由 backend 显式变换到 NPU 物理布局；不要复制 CUDA 专用 sheared bias 缓冲区布局作为数学要求。

| 接口（建议名称） | 输入 | 输出 | 核心约束 |
|---|---|---|---|
| `attn_res` | `stack[N,T,D]`、`last[T,D]`、可选 addend；FP32 queries `[Q,D]`；可选 norm `[Q,D]` | `[Q,T,D]` | 无 norm 返回 FP32；有 norm 返回 `last.dtype` |
| `pair_rms_norm` | FP32 dest/message `[S,T,D]` | BF16 `[S,T,2D]` | 分别归一化再拼接；允许 dest 流维 stride=0 |
| `controller_project` | BF16 pair `[S,T,2D]`、down `[S,r,2D]`、up `[S,2D,r]` | FP32 gain/forget `[S,T,2D]` | down 输出 BF16；第二次矩阵乘积输出 FP32 |
| `blend_rms_norm` | dest/message FP32、gain/forget FP32、norm `[S,D]` | BF16 `[S,T,D]` | FP32 混合后 RMSNorm；gain 在前 D，forget 在后 D |
| `short_conv` | `x[T,C]`、静态权重、可选动态 taps、request metadata、state view | BF16 `[T,C]` | 因果、按请求隔离、按 pass 独立状态 |
| `relative_paged_attention` | Q/K/V、相对偏置、页表、序列长度、窗口 | FP32 out `[T,Hq,d]`、自然 log LSE `[T,Hq]` | LSE 与实际参与 softmax 的 logits 完全一致 |
| `gate_sink` | out、gate `[T,Hq*d]`、sink/LSE `[T,Hq]` | BF16 `[T,Hq*d]` | 每元素 gate、每 head sink |
| `inkling2_route` | hidden `[T,D]`、gate weight `[E,D]`、bias `[E]`、top-k、route scale | expert ids `[T,k]`、FP32 weights `[T,k]` | bias 只影响选择；权重由所选原始 logits 计算 |
| `inkling2_experts_bf16` | routed tokens、专家 weights、routing weights | `[T,D]` 与归约状态 | SiTU、softcap 和权重位置必须一致 |

这些是工程接口契约，不能直接当作 `torch_npu` 或 CANN 现成 API 名称。具体调用名、支持 dtype、layout 与动态 shape 范围必须对安装版本核验。

## 2. AttnRes：沿 residual block 维做 softmax

令来源集合是 `stack` 中的 N 个完整 block 加上 `last + addend`。每个 token、每个来源分别计算：

$$
r_n=(\operatorname{mean}(x_n^2)+\epsilon)^{-1/2},\qquad
a_{q,n}=\langle q_q,x_n\rangle r_n,
\qquad p_{q,n}=\operatorname{softmax}_n(a_{q,n}).
$$

`normalize_values=False` 时，`mix = Σ p_n x_n`；为 true 时，`mix = Σ p_n (r_n x_n)`。可选输出 RMSNorm 在 mix 之后执行。`queries` 已在加载后乘过 `D**-0.5`，kernel 不得再次缩放。查询没有 token 维，不能误传成 `[Q,T,D]`。`last` 与 addend 的加法、评分、softmax 和混合保留 FP32 边界。

测试除常规随机输入外，应覆盖 N=0、query 多路、全零输入、大幅值、normalize_values 两种模式、norm 有/无、T=0、非连续 stack 首维，以及 `addend` 的存在与否。

## 3. 低秩 loop controller

四个流依次为 attention-query、K、V、MLP。每个流：

$$
p=\operatorname{BF16}([\operatorname{RMS}(dest),\operatorname{RMS}(message)]),
\quad h=\operatorname{BF16}(p W_{down}^{\mathsf T}),
\quad [g,f]=\operatorname{FP32}(h W_{up}^{\mathsf T}),
$$

$$
z=(1-\tanh(f))\,dest+\tanh(g)\,message,
\quad y=\operatorname{BF16}(\operatorname{RMSNorm}(z)).
$$

pair 的 RMS epsilon 是 `1e-6`；最后 RMSNorm 使用模型配置 epsilon。局部层在后续 loop pass 将初始 dest 广播给 q/k/v 三路后分别融合，因此不能把这三路继续绑成同一个张量。上游混合 kernel 关闭 FMA contraction；NPU 参考路径保留相同 FP32 计算次序和 BF16 舍入点，再由差分测试评估优化后的融合误差。不要把 FP32 输出 BMM 假定为所有 `torch.bmm(..., out_dtype=...)` 后端均支持；可用 FP32 fallback 建立 oracle，再选设备支持的实现。

## 4. Attention：标准 SDPA 不能直接覆盖全部语义

投影拆分顺序为 **q、r、gate、sink、k、v**。局部层第一 pass 可以共享 xq/xk/xv；全局层以及局部层后续 pass 必须支持不同输入。保留以下步骤：q/k/r L2 clipping、q/r scale、global log scaling、K/V short-conv、无 gamma 的 value normalization、相对位置 logits、causal / sliding-window mask、gate/sink。

attention 的缩放是 `d**-0.5`。相对偏置是每 query/head 生成的分量，不能随意改成 RoPE 或 ALiBi。标准 attention 算子只有在能表达精确偏置、窗口并返回兼容 LSE 时才可复用。否则实现模型专用 paged kernel。短序列参考可显式构造 logits 验证，长上下文生产路径应分块计算，避免物化 `[T,H,L]` 全矩阵。

上游的 sink 修正是：

$$
y_{t,h,d}=o_{t,h,d}\,\sigma(g_{t,h,d})\,\sigma(LSE_{t,h}-s_{t,h}).
$$

这里 LSE 是未加入 sink 的正常注意力 logits 的 `logsumexp`，不是概率和，也不是 base-2 log。若底层返回 base-2 LSE，需要显式换底；若 kernel 已把 sink 纳入分母，就不应再次乘该因子。prefill 与 decode 的 output/LSE 必须共享同一约定。

## 5. 短卷积与缓存输入

建议接口用 `stream_id`、`owner` 和 metadata 明确状态归属：

```python
short_conv(
    x, static_weight, *, dynamic_taps, positions,
    state, block_table, slot_mapping, seq_idx, query_start,
    stream_id, addend=None, residual=None, out=None,
)
```

局部 K/V 使用静态卷积；全局 K/V 的 taps 依赖输入投影，需支持 `[T,Hkv*W]` 动态项。attention-output 和 MLP-output 是另两条独立流。确认 kernel 保存的是正确的卷积输入状态，而不是随意将卷积结果回写。当前 forward 内的历史 taps 从不可变输入读取，跨 forward 的历史从缓存读取，防止线程执行次序改变结果。

## 6. Router 与 SiTU 专家

上游 router 先对 logits 取 sigmoid，再加 bias 选 top-k，平分时选择较小 expert id。routing weights 使用所选 **未加 bias** logits 的 sigmoid，在所选 k 个专家内归一化，再乘 route scale。极小概率用 log-sigmoid / logsumexp 实现以避免下溢。不能使用普通 softmax router 替代。

专家激活是带 softcap 的 SiTU：先从原始 gate 计算 sigmoid，再分别对 gate/up 应用 `cap*tanh(x/cap)`（cap≤0 时不变），执行 `softcap(gate)*sigmoid(original_gate)*softcap(up)*routing_weight`，并在 down projection 前按上游舍入到 BF16。上游代码把这条路径包装在名称为 `swiglu_quantize` 的函数中，但它不是可直接替换的普通 SwiGLU。量化激活再量化与纯 BF16 路径会产生差别，应分开建立参考。

固定 Mini 路径没有 shared expert（上游构造器断言 `n_shared_experts==0`）。第一版 BF16 专家要校验 gate/up 排列、专家维分片与 down 权重转置；适配通用 FusedMoE 时不能仅检查输入输出 shape。通信契约见 [集成方案](vllm-ascend-integration.md)。

## 源码依据

- [AttnRes](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/ops/attn_res.py)
- [pair / blend norm](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/ops/loop.py)
- [attention prologue 与 gate/sink](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/ops/attention.py)
- [router](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/ops/moe.py)
- [SiTU 与量化激活](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/ops/activation.py)
