# Loop 状态、KV 与短卷积缓存设计

状态：设计与源码阅读已完成，NPU 实现及缓存集成测试尚未执行。依据 [Mini loop 实现](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/loop.py) 和 [attention pass owner](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling2/nvidia/attention.py)，核验日期 2026-11-24。

## 1. 权重共享，状态按访问区分

固定 Mini 为 24 个物理层。prelude 为层 0–5，core 为层 6–17，每 token 执行三个 pass，coda 为层 18–23。因此访问数是：

$$
6 + 12 \times 3 + 6 = 48.
$$

| 部分 | 物理层数 | 每层执行次数 | attention KV owner 数 | conv state owner 数 |
|---|---:|---:|---:|---:|
| Prelude | 6 | 1 | 6 | 6 |
| Core | 12 | 3 | 36 | 36 |
| Coda | 6 | 1 | 6 | 6 |
| 合计 | 24 | — | 48 | 48 |

每个 conv owner 打包 K、V、attention-output、MLP-output 四条流。这里的 48 个 conv owner 不是额外 192 个独立 cache manager 层。核心层的三次访问使用同一组投影和专家参数，却必须拥有不同的 KV 和卷积历史。默认不得启用 research checkpoint 才允许的跨 pass KV 共享。

建议规范 cache key 的唯一性：`layers.<physical_id>.attn` 对应 pass0，额外 pass owner 使用 `.pass1` / `.pass2`；conv owner 同样带 pass 后缀。请求 id 和 token position 继续由 scheduler/metadata 表示。**pass 不应通过给 position 人为加偏移实现。**

## 2. 前向中的 AttnRes 与 message bank

三种状态有不同生命周期：

| 状态 | 用途 | 生命周期 |
|---|---|---|
| KV / conv cache | 跨 token、跨调度步保留历史 | 请求生命周期，由 cache manager 管理 |
| 当前 AttnRes stack / partial | 当前 forward 的 residual block 混合 | 当前 forward，包含批内 token |
| 前一 pass 的 message bank | 向下一 pass 提供四流 message | 当前 forward 的 pass 边界 |

loop driver 的关键步骤：

1. 运行 prelude 并保存其完整 block 数与 partial block 的快照。
2. 运行 core pass0。
3. 在进入 pass1 时复制前一 pass 的 message bank；恢复 prelude 的 block 数和 partial；从 prelude 的状态开始本 pass，将 bank 经 AttnRes 读出后送入四流 controller。
4. pass2 重复同一过程，使用 pass1 的 bank。
5. coda 从最后一次 core pass 的结果继续执行。

message bank 是当前 forward 的临时张量，不应错误估算为另一整套历史 token KV cache。复制 bank 的时机影响别名：不能让随后写入的新 pass stack 覆盖仍在读取的 bank。辅助 hidden states、路由记录与后续 speculative integration 的索引按 **visit** 定义，不应仅按物理 layer id 记录。

## 3. Conv state 的逻辑布局

上游短卷积 owner 复用 `AttentionLayerBase`，通过 `get_kv_cache_config()` 返回 `KVCacheLayerConfig`：`AttentionSpec(..., head_size_v=0)` 加 `SlidingWindowManager.layer_config(...)`。attention KV 与 conv state 都以 owner 形式进入缓存发现和绑定流程。

设每 rank KV heads 为 `Hlocal`，head dimension 为 d。Mini 路径使用 `replicate_hidden=True`，因此 attention-output / MLP-output 的完整 hidden 状态在每个 TP rank 上保留：

$$
D_{packed}=\operatorname{nextpow2}\left(2d+2\frac{D}{Hlocal}\right).
$$

上游逻辑 view 为 `[num_blocks, Hlocal, physical_block_tokens, Dpacked]`。四流分别占 K、V 和两段 hidden 子区间。NPU 可以采用适合自身算子的物理 view，但必须保留流范围、总字节数、页映射和 strides 的明确转换。

**卷积核窗口 W=4 不等于物理 cache block 必须为 4。** page-size unifier 可能放大 token 维，使 conv page 与 attention page 对齐。读取 cache 的 kernel 应使用绑定后的 `cache_block_size` 或显式 stride，不能把常数 4 写入页寻址。

来源：[InklingConvState（Mini 复用的公共组件）](https://github.com/vllm-project/vllm/blob/ab18334a4f60da74db7de89fe95594e5ed0cda36/vllm/models/inkling/nvidia/sconv_swa_attn.py)。该文件顶层对旧模型默认 hidden-sharded 的说明，不能覆盖构造器 `replicate_hidden=True` 分支。

## 4. Metadata 最小契约

| 字段 | 逻辑形状 / 类型 | 作用 |
|---|---|---|
| `block_table` | `[num_requests, max_blocks]`，整数 | 请求的逻辑块到物理块映射 |
| `slot_mapping` | `[padded_tokens]`，int64 | 当前 token 写入槽；`-1` 禁止写入 |
| `seq_idx` | `[padded_tokens]`，int32 | token 属于哪个请求 |
| `query_start` | `[padded_tokens]`，int32 | 本次 forward 中所属请求的起始 token 行 |
| `query_start_loc` | `[num_requests+1]`，整数 | 打包 batch 的 query 边界 |
| `seq_lens` | `[num_requests]`，整数 | 包括历史与当前 query 的长度 |

同一 forward 内的旧 taps 从不可变输入读取；早于本次 query 开始的 taps 才从历史 cache 读取。新请求起始不足窗口的部分按参考实现填零。padding 不写状态、不参与路由，也不污染 LSE 或输出。decode/prefill 混排时，所有 per-token metadata 必须采用同一 token 排序。

Ascend V2 runner 已有共享 backing 与 backend-specific view 机制。优先通过新 backend 接入；检查 `initialize_kv_cache()` 的物理 view、清零器和别名行为后，再决定是否修改 runner。不能仅因上游 GPU 无需改 manager 就断言 NPU 无需改动。阅读依据：[NPUModelRunner](https://github.com/vllm-project/vllm-ascend/blob/1999fa582aba279dc4214de7ea3035535917d4c4/vllm_ascend/worker/v2/model_runner.py)。

## 5. 内存估算规则

KV 容量需按展开后的访问数及 local/global 分组估算，而不是仅乘 24 层。对每请求，近似逻辑字节数为：

$$
M_{KV}\approx2\,b\,H_{local}\,d\left(n_{global}L+n_{local}\min(L,W_{swa})\right),
$$

其中 b 是每元素字节数，`n_global/n_local` 是展开后的 owner 数。若目标 config 确认为每六层五个 local、一个 global，则展开后为 40 local / 8 global。实际分配还要加上整页取整、批内请求、conv state、prefix 保留、metadata 和 workspace。不要把 SWA 的窗口上界机械套到 global 层。

`replicate_hidden=True` 的 conv state 与临时 activation 不会按 TP 完全等比分摊。权重加载临时缓冲与解量化峰值也不属于 KV 公式；以 runner 完成加载和 warmup 后可用显存测定服务参数。A2 单卡 64 GB 不直接等于可给 KV 的 64 GiB。

## 6. 缓存验收断言

- 48 个 attention owner 与 48 个 conv owner 均被发现、分配、绑定；每个核心 pass 独立，物理层权重仍共享。
- 同一请求按一次 prefill、不同 chunk 切分、逐 token 输入，状态和 logits 在规定误差内一致。
- 两请求交错执行与分别执行一致；请求结束后物理块复用无旧状态泄漏。
- 跨 `W-1/W/W+1`、物理页边界、SWA 边界的输出一致；打乱物理 block id 不改变逻辑结果。
- prefix caching 初期关闭。启用前必须同时验证 KV、conv 历史、pass owner 与 chunk 边界恢复；只验证普通 attention KV 命中不够。
- 图捕获前先验证 padded metadata、固定 buffer 地址和 alias 行为；eager 通过不自动代表 ACL Graph 通过。
