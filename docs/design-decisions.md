# 设计决策与易错点

所有条目是核验后形成的实施约束，原始依据见 [sources.md](sources.md)。

| 决策 | 原因与工程影响 |
|---|---|
| 以 24 个物理层、48 次访问建模 | core `[6,18)` 重复三次，访问计划为 `6 + 12×3 + 6`；权重共享不能推导为状态共享 |
| 显式使用 head_dim=128 | hidden_size=2304 与 Q heads=24 不要求相除得到投影头维度 |
| BF16 首期是一条新实现路线 | 上游专家 kernel / loader 按 NVFP4、MXFP8 组织；设置 dtype 不会自动得到 BF16 专家模型 |
| 每个 pass 的 KV / conv 状态独立 | 同一 token 在不同循环中的 K/V 与卷积输入可以不同，不能直接覆盖上一 pass |
| 模型消息 bank 与历史 KV 分开预算 | message bank 是当前 forward 的块残差混合状态，不应当作另一份全历史 KV |
| 保留混合 FP32 / BF16 舍入点 | query / logits / controller 等 FP32 语义影响循环数值；不得 blanket cast |
| `TP × EP = 卡数` 不是通用配置公式 | 上游 Inkling2 专家分片和 Ascend MoE 通信组必须逐实现核对；只按真实 group world size 规划 |
| 优先使用 backend / spec 接口扩展 | 是否需改 runner、缓存分配器须由页面尺寸、缓存布局与生命周期集成测试决定 |
| NVFP4 与 MXFP4 分开处理 | 编码、scale 与 block 格式不同，设备支持某种 4-bit 算术不表示兼容 checkpoint |
| 未上机即标记未执行 | 不预填吞吐、显存、加速比或“完成”的里程碑 |

固定模型、源码、软件环境和基线输入后，再逐项启用优化；一个优化对应一组可复现的正确性与性能记录。
