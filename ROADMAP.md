# 移植路线图

状态日期：2026-11-24。以下开发阶段均未宣称完成，时间由实际人力、源码兼容性与设备可用性决定。

## 里程碑与验收

| 阶段 | 可交付结果 | 验收条件 | 前置依赖 |
|---|---|---|---|
| M0 环境与权重锁定 | 软件锁、模型 revision、tensor manifest、A2 / A5 硬件矩阵 | 版本可复现，权重数量、形状、dtype、量化 metadata 对齐 | 无 |
| M1 CPU / 框架语义基线 | AttnRes、controller、conv、attention、MoE 参考路径 | 单算子与上游固定输入比对；记录 FP32/BF16 舍入点 | M0 |
| M2 A2 单层与循环核心 | NPU 算子组合、独立 pass 状态、6/12/6 访问计划 | prefill / decode / chunk 分割等价，4 流 conv 正确 | M1 |
| M3 A2 纯文本完整模型 | 模型注册、BF16 专家加载、TP2 / TP4 服务 | logits 与贪心生成基线通过；取消、回收、混批无状态串扰 | M2 |
| M4 A2 八卡优化 | TP8 / 多实例实测、算子融合、图模式候选 | 正确性不回退；性能相对 eager 基线有测量收益 | M3 |
| M5 A5 移植与优化 | 分别针对 950PR / 950DT 的环境锁和结果 | 各自运行相同测试集；验证设备支持的低精度与通信路径 | M3 + A5 信息 |
| M6 图像 | processor、vision、projector / 特征接入、MM 调度 | 文本回归通过；单图、多图、不同分辨率数值与服务测试 | M3 |

## 任务拆分

- [ ] **ENV-01** 采集 CANN / NNAL、驱动、固件、torch / torch_npu、vLLM / vLLM Ascend 的实际版本，记录 A5 显存与 SKU。
- [ ] **SRC-01** 以来源文档中的源码快照建立工作分支，验证该组合接口；补齐依赖锁，不直接把两个 SHA 称为兼容组合。
- [ ] **LOAD-01** 读取模型索引和 `hf_quant_config.json`，导出逐 tensor 的格式、shape、scale、block shape 与目标 dtype manifest。
- [ ] **LOAD-02** 流式解量化与分片加载 BF16 专家，保留 FP32 参数，验证缺失/额外键和重复加载。
- [ ] **MODEL-01** 在 vLLM Ascend 注册 `Inkling2LoopForConditionalGeneration` 的 Ascend 实现；不从 CUDA 模块直接导入运行路径。
- [ ] **LOOP-01** 实现显式访问计划、共享参数引用、每 pass 缓存 owner、prelude 快照与消息 bank 生命周期。
- [ ] **OP-01** 实现 AttnRes 与 loop controller 组合路径，先正确，再融合。
- [ ] **OP-02** 实现静态 / 动态短卷积与 4 流状态读写，覆盖 page 边界和混批。
- [ ] **ATTN-01** 实现 relative GQA、clip、norm、sink / gate 与 LSE 的一致性路径。
- [ ] **MOE-01** 实现 BF16 expert 与路由契约，明确专家分片、通信和输出是否已经 reduce。
- [ ] **CACHE-01** 接入 backend spec / metadata、缓存视图和 slot mapping；以集成测试决定是否需要 runner / manager 扩展。
- [ ] **TEXT-01** eager、无 speculative、无 prefix caching、关闭 image，小上下文服务跑通。
- [ ] **TEST-01** 执行数值 / 缓存 / 服务测试，保存机器可读报告，填写 qualification。
- [ ] **PERF-01** 测量 batch、context、TP 与多实例矩阵，选择可解释的优化目标。
- [ ] **A5-01** 分别验证 950PR 与 950DT，不把一种板卡结果替代另一种。
- [ ] **MM-01** 按 [图像计划](docs/multimodal.md) 实现第二阶段，保持文本回归。

## 本仓库现有材料

文档、任务模板、配置规划工具、环境检查和受控启动模板已经提供。NPU 模型实现、权重转换器、算子内核和性能报告应随开发逐项新增，不以占位代码表示完成。
