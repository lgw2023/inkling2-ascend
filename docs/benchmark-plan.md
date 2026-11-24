# 测试、验收与性能计划

截至 2026-11-24，本项目没有执行 Ascend 上机测试，也没有实测吞吐、时延或质量成绩。本页是实施与验收计划；上游 NVIDIA 测试只能作为参考设计，不能作为 A2 / A5 通过记录。

## 1. 建立可复现的基线

每次实验保存：模型仓库 revision、配置与 tokenizer hash、权重转换清单、vLLM/Ascend commit、CANN/driver/firmware/torch/torch_npu 版本、卡型号与显存、TP/EP/DP 实际分组、启动参数、输入 token ids、随机种子及输出。公开结果使用可分享的合成文本或公开评测集，不上传真实业务请求或完整环境变量。

比较分两条路径：

- **数值移植基线：**同一份解码后的 BF16 专家权重、相同算子舍入点，用 torch/已知正确参考与 NPU 结果做逐层比较。
- **原发布模型基线：**原始低精度 NVIDIA 路径，用于衡量 BF16 解码方案与原量化执行路径之间的差别。量化激活是否保留必须写入报告，不能把差别全部归因于 NPU。

优先在小规模合成模型上排查语义，再运行完整权重。完整 TP1 是否能装入 A2 由显存实测决定；单算子和小模型 TP1 不依赖完整模型单卡装入。

## 2. 分层测试矩阵

| 层级 | 测试内容 | 必测形状 / 场景 | 放行条件 |
|---|---|---|---|
| T0 配置/加载 | architecture、visit、分片、转换清单、missing/unexpected、dtype | 固定 Mini；无图像；TP1/2/4/8 的合法性；错误配置 | 错误明确失败；48 visits；参数不重复；加载覆盖完整 |
| T1 算子 | AttnRes、pair/blend norm、controller、SiTU、router、静态/动态 conv | T=0/1/2/7/32/257；真实 D=2304；stream 广播；零与极值 | 对 torch oracle；无 NaN/Inf；逐组件误差达标 |
| T1 Attention | Q/K/V clipping、relative bias、value norm、gate/sink、LSE | local/full；单/多 query；页边界；多请求；padding | 与短序列显式 attention 对齐 |
| T2 Cache | owner 绑定、pass 隔离、流偏移、块回收 | W−1/W/W+1；窗口边界；随机页表；两个交错请求 | 与无缓存参考等价，状态无串扰 |
| T3 Layer | 单 local/full 层、含 controller 的核心层 | pass0/1/2；相同输入的 CUDA/CPU/NPU 对比 | 定位到投影、conv、attention、MoE 和残差各边界 |
| T4 Model | 48 次访问、tokenizer/LM head、序列分块 | 短文本；多轮 chat；prefill→decode；不同 chunk | 每 visit hidden 与最终 logits 对齐 |
| T5 Distributed | HCCL、专家分片、路由与合并 | TP2/4/8；专家空分片 token；跨 rank 路由 | 无重复 reduce；与 TP1 小模型一致 |
| T6 Serving | 服务、并发、取消、请求回收 | batch1、多请求、流式与非流式、超长输入拒绝 | 输出格式与结束原因正确；缓存稳定回收 |
| T7 Optimizations | ACL Graph、prefix cache、优化 kernel | 每项独立开关，对照 eager | 精度与状态不回退后才记录性能 |
| T8 Vision | processor、image encoder、投影与 token 拼接 | 单图、多分辨率、文图混批、图像数限制 | 处理器参考一致；图文质量单独评估 |

测试路径建议为 `tests/ut/models/test_inkling2_*.py`、`tests/e2e/inkling2/`（待新增）。NPU 测试应以显式设备标记跳过无 NPU 的环境，跳过不算通过。

## 3. 数值验收方法

每个张量至少记录最大绝对误差、均方误差、相对误差分位数与余弦相似度；不要只依赖一个 global cosine。BF16 较小输出接近零时相对误差会失真，需结合绝对误差。

初始调试可采用如下**待校准阈值**，它们不是本项目已达到的指标：纯 FP32 elementwise oracle 使用 `atol=1e-5, rtol=1e-4`；包含 BF16 舍入的单算子使用 `atol=1e-2, rtol=1e-2` 作初筛。首次实测后按算子、输入幅度与已有正确实现的误差范围固定验收阈值，并提交理由；不能为让整体测试通过而静默放宽阈值。

router 应检查 expert ids 和 tie-break。对完全同一 logits 输入，ids 必须一致；端到端 matmul 误差造成 top-k 边界变化时，报告 margin、变化 token 数与对应输出误差。固定 tie 输入用较小 expert id 验证，不能把此类差异归为随机波动。

模型检查点建议包含：embedding 后、每 visit 的 attention 输出、MoE 输出、AttnRes mix、loop controller 输出、最终 hidden 和 logits。若某 visit 开始发散，先固定相同输入重放该子层，避免用最后生成文本猜测问题来源。文本输出可出现近似 logits 导致的贪心分叉，质量评测与算子正确性分别判定。

## 4. 纯文本质量集

第一组是可复现的 20–50 条烟测：中文、英文、代码、数值计算、多轮对话、长输入、不同停止条件。使用固定 chat template 和原 tokenizer，保存 token ids 以排除模板差异。随后运行固定版本的公开任务子集，并与同一权重方案的参考实现成对比较。

建议先取 200 条可稳定复现的数学/推理样本及一组中文任务作为回归，再扩展到完整评测。发布结果时记录样本选择、评分器、是否启用 thinking、最大输出 token 数、temperature 和数据集 revision。1M architectural context 只作为后续专项，不作为 A2 首次可运行的默认长度。

## 5. 性能实验设计

| 维度 | 第一阶段候选值 | 说明 |
|---|---|---|
| 设备 | A2 910B1 64 GB；A5 950PR；A5 950DT | A5 先补齐实际显存和软件支持信息 |
| 并行 | TP2、TP4、TP8 | 记录真实专家组映射；八卡还可比较多副本 |
| 输入长度 | 128、2K、8K、32K | 逐级扩大，在配置资源内测试 |
| 输出长度 | 128、512 | 固定输出长度用于可比吞吐 |
| 并发 | 1、8、32、64 | 显存可行后执行；不把 OOM 点剔除后隐瞒 |
| 执行路径 | eager；优化 kernel；ACL Graph | 每次只改变一个变量 |
| 阶段 | 纯 prefill、纯 decode、混合负载 | 单独记录及服务综合记录 |

每配置完成预热，再至少测三轮，报告每轮与汇总。单请求延迟和饱和吞吐分开测试；性能稳定阶段再按设备功耗/频率和温度条件排除明显干扰。

服务侧记录 TTFT、ITL/TPOT 的 p50/p95/p99、输出 token/s、成功请求/s、输入 token/s、实际并发、峰值显存、队列等待及完成率。kernel 侧记录 attention、MoE GEMM、dispatch/combine、HCCL、短卷积、AttnRes、controller 的占比，区分计算与通信。基准工具的参数以安装版本 `--help` 为准。

性能报告表在首次实测前保持空白：

| 设备/版本 | TP/实际专家分组 | 输入/输出/并发 | TTFT p95 | TPOT p95 | 输出 tok/s | 峰值显存 | 质量对照 | 状态 |
|---|---|---|---|---|---|---|---|---|
| A2 | 待测 | 待测 | — | — | — | — | — | 未执行 |
| A5 950PR | 待测 | 待测 | — | — | — | — | — | 未执行 |
| A5 950DT | 待测 | 待测 | — | — | — | — | — | 未执行 |

## 6. 发布前放行清单

- 完整模型加载、纯文本 eager 服务、至少一种 A2 多卡配置的 T0–T6 通过，才可把状态从“工程设计”改为“可运行”。
- 每个优化项都有同配置 eager 对照；明确支持的设备、版本、dtype、TP、上下文与并发范围。
- A2 的结果不自动覆盖 A5；950PR 与 950DT 分别记录。
- 文本通过后再验收图像；图像限制为零的文本服务应明确拒绝图片请求。
- 将可复现命令、环境摘要、检查结果和测量报告提交；不要将上游 NVIDIA 性能直接写入本工程的性能表。

## 上游测试参考

[Mini PR #69028](https://github.com/vllm-project/vllm/pull/69028) 说明了上游的 core ops、attention、model、MoE、vision 与 DSpark 检查。对应源码目录为 [`tests/models/inkling2`](https://github.com/vllm-project/vllm/tree/ab18334a4f60da74db7de89fe95594e5ed0cda36/tests/models/inkling2)。移植时复用数学用例与输入分布，替换 GPU 依赖；上游通过记录不转记为 NPU 通过。
