# 协作约定

优先按 [ROADMAP.md](ROADMAP.md) 的任务编号提交小范围变更，注明属于资料更正、工程实现还是实机验证。

执行路径固定为 **vLLM V1 Engine + Ascend Model Runner V1**。模型、缓存、图像和性能改造均不引入 MRV2 运行依赖；修改启动或 worker 选择逻辑时，补充 V1 类实际构造及 MRV2 未构造的验证，见 [runner-selection.md](docs/runner-selection.md)。

## 修改前

1. 固定模型 revision、vLLM / vLLM Ascend commit 与运行环境版本。
2. 查阅相关文档的上游依据，确认讨论的是 Inkling 2 Mini。
3. 为改动列出可观察的验收结果，例如特定输入的 logits、独立缓存 owner、取消后的页回收、每 rank 通信次数。

## 提交内容

- **资料更正**：附官方来源与固定 revision，写清影响哪个实现决策。
- **代码变更**：说明问题、修改与结果，标明新增/复用的算子以及 dtype / layout / 通信契约。
- **实机结果**：记录设备 SKU、可用显存、版本、模型 revision、命令、seed、输入长度与并发、实际测量方法。日志仅包含必要技术字段。
- **性能改动**：同时提交正确性回归与相同条件的前后结果，区分预热、prefill、decode 和端到端服务。

## 不随仓库提交的材料

模型权重、数据集样本中不适合公开的内容、凭据、序列号、完整环境变量、个人身份信息和未整理的设备日志。`.gitignore` 已包含常见本地产物。

## 现有工具检查

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tools tests
bash -n scripts/collect_env.sh scripts/serve_a2.sh scripts/serve_a5.sh
```

这些检查覆盖辅助工具。完整模型验收按 [测试计划](docs/benchmark-plan.md) 执行，尚未上机的项目保持“未执行”。
