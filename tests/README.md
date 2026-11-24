# 验证记录与测试入口

记录日期：2026-11-24。以下已执行检查发生在普通 CPU 工作区，未连接 Ascend 设备，未加载模型，未启动 vLLM 服务。

## 已执行：规划工具

```bash
python3 -m unittest discover -s tests -p test_planning.py -v
```

结果：**6 个测试通过**。

| 用例 | 验证目的 |
|---|---|
| 层访问顺序与 owner 唯一性 | 确认 48 次访问、core pass 边界、48 个独立 KV / conv 逻辑 owner、40 local |
| SWA 与 full context | local 长度饱和于窗口后，global 仍按上下文增长 |
| 多请求与 TP | 逻辑 KV 随请求数增长，并按有效 head 分片计算 |
| head_dim 显式读取 | hidden size 改变不应错误改变 attention head_dim |
| 非法 TP | TP 无法整除 Q / KV heads 时拒绝估算，避免截断 |
| loop 参数校验 | variable-loop 或越界 core 不静默套用固定模型规划 |

一次实际 CLI 示例：

```bash
python3 scripts/inspect_model_config.py configs/model_shape.example.json --tp 4 --tokens 8192
```

结果为 24 物理层、48 访问、40 local / 8 global；单请求每 rank **逻辑 KV 主体 109,051,904 bytes（0.1015625 GiB）**。这是公式核验，排除了 conv、分页保留、权重、临时空间等，不能解释成模型总显存。

## 已执行：部署辅助脚本

初次准备中执行并通过：

- 三份 shell 脚本 `bash -n`，以及 Python 编译检查。
- A2 TP2 / TP8、A5 950PR / 950DT 共四种 dry-run 命令生成。
- 默认未资格化配置的 `--execute` 拒绝执行。
- 重复设备 ID 拒绝执行。
- 默认 collector JSON 输出检查，不触发 NPU probe。

脚本复核命令：

```bash
python3 -m compileall -q scripts tools tests
bash -n scripts/collect_env.sh scripts/serve_a2.sh scripts/serve_a5.sh
bash scripts/serve_a2.sh --tp 2 --devices 0,1 --model /models/inkling2-mini-bf16
bash scripts/serve_a5.sh --profile a5-950dt --tp 2 --devices 0,1
```

这些检查不证明目标设备支持特定 vLLM CLI flag。实际执行模式会查询用户安装版本的 `vllm serve --help=all`，随后仍需模型适配和实机验收。

## 已执行：V1 runner 启动约束回归

本次 V1-only 修改新增 `test_runner_policy.py`，全部测试命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：**12 个测试通过**，包括 6 个规划测试和 6 个启动策略测试。后者验证：

- 调用者环境为 MRV2=1 时，dry-run 仍显示 `VLLM_USE_V2_MODEL_RUNNER=0`，且不启动 CLI、模型或设备。
- CLI 检查与实际启动使用同一 V1 子进程环境，父环境保持原值。
- 缺失 runner 策略、V2 策略、非 JSON 布尔 false 均被拒绝。
- 资格记录必须有实际 V1 类路径及 `v1_runner_verified`。
- schema 1 不能绕过 schema 2 新增的 runner 验证。
- 默认未资格化配置在 A2、950PR、950DT 三种 profile 下均阻止执行。

同期重新通过 Python 编译、3 份 shell 语法检查与 A2 TP2/TP8、A5 950PR/950DT 共 4 种 dry-run。策略单元测试中的已资格化记录、CLI 帮助和启动调用均为 CPU mock；这些结果只证明脚本约束，真实 V1 类构造与 NPU 推理仍按 [测试计划](../docs/benchmark-plan.md) 上机验证。

## 尚未执行：模型与硬件

没有执行 BF16 权重转换、NPU 算子对齐、完整模型生成、HCCL 多卡测试、A2/A5 显存测量或性能 benchmark。分层测试与放行条件见 [benchmark-plan.md](../docs/benchmark-plan.md)，测试结果用 [实机报告模板](../.github/ISSUE_TEMPLATE/hardware-validation.md) 记录。
