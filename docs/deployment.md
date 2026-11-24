# 部署环境与可审查启动模板

本仓库当前提供方案、检查工具和命令模板，**尚未实现完整 Inkling2 Mini NPU 模型，也没有 A2/A5 跑通结果**。模板默认 dry-run；它不会下载权重、安装软件或启动服务。

## 1. 先确定完整软件组合

2026-11-24 核验的 vLLM Ascend 官方 installation 页面给出以下验证组合：

| 层 | 官方页面中的参考版本 |
|---|---|
| Python | 3.12 |
| CANN Toolkit + 对应硬件 Ops / NNAL | 9.2.1 / 9.2.1 |
| PyTorch / TorchNPU | 2.10.0 / 2.10.0.post7 |
| vLLM / vLLM Ascend | 0.32.0 / 0.32.0rc1 |
| Triton Ascend / FLA NPU | 3.6.0.post2 / 26.9.1，适用产品见官方说明 |

这是引擎参考环境，不是本项目兼容性锁文件。当前已有 CANN 的实际版本未知；应先采集，再选择同一发布组合，或为 Inkling2 代码所需的开发提交完成独立兼容验证。不要把上游 `main` 的模型代码和一个旧 release 随意组合。安装入口：[官方 installation](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html)、[dependency version management](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/contribution/dependency_version_management.html)。

A2 的官方常规 pip 构建说明不能自动套到 A5。A5 应选与具体 PR/DT 产品匹配的镜像、wheel variant 或源码构建配置。驱动/固件按实际主机与发布矩阵确定；本仓库不提供自动替换驱动的脚本。

## 2. 首次准备步骤

1. 用 `scripts/collect_env.sh` 取得包版本白名单。按本机真实安装路径加载 CANN/NNAL 环境；版本缺失项人工填写。
2. 固定四个提交：模型 checkpoint、vLLM、vLLM Ascend、适配代码。同时记录镜像 digest 或 Python 依赖锁、CANN/NNAL/driver/firmware 版本。
3. 完成 [文件级改造](vllm-ascend-integration.md)，确认模型 registry、NPU attention、conv/cache、controller 与 BF16 MoE/loader 已接入。
4. 按 [测试计划](benchmark-plan.md) 跑完目标设备的算子、缓存隔离、TP 和纯文本端到端验收，留下实际结果文件。
5. 用测试证据填写 `configs/qualification.json`。只有已验证的卡型和 TP 值加入 `validated_tp_sizes`，未完成项保持 `false`。

`--dtype bfloat16` 只决定所支持路径的运行 dtype，不能把 upstream 的 NVIDIA NVFP4/MXFP8 专家实现自动转成昇腾 BF16 实现。需要独立转换/加载器及 BF16 expert 模块。转换结果是量化权重的反量化近似，不是恢复训练时原始 BF16 权重，见 [量化计划](quantization.md)。

## 3. 查看启动配置

在仓库根目录执行：

```bash
# 默认 dry-run，不读取模型、不申请 NPU。
bash scripts/serve_a2.sh --tp 2 --devices 0,1 --model /models/inkling2-mini-bf16
bash scripts/serve_a2.sh --tp 4 --devices 0,1,2,3 --model /models/inkling2-mini-bf16
bash scripts/serve_a5.sh --profile a5-950dt --tp 2 --devices 0,1 --model /models/inkling2-mini-bf16
bash scripts/serve_a5.sh --profile a5-950pr --tp 2 --devices 0,1 --model /models/inkling2-mini-bf16
```

默认命令包含：

| 参数 | 首期设置 | 目的 |
|---|---|---|
| served model name | `inkling2-mini` | 请求端与服务端名称一致 |
| TP / PP | 2 / 1 | 从双卡真实模型起步 |
| distributed executor | `mp` | 单机进程执行 |
| dtype | BF16 | 适配完成后的首期计算路径 |
| enforce eager | 开启 | 先验证算子与缓存语义 |
| context / batched tokens | 2048 / 2048 | 初期短上下文、无 chunked prefill |
| max sequences | 1 | 先建立单请求参照 |
| memory utilization | 0.80 | 启动探测初值；不是显存容量证明 |
| prefix cache / chunked prefill | 关闭 | 后续分别做功能验收 |
| image limit | 0 | 不构建或接受图像路径；模型接入需兑现该限制 |
| speculative decoding | 不配置 | 首期不接 MTP/DSpark 等附加路径 |
| bind / port | 127.0.0.1 / 8000 | 本机验证 |

脚本使用 `ASCEND_RT_VISIBLE_DEVICES` 指定实例设备；只在实际执行的子进程环境设置，不修改用户 shell。设备列表必须与 TP 数量相等且无重复。TP=8 可用于八卡对照，不作为默认性能建议。

## 4. 完成适配后的实际执行

只有在上述步骤产生真实测试结果后，才将对应 qualification gates 设为 true、状态设为 `qualified`。随后：

```bash
bash scripts/serve_a2.sh \
  --tp 2 --devices 0,1 \
  --model /models/inkling2-mini-bf16 \
  --qualification configs/qualification.json \
  --execute
```

执行模式检查资格文件、完整提交哈希记录、实际安装包版本、内存字段、卡型/TP 验证记录和 checkpoint 的 `config.json`。它再读取**本机安装版本**的 `vllm serve --help=all`，核实每个将使用的 flag 都存在；遇到不支持的 flag 会停止，不静默删除参数。该检查能防止明显配置误用，不替代代码或测试证据审查。

CLI 参考：[vLLM CLI Guide](https://docs.vllm.ai/en/latest/cli/)、[v0.32.0 serve](https://docs.vllm.ai/en/v0.32.0/cli/serve/)。如果所选提交改变参数，先改脚本与文档，再重跑验证。

## 5. 请求验证

服务启动成功后，先检查模型列表，再发送确定性短请求：

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling2-mini","messages":[{"role":"user","content":"用一句话解释矩阵乘法。"}],"temperature":0,"max_tokens":64}'
```

chat template 使用固定 checkpoint 所附模板，并与参照环境保持一致。确认请求稳定后，才逐项增加 context、并发、chunking、prefix cache 和图模式。图像阶段需要新增 processor/vision/projector 的端到端校验，然后明确调整 multimodal 限制。

## 6. 当前已验证内容

脚本只在普通 CPU 工作区检查语法、dry-run 参数生成和未资格化配置的阻断行为；没有调用 Ascend 设备、没有启动 vLLM、没有加载模型。具体本地检查结果由本仓库初次提交的验证说明记录。A2/A5 的精度、速度、峰值内存和连续服务稳定性仍待实际机器测试。
