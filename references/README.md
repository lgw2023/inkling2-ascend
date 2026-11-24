# 上游推理源码参考

本目录通过 Git submodule 固定九个上游推理仓库的 commit。完整代码由 Git submodule 提供，不重复复制全部上游历史。克隆并下载源码：

```bash
git clone https://github.com/lgw2023/inkling2-ascend.git
cd inkling2-ascend
# 先拉与模型开发直接相关的仓库（按需选择）：
git submodule update --init references/upstream/vllm references/upstream/vllm-ascend references/upstream/transformers references/upstream/tokenspeed references/upstream/unsloth
# 需要全部完整上游源码时：
git submodule update --init --recursive
```

| 项目 | 用途 | 固定 commit |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 官方模型实现、算子、权重加载与测试 | `28fb1365a90e9929e4596a72906e309dd7ac701f` |
| [vLLM Ascend](https://github.com/vllm-project/vllm-ascend) | 昇腾 registry、runner、cache、attention、MoE | `79ee8d9bac80585816bacf6932222fc7cc30a00c` |
| [SGLang](https://github.com/sgl-project/sglang) | 官方支持的替代推理引擎 | `1173264b586a92e534f855a91e59f07644283248` |
| [Transformers](https://github.com/huggingface/transformers) | HF 模型、processor 和参考前向 | `d8f10a82213a3a9db48385aba3a4b57ab535de50` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF、量化和独立推理实现 | `af5c58347fdbc2cd19e65b278e81c8f528f8a175` |
| [Maya Inkling Mini](https://github.com/alesha-pro/maya-inkling-mini) | MIT 许可的 CUDA/HIP 优化实现 | `27daec5e608aa59dab45184e82a1f0611ea7c1aa` |
| [TokenSpeed](https://github.com/lightseekorg/tokenspeed) | Inkling2 runtime、gfx950 monokernel、测试和 serving recipe | `ffdc204c6e6c9c96e2c801b9fac2f2ed5ab7c237` |
| [Unsloth](https://github.com/unslothai/unsloth) | Inkling2 Mini loader、FlexAttention 和训练支持 | `da7b561499a3fbbfc625f665f284342bee792691` |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) | Inkling 2 Mini SGLang 聚合/分离部署 recipes | `28697ef5c4115d2fd5793b8f1065ed711e5459cc` |

SHA 是可复现的源码参考 pin，不表示这些仓库构成已经验证兼容的软件组合。尤其不能把最新 vLLM 与最新 vLLM Ascend 自动视为配套 release。

## 官方模型与权重

[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) 的 Apache-2.0 checkpoint revision 固定为 `1fb4134b9a74f9681054f4ef82e4bf1f458a2337`。仓库约 20.2 GB，包含约 19.5 GB safetensors 权重和 MTP drafter。模型权重不提交到 Git；请从官方 HF revision 单独下载。

## 推荐阅读顺序

1. vLLM：`vllm/models/inkling2/` 与 `tests/models/inkling2/`，核对官方模型、loop、attention、MoE、processor 和测试。
2. vLLM Ascend：模型注册、V2 runner、KV cache backend、fused MoE，设计 Ascend 接口接入。
3. SGLang / Transformers：交叉核对模型语义、processor 和服务路径。
4. llama.cpp / Maya：研究 GGUF 量化、缓存、算子融合和低显存优化。

Maya 仓库代码为 MIT，但树中还包含其他第三方许可。各 submodule 的 LICENSE/NOTICE 仍然有效；复制代码时保留原始 SPDX、版权和许可声明。
