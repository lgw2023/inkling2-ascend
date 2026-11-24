# Inkling 2 Mini inference references

This index distinguishes model-specific code, framework integrations, deployment recipes and explanatory articles. Full upstream repositories are pinned as Git submodules where useful; selected model-specific source files are mirrored under `reference-snapshots/`.

## Official model and framework

- [Inkling 2 Mini model repository](https://huggingface.co/thinkingmachines/Inkling-2-Mini) — model card, config, tokenizer, processor, safetensors and MTP drafter. Weights are not mirrored into Git.
- [Transformers Inkling 2 model docs](https://huggingface.co/docs/transformers/main/model_doc/inkling2) — HF API and processor reference.
- [Transformers source](https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2) — pinned in `references/upstream/transformers`.
- [vLLM model implementation](https://github.com/vllm-project/vllm/tree/main/vllm/models/inkling2) — primary model-execution reference.
- [vLLM Ascend](https://github.com/vllm-project/vllm-ascend) — target NPU backend; no implication that this model is already supported on Ascend.

## Model-specific inference projects

- [TokenSpeed Inkling recipe](https://github.com/lightseekorg/tokenspeed/blob/main/docs/recipes/models.md) — TP/EP serving recipe and decode monokernel options.
- [TokenSpeed Inkling2 GFX950 kernels](https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/monokernel/inkling2) — AMD-specific attention, conv, mix, expert and quantization kernels. Selected files and tests are mirrored locally.
- [Unsloth Inkling2 implementation](https://github.com/unslothai/unsloth/blob/main/unsloth/models/inkling2.py) — loading, projection wrappers, training and adaptation logic.
- [Unsloth FlexAttention](https://github.com/unslothai/unsloth/blob/main/unsloth/models/inkling2_flex.py) — attention optimization reference.
- [NVIDIA Dynamo Mini recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes/inkling-2-mini) — SGLang aggregate/disaggregated deployment and model cache; selected YAMLs and docs are mirrored locally.
- [Maya Inkling Mini](https://github.com/alesha-pro/maya-inkling-mini) — independent CUDA/HIP engine and optimization reference.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF and quantized inference path.

## Explanatory articles

- [Hugging Face: Welcome Inkling 2 Mini](https://github.com/huggingface/blog/blob/main/inkling2mini.md)
- [Hugging Face: Thinking Machines Inkling](https://github.com/huggingface/blog/blob/main/thinkingmachines-inkling.md)

These articles are linked rather than copied in full. The first is Mini-specific; the second is broader Inkling background.

## License and applicability

Code snapshots retain upstream headers and corresponding license texts. TokenSpeed snapshot is MIT; Dynamo is Apache-2.0; Unsloth core and its copied model files have file-level license distinctions (including AGPL-3.0-only). Verify each file's SPDX/NOTICE before copying or adapting. AMD GFX950 code is not a direct CANN implementation.
