# Model Runner 选择：只使用 V1，不使用 MRV2

## 决策

项目目标执行路径固定为 **vLLM V1 Engine + Ascend Model Runner V1**。

状态：本页是适配设计，完整 NPU 模型、V1 兼容分支及实机验收均待实施；源码复核日期为 2026-11-24。V1 Engine 与 Model Runner 的代际是两个概念，保留 `vllm.v1` 的 scheduler / cache 接口符合本项目目标。

- 不适配、不实例化、不依赖 Ascend Model Runner V2（MRV2）。
- Worker 必须构造 V1 的 `NPUModelRunner`。
- V1 scheduler、V1 cache manager、V1 attention backend / metadata builder 是执行与缓存生命周期基线。
- ACL Graph 是否启用属于后续 V1 runner 优化项，不改变 runner 代际选择。

## 为什么需要单独处理版本

当前仓库的 Ascend 阅读快照 `1999fa582aba279dc4214de7ea3035535917d4c4` 和 submodule 快照 `79ee8d9bac80585816bacf6932222fc7cc30a00c` 均已删除历史 `worker/v1/model_runner.py`。后者的 `mrv2_utils.py::use_v2_model_runner()` 无条件选择 V2，配置 patch 遇到 `VLLM_USE_V2_MODEL_RUNNER=0` 会明确报错；`worker.py` 直接构造 V2 runner。因此当前 submodule 是源码参考，不能直接作为本项目 V1 执行基线。[当前选择逻辑](https://github.com/vllm-project/vllm-ascend/blob/79ee8d9bac80585816bacf6932222fc7cc30a00c/vllm_ascend/mrv2_utils.py)、[当前 Worker](https://github.com/vllm-project/vllm-ascend/blob/79ee8d9bac80585816bacf6932222fc7cc30a00c/vllm_ascend/worker/worker.py)。

历史提交 `f16e5e26801f86b071bca6454c9b9b2b63362e0a` 包含：

- `vllm_ascend/worker/v1/model_runner.py`：V1 NPU Model Runner，继承当时的 upstream GPUModelRunner 并扩展 NPU 执行、cache、attention metadata 和 ACL Graph。
- `vllm_ascend/worker/worker.py`：Worker 通过 `vllm_config.use_v2_model_runner` 选择 V1 或 V2；V1 分支构造 `NPUModelRunner`。
- `vllm_ascend/mrv2_utils.py`：显式 `VLLM_USE_V2_MODEL_RUNNER=0` 优先于默认策略，选择 V1。该文件虽含 `v2` 名称，但其中的选择逻辑也是历史 V1 启动链的一部分。
- 本仓库在 `reference-snapshots/vllm-ascend-v1/` 保留 runner 与 worker 两个文件供阅读和 diff；配置选择逻辑见下方固定上游链接。

这个历史提交只作为结构参考，尚未锁定与 Mini 兼容的安装组合。**推荐先选定仍保留 MRV1 的匹配 vLLM / vLLM Ascend 基线，再移植 Mini 模型和必要的缓存接口。** 先验证该组合的普通模型 V1 推理，再接 Inkling2。

若确需采用较新的引擎，须在项目分支维护完整 V1 兼容链。当前 vLLM submodule `28fb1365a90e9929e4596a72906e309dd7ac701f` 已无历史父类文件 `vllm/v1/worker/gpu_model_runner.py`，`VllmConfig` 也已无旧 runner 选择字段；不能仅把 Ascend 的单个 V1 runner 文件覆盖到新版本。选择兼容历史基线可避免把恢复整套旧执行链作为首期前置工作。[当前 vLLM worker 目录](https://github.com/vllm-project/vllm/tree/28fb1365a90e9929e4596a72906e309dd7ac701f/vllm/v1/worker)、[当前配置类](https://github.com/vllm-project/vllm/blob/28fb1365a90e9929e4596a72906e309dd7ac701f/vllm/config/vllm.py)。

## 目标文件改造

| 文件 | 计划 |
|---|---|
| `vllm_ascend/worker/v1/model_runner.py` | 在保留该路径的历史基线上修改 V1 runner；补齐 Inkling2 cache owner、cache bind、metadata 契约。采用更新引擎时才另行维护完整 V1 兼容链 |
| `vllm_ascend/worker/worker.py` | 固定 V1 runner 构造路径；拒绝 `use_v2_model_runner=True` |
| `vllm_ascend/mrv2_utils.py`、`vllm_ascend/platform.py` | 核对所选基线的配置 hook；前端、engine-core 和 worker 均解析为 V1，禁止无条件 V2 patch 覆盖选择 |
| `vllm_ascend/attention/inkling2.py`（拟新增） | 实现所选 V1 版本的 AttentionBackend、MetadataBuilder 和 Impl |
| `vllm_ascend/attention/inkling2_sconv.py`（拟新增） | 将 Short Conv State 注册为 V1 可发现的 cache owner，桥接上游新缓存配置 |
| `vllm_ascend/models/inkling2/*`（拟新增） | 模型结构、loop、权重加载与算子调用；不依赖 MRV2 ModelState 执行链 |

不要为了接入模型而改写 V1 scheduler 或绕过 V1 KVCacheManager。若某种多 pass cache owner 无法由标准 V1 API 表达，先提交最小可测试的 V1 扩展。

## Mini 新缓存接口与历史 V1 的桥接

当前上游 Mini 的 attention / conv owner 实现 `get_kv_cache_config()`，返回含物理 spec 和 manager 配置的 `KVCacheLayerConfig`。历史 V1 runner 则调用 owner 的 `get_kv_cache_spec(vllm_config)`。适配必须同时保存 local/full 窗口规则、conv 单块布局、pass owner 唯一性，以及分配、绑定、回收的语义，不能只改方法名。详细映射和验收见 [cache-and-loop.md](cache-and-loop.md)。

这个新接口位于 vLLM V1 Engine 的公共缓存层，并不等同于 MRV2。历史 Ascend V1 runner 也支持部分共享 KV backing 布局；共享 backing 或 backend view 是否可复用应按所选 V1 实现判断。[上游 attention 基类](https://github.com/vllm-project/vllm/blob/28fb1365a90e9929e4596a72906e309dd7ac701f/vllm/model_executor/layers/attention_layer_base.py)、[历史 V1 runner](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/v1/model_runner.py)。

## 启动和代码级保护

部署脚本对 dry-run 与实际启动均固定 `VLLM_USE_V2_MODEL_RUNNER=0`。该环境变量是上述历史基线已核验的选择方式；不设置 `VLLM_USE_V1=0`。对已删除 V1 的版本，启动失败表示需要换用 V1 兼容组合，不能通过清除变量继续运行 MRV2。

配置层和实际 Worker 构造仍须验收：仅设置环境变量不能证明使用了 V1。所选分支应在各进程应用配置后确认 `use_v2_model_runner is False`，并记录实际 runner 的完整类路径。环境变量、qualification 记录与本机代码必须一致。

在 Worker 初始化处增加硬断言，避免静默回退：

```python
if getattr(vllm_config, "use_v2_model_runner", None) is not False:
    raise RuntimeError(
        "Inkling2 Ascend port requires Model Runner V1; "
        "MRV2 is explicitly unsupported by this project."
    )
self.model_runner = NPUModelRunner(vllm_config, device)
```

上面是待实现的 Worker 行为示意，实际字段访问和构造签名须按选定版本核对；字段缺失不能被当作已选择 V1。

## 验收

1. 启动日志明确记录 V1 runner 类路径、解析后的选择结果和版本；前端与各 worker 一致。
2. 测试断言 V2 runner 构造器没有被调用。
3. V1 worker 完成 load_model、profile_run、get_kv_cache_spec、initialize_kv_cache、execute_model、sample_tokens。
4. 48 个 attention owner 与 48 个 conv owner 通过 V1 manager 分配、绑定、释放；新旧接口桥接保留窗口规则和物理布局。
5. prefill、decode、chunked prefill、请求取消、块复用和并发请求通过。
6. V1 eager 正确后再验证 V1 ACL Graph；不得以迁移到 MRV2 作为性能优化方案。

## 固定来源

- [历史 V1 runner](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/v1/model_runner.py)
- [历史 V1/V2 worker 选择逻辑](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/worker.py)
- [历史 V1 环境变量与配置选择](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/mrv2_utils.py)
- [当前 Ascend 仅保留 V2 的选择逻辑（排除依据）](https://github.com/vllm-project/vllm-ascend/blob/79ee8d9bac80585816bacf6932222fc7cc30a00c/vllm_ascend/mrv2_utils.py)
