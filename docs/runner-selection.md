# Model Runner 选择：只使用 V1，不使用 MRV2

## 决策

项目目标执行路径固定为 **vLLM V1 Engine + Ascend Model Runner V1**。

- 不适配、不实例化、不依赖 Ascend Model Runner V2（MRV2）。
- Worker 必须构造 V1 的 `NPUModelRunner`。
- V1 scheduler、V1 cache manager、V1 attention backend / metadata builder 是执行与缓存生命周期基线。
- ACL Graph 是否启用属于后续 V1 runner 优化项，不改变 runner 代际选择。

## 为什么需要单独处理版本

当前仓库固定的 vLLM Ascend 阅读快照 `1999fa582aba279dc4214de7ea3035535917d4c4` 已采用 `vllm_ascend/worker/v2/model_runner.py`，没有历史 `worker/v1/model_runner.py` 路径。不能把当前 main 当作可直接满足 V1 目标的执行器。

历史提交 `f16e5e26801f86b071bca6454c9b9b2b63362e0a`（2026-10-21）包含：

- `vllm_ascend/worker/v1/model_runner.py`：V1 NPU Model Runner，继承当时的 upstream GPUModelRunner 并扩展 NPU 执行、cache、attention metadata 和 ACL Graph。
- `vllm_ascend/worker/worker.py`：Worker 通过 `vllm_config.use_v2_model_runner` 选择 V1 或 V2；V1 分支构造 `NPUModelRunner`。
- 本仓库在 `reference-snapshots/vllm-ascend-v1/` 保留这两个文件供阅读和 diff。

这个历史提交比 Inkling 2 Mini 的 vLLM 模型实现更早，不能直接视为可安装组合。应把 V1 runner 结构回移到所选 vLLM Ascend 分支，并适配当前 V1 engine API；先跑 V1 基础模型回归，再接 Inkling2。

## 目标文件改造

| 文件 | 计划 |
|---|---|
| `vllm_ascend/worker/v1/model_runner.py` | 恢复/回移 V1 runner；实现 Inkling2 所需的 cache spec、cache bind、metadata 生命周期 |
| `vllm_ascend/worker/worker.py` | 固定 V1 runner 构造路径；拒绝 `use_v2_model_runner=True` |
| `vllm_ascend/attention/inkling2.py` | 实现 V1 AttentionBackend、MetadataBuilder 和 Impl |
| `vllm_ascend/attention/inkling2_sconv.py` | 将 Short Conv State 注册为 V1 可发现的 cache owner |
| `vllm_ascend/models/inkling2/*` | 模型结构、loop、权重加载与算子调用，不依赖 MRV2 专有的 ModelState 接口 |

不要为了接入模型而改写 V1 scheduler 或绕过 V1 KVCacheManager。若某种多 pass cache owner 无法由标准 V1 API 表达，先提交最小可测试的 V1 扩展。

## 启动和代码级保护

配置层必须保证 `use_v2_model_runner=False`。具体 CLI / config 传递方式由选定的 vLLM 版本决定，不在未锁版本时猜测启动参数。

在 Worker 初始化处增加硬断言，避免静默回退：

```python
if vllm_config.use_v2_model_runner:
    raise RuntimeError(
        "Inkling2 Ascend port requires Model Runner V1; "
        "MRV2 is explicitly unsupported by this project."
    )
self.model_runner = NPUModelRunner(vllm_config, device)
```

上面是目标行为示意，实际字段访问和构造签名须按选定版本核对。

## 验收

1. 启动日志明确记录 V1 runner 类路径和版本。
2. 测试断言 V2 runner 构造器没有被调用。
3. V1 worker 完成 load_model、profile_run、get_kv_cache_spec、initialize_kv_cache、execute_model、sample_tokens。
4. 48 次 layer visit 的 KV/Conv cache owner 通过 V1 manager 分配、绑定、释放。
5. prefill、decode、chunked prefill、请求取消、块复用和并发请求通过。
6. V1 eager 正确后再验证 V1 ACL Graph；不得以迁移到 MRV2 作为性能优化方案。

## 固定来源

- [历史 V1 runner](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/v1/model_runner.py)
- [历史 V1/V2 worker 选择逻辑](https://github.com/vllm-project/vllm-ascend/blob/f16e5e26801f86b071bca6454c9b9b2b63362e0a/vllm_ascend/worker/worker.py)
- [当前 Ascend main runner（只用于确认排除对象）](https://github.com/vllm-project/vllm-ascend/blob/1999fa582aba279dc4214de7ea3035535917d4c4/vllm_ascend/worker/v2/model_runner.py)
