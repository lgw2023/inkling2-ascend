# 第二阶段：图像接入

首期验收以纯文本为范围；本章为后续实现任务。源码依据采用 [来源清单](sources.md) 中固定的 Mini revision 与模型配置。

## 依赖关系

先完成文本骨干、loop / cache、专家权重加载与服务正确性，然后实现 vision 路径。上游模型具有条件初始化的视觉分支；关闭 image limit 后，首期不应为了文本推理装载不需要的 vision 权重。是否真正跳过必须用权重加载统计和显存观测确认。

## 文件与接口任务

| 位置 | 要完成的行为 | 验收 |
|---|---|---|
| 上游 `vllm/models/inkling2/processing.py` / 对应 revision 的处理器实现 | 固定 tokenizer、image processor、chat template 和 special token 规则 | 同一图片预处理后的 tensor shape / 数值一致 |
| 建议新增 `vllm_ascend/models/inkling2/vision.py` | 迁移视觉编码、归一化、激活、attention 和位置处理 | 逐模块输出与上游参考一致 |
| 建议新增 `vllm_ascend/models/inkling2/multimodal.py` | 对接 vLLM MM 输入与特征接口；图像 token、特征到文本嵌入的对应关系 | token 数量、位置和 feature shape 精确匹配 |
| `vllm_ascend/models/inkling2/model.py`（建议新增） | `get_multimodal_embeddings` / 输入嵌入接口按锁定版本签名实现 | text-only 输出不回退，图像特征进入正确 token 位置 |
| 模型 loader | 在 MM 开启时装载 vision / projector，核对 dtype 和格式 | 缺失或额外键均可解释，禁用图像时不误报未加载视觉键 |
| attention / runner | 如 vision 使用非因果 attention，选用对应 backend；合理预算 encoder 峰值和 MM cache | 混合文本/图像请求可调度，无请求间复用错误 |

表内 Ascend 文件名是建议，非已有实现；接口名称以实际固定的 vLLM MM protocol 为准。

## 图像测试集

1. 纯文本同输入回归，确保打开 MM 能力不改变文本行为。
2. 单图：小/大分辨率、不同长宽比、边界像素值；校验 resize / normalize / patch 的处理结果。
3. 多图：不同图片、重复图片、顺序交换；校验 token 插入和各图片特征归属。
4. 服务：文本与图像混批、请求取消、长文本+图片、达到 image limit 的请求。
5. 容量：单独记录 encoder 峰值、图像特征缓存、prefill 和 decode；不要把纯文本容量直接用于多模态。

数值门槛先由同精度参考与实际基线测得，再在报告中固定。第二阶段启动配置单独维护，首期文本脚本继续保留 `image=0` 的约束。
