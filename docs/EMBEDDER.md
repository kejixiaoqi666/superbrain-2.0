# 真实 Embedding 模型（可插拔）

SuperBrain 2.0 的记忆检索默认用**零依赖哈希嵌入**（`HashingEmbedder`，256 维）。
它跨进程/平台结果一致、零依赖，但对**零字面重叠的同义词**有天花板（如「卡顿」
↔「延迟高很卡」「开通」↔「启用」）。本模块引入**可插拔的真实语义模型**（默认
`bge-base-zh-v1.5`，ONNX，768 维，约 407 MB），按需单独安装、不改变默认行为。

## 为什么「脱离架构单独安装」

真实模型及其运行依赖（`onnxruntime`、`tokenizers`）都是重组件，且模型本体数百
MB。为保持「无 Rust / 无重依赖也照样跑」的可移植性：

- 模型文件**不进 git 历史**（会拖垮 clone、撑爆仓库），而是作为**开源仓库
  GitHub Release 资产**发布，安装时直接从那里下载。
- `onnxruntime` / `tokenizers` **懒加载**：只在真正调用真实嵌入器时才 import，
  缺失时报清晰安装指引，绝不影响默认哈希路径。

## 安装（三步）

```bash
# 1) 运行依赖
pip install onnxruntime tokenizers

# 2) 模型（从开源仓库 GitHub Release 拉取，约 407 MB）
python -m superbrain2 install-model
#    可选：SUPERBRAIN_EMBEDDER_MODEL_DIR=/data/models python -m superbrain2 install-model

# 3) 接线使用（二选一）
#    a. 代码
from superbrain2 import AgentConfig
cfg = AgentConfig(embedder="bge")            # 或 embedder_kwargs={"model_dir": "...", "dim": 768}

#    b. 环境变量（见下）
```

## 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `SUPERBRAIN_EMBEDDER` | 嵌入器名 `hashing` / `bge` | `hashing` |
| `SUPERBRAIN_EMBEDDER_MODEL_DIR` | 模型目录（含 `model.onnx` + `tokenizer.json`） | `~/.superbrain/models/bge-base-zh-v1.5` |
| `SUPERBRAIN_EMBEDDER_MODEL` | 安装时要下载的模型名 | `bge-base-zh-v1.5` |
| `SUPERBRAIN_EMBEDDER_RELEASE` | GitHub Release 标签 | `models` |

## 语义漂移守卫（换嵌入器绝不静默混算）

记忆向量是用某个嵌入器算出来的；一旦换嵌入器（模型 / 维度 / 版本变），旧向量
与新向量**不可比**，直接混算会把检索搞坏。内核内置两道防线：

1. **指纹握手** `embedder_identity(emb)` → `{type, name, dim, revision, fp}`，
   跨设备同步前用 `fp` 比对。
2. **存储守卫** `store.embedder_compat(fp)`：库内有向量但指纹不一致 → 返回
   `needs_reembed`。Agent 侧 `ensure_embedder_ready(auto_reembed=True)` 会自动
   全量 `reembed_all` 并重打指纹，修复语义漂移。

```python
brain = SuperBrain.from_env()
info = brain.embedder_info()          # 当前嵌入器自描述
r = brain.ensure_embedder_ready()     # {'compat', 'fingerprint', 'reembedded'}
```

## 实测对比（bge-base-zh-v1.5 fp32，CPU）

| 文本对 | 哈希嵌入 cos | bge cos |
|---|---:|---:|
| 卡顿不稳定 ↔ 延迟高很卡 | 0.21 | **0.72** |
| 开通启用 ↔ 启用新线路入口 | 0.11 | **0.89** |
| 无关对（晚饭/退款） | ~0.00 | **0.26~0.38** |

真实模型突破零字面重叠的同义词天花板。**fp32 全精度对无关对的余弦基线低至
0.26**，语义分离干净，`rel_floor` 过滤有效（量化 int8 会抬高无关基线到 0.6+，
是「语义偏移」的常见来源，故默认发布 fp32）。

## 性能与优化（诚实结论）

- **稳态推理**：fp32 单条约 **30~35 ms**（CPU，pre-warm 后）。
- **内容缓存**（核心提速）：同文本 → 同向量，结果以 LRU 缓存（默认 1024 条）。
  重复查询 / 记忆去重直接命中，**命中耗时 ≈ 0**。换嵌入器迁移、重复索引场景收益最大。
- **批量嵌入** `embed_many`：公开 API。诚实说明——在**单核 CPU 跑短文本**时批量
  与逐条**几乎无差异**（实测 1.1x~0.9x，噪音级），未接入 `reembed_all` 以避免回归；
  多核 / GPU 部署因模型可跨序列并行，批量才有实际吞吐收益。
- **进一步提速的取舍**：换 int8 量化模型可再快约 4x，但无关基线抬高、语义偏移略增，
  需按你的真实语料实测权衡——`SUPERBRAIN_EMBEDDER_MODEL_DIR` 指向 int8 包即可切换。

## 发布模型资产

模型以 GitHub Release 资产分发（`kejixiaoqi666/superbrain-2.0`，tag `models`）：

```
bge-base-zh-v1.5-onnx-linux-x64.tar.gz      # 内含 model.onnx + tokenizer.json
bge-base-zh-v1.5-onnx-linux-x64.tar.gz.sha256
```

安装脚本 `python -m superbrain2.model install-model` 负责下载、sha256 校验、
解压到目标目录。
