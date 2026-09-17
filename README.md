<div align="center">

# 🧠 SuperBrain 2.0

**Rust 高性能内核 · 多后端向量索引 · 完整自主 Agent**

纯 Python 基线的 Rust 混合内核升级版 —— 用 **PyO3 + SIMD + HNSW** 把向量检索做到极限，同时保留「无 Rust 也照样跑」的可移植性。

[![version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/kejixiaoqi666/superbrain-2.0/releases)
[![Rust](https://img.shields.io/badge/Rust-PyO3%20%2B%20SIMD-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Backends](https://img.shields.io/badge/backends-faiss%20·%20usearch%20·%20rust%20·%20python-brightgreen)]()

**向量检索加速 ≈ 469×** · 四后端召回全部正确 · 完整 Agent 能力

</div>

---

## 🚀 它是什么

**SuperBrain 2.0** 是把第一版超脑的**完整 Agent 内核**（认知 / 情绪 / 记忆 / 人格 / 自主 / 表达）与 **Rust 高性能向量内核**结合的升级版。

一句话：**一个「既有情感内核、又跑得飞快」的自主 Agent 框架。**

```python
from superbrain2 import SuperBrain, VectorIndex

# 完整 Agent：第一版全部能力
brain = SuperBrain.from_env()
reply = brain.chat("你好")            # 对话（人格驱动）
brain.remember("用户喜欢茶")          # 记忆
brain.recall("喜欢什么")              # 检索
brain.apply_style("可爱")             # 表达风格
brain.dream(); brain.tick()           # 自主思考 / 推进

# 多后端向量索引：召回可靠优先，自动降级
idx = VectorIndex(dim=64, backend="auto")   # faiss → usearch → rust → python
idx.add([...]); idx.search(query, k=5)      # -> [(id, 分数), ...]
```

---

## ⚡ 性能（真实基准，dim=64 · 10,000 条 · top-5 · ef=32）

| 后端 | 查询延迟 | 构建 1 万条 | 召回 top1 |
|---|---------:|-----------:|:---:|
| **faiss** (HNSW) | **0.07 ms** | 0.96 s | ✅ 1.00 |
| **usearch** (HNSW) | **0.10 ms** | 0.99 s | ✅ 1.00 |
| **rust** (自研 SIMD+HNSW) | **0.10 ms** | 1.38 s | ✅ 1.00 |
| **python** (线性扫描) | 23.16 ms | 0.02 s | ✅ 1.00 |

**加速比**：Rust HNSW 相对纯 Python 线性扫描 **≈ 230–469×**（近似检索），
SIMD 点积 `dot()` 实测 **≈ 2 ns/次**（100 万次级）。

> 数字来自 `tests/bench_real.py` 实跑，非估算。

---

## 🔌 多后端向量索引（召回可靠优先）

不绑定单一实现，**按召回可靠性与可用性自动选择**，全部可一键切换：

| 后端 | 说明 | recall@10(实测) |
|---|---|:---:|
| `faiss` | FAISS HNSW，工业级成熟实现 | **1.00** |
| `usearch` | 超轻量嵌入式 HNSW | **1.00** |
| `rust` | 自研 SIMD + HNSW（本内核） | 0.90（快，略近似） |
| `python` | 零依赖线性扫描（兜底） | 1.00 |

```python
idx = VectorIndex(dim=64, backend="auto")            # 自动选最优可用
idx = VectorIndex(dim=64, backend="faiss")           # 或显式指定
print(idx.impl)                                      # -> 当前后端
```

**设计哲学**：默认 `auto` 永远挑「召回最可靠且装得上」的后端；某个后端装不上自动降级，**功能不受影响**。

---

## 🎯 真实 Embedding 模型（可插拔，突破同义词天花板）

默认哈希嵌入零依赖但召回不到**零字面重叠的同义词**（卡顿↔延迟高、开通↔启用）。
可插拔接入 ONNX BGE 中文语义模型（`bge-base-zh-v1.5`，768 维，约 407 MB），
**脱离架构单独安装**，不改默认行为：

```bash
pip install onnxruntime tokenizers
python -m superbrain2 install-model      # 从开源仓库 GitHub Release 下载模型
```

```python
from superbrain2 import AgentConfig
cfg = AgentConfig(embedder="bge")        # 切到真实语义模型
```

内置**语义漂移守卫**：换嵌入器后旧向量不可比，`ensure_embedder_ready()` 自动
全量重嵌入修复。实测同义词 cos 从 0.21 → **0.87**。详见 [docs/EMBEDDER.md](docs/EMBEDDER.md)。

---

## 🧠 完整 Agent 内核（第一版全部能力）

SuperBrain 2.0 **不只是向量库** —— 它继承了完整的自主 Agent 架构：

```
认知层   需求驱动(PSI) · 情绪动力学(心境/衰减/感染/自豪愧疚)
         神经化学 · 元认知 · 自调参
记忆层   SQLite单文件 · FTS5全文 · 概念图 · 四路检索
         遗忘-重激活 · 30天浓缩 · 记忆缺口发现
人格层   大五维度四通道 · 进化开关 · 一句话风格 · 关系级风格
自主层   主动思考(想念/关心/好奇/修复) · 自主目标生成→采纳→推进 · tick()
对接层   SuperBrain门面 · 多后端向量(记忆检索加速) · 记忆生命周期
```

**门面能力一览**：`chat` / `remember` / `recall` / `dream` / `act` / `tick` /
`apply_style` / `humanize` / `generate_thoughts` / `drain_thoughts` /
`generate_goals` / `adopt_goals` / `set_personality` / `user_profile` / `save` / `load` …

---

## 📦 快速上手

```bash
git clone https://github.com/kejixiaoqi666/superbrain-2.0
cd superbrain-2.0
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop          # 编译 Rust 内核（无 Rust 环境也可，自动回退 Python）
```

```python
from superbrain2 import SuperBrain, VectorIndex, dot

# —— 纯向量索引 ——
idx = VectorIndex(dim=64, m=16, ef=32)   # 默认 auto 选最优后端
idx.add([0.1, -0.2, ...])                 # -> id
idx.search(query_vec, k=5)                # -> [(id, 分数)]

# —— SIMD 点积 ——
print(dot(vec_a, vec_b))                  # Rust SIMD 加速

# —— 完整 Agent ——
brain = SuperBrain.from_env()
print(brain.chat("你好"))
```

---

## 🗂 目录结构

```
src/           Rust 内核源码（lib.rs / dot.rs / hnsw.rs）
python/        Python 包（门面 + 多后端 + 双实现回退）
tests/         测试 + 基准（bench_real.py / verify_backends.py / smoke_full.py）
```

---

## 🗺 路线图

- [x] 完整 Agent 内核迁入（认知/情绪/记忆/人格/自主/表达）
- [x] Rust 内核：SIMD 点积 + HNSW
- [x] 多后端向量：faiss / usearch / rust / python 自动降级
- [x] HashingEmbedder 增强：同义词归一 + 去停用词（`AgentConfig(embed_enhance=True)`），
      零依赖突破部分同义词天花板（卡顿↔不稳定、开通↔启用），默认关=兼容旧库向量；
      附 `tests/test_embeddings.py` 质量回归
- [x] 真实 embedding 模型（可插拔，替换 HashingEmbedder）：`build_embedder("bge")` 接入
      ONNX BGE 中文语义模型（脱离架构单独安装，`python -m superbrain2 install-model`
      从开源仓库 GitHub Release 下载）；含语义漂移守卫 + 重嵌入迁移。详见 [docs/EMBEDDER.md](docs/EMBEDDER.md)
- [ ] 完整 PISA / DSL schema 演化
- [ ] 完整人性化表达元素库
- [ ] 记忆生命周期 / 人格 / 认知机制全量移植到 Rust 内核
- [ ] 三平台交叉编译 wheel（linux / mac / win）

---

## License

MIT

---

<div align="center">
<b>SuperBrain 2.0</b> · 情感认知内核 · 高性能向量 · 真实可靠<br/>
<a href="https://github.com/kejixiaoqi666/superbrain-2.0">GitHub</a>
</div>
