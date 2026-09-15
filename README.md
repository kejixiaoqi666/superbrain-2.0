# superbrain2

**SuperBrain 2.0 — Rust 高性能内核升级版。**

超脑 v2.0（纯 Python 基线）的 Rust 混合内核版本：用 **PyO3 + SIMD + HNSW** 把向量检索/索引的性能热点做到极限，同时保留"无 Rust 时回退纯 Python"的可移植性。

## 架构

```
Python (superbrain2/)      Rust 内核 (superbrain2._native)
├── HnswIndex 门面  ──try─▶  ├── HNSW 层次图向量索引（百万级亚毫秒）
├── dot()          ──try─▶  ├── SIMD 点积（AVX2 FMA，快 10-50x）
└── 无 Rust → 纯 Python 回退   └── PyO3 绑定
```

**双实现**：有编译好的 Rust 扩展用 HNSW/SIMD；没有 → 自动回退纯 Python 线性扫描，功能一致。

## 快速上手

```bash
pip install maturin
cd superbrain2 && maturin develop     # 开发安装（编译 Rust 内核）
```

```python
from superbrain2 import VectorIndex, dot

# 多后端向量索引：默认 auto 自动选最优（faiss → usearch → rust → python）
idx = VectorIndex(dim=64, m=16, ef=32)
# 也可显式指定后端：VectorIndex(dim=64, backend="faiss"/"usearch"/"rust"/"python")
idx.add([...])                          # -> id
r = idx.search(query_vec, k=5)          # -> [(id, 分数), ...] 分数越大越相似

s = dot(vec_a, vec_b)                   # SIMD 加速点积
```

### 多后端（召回可靠优先）
| 后端 | 说明 | recall@10(2000条,dim64,实测) |
|---|---|---|
| `faiss` | FAISS HNSW，成熟可靠 | **1.00** |
| `usearch` | 超轻量嵌入式 | **1.00** |
| `rust` | 自研 SIMD+HNSW | 0.90（快但召回略差）|
| `python` | 零依赖线性扫描 | 1.00 |

## 目录

```
src/           Rust 内核源码（lib.rs / dot.rs / hnsw.rs）
python/        Python 包（facade + 双实现回退）
tests/         测试（rust 与 python 实现都覆盖）+ bench.py 基准
```

## 性能（已实测）

| 指标 | 纯 Python 线性扫描 | Rust HNSW |
|---|---|---|
| 1 万条向量检索（dim=64, top-5）| ~56.7 ms/查询 | **~0.12 ms/查询** |
| 加速比 | — | **≈ 469x** |
| HNSW 构建 1 万条 | — | ~1.4s |
| 正确性 | — | top1 匹配真最近 ✅ |

## 完整 Agent 能力（SuperBrain 内核）

superbrain2 不只是向量库——已继承第一版超脑的**完整 Agent 内核**，叠加 Rust/多后端向量加速：

```
认知层   需求驱动(PSI) + 情绪动力学(心境/衰减/感染/自豪愧疚) + 神经化学 + 元认知
记忆层   SQLite单文件 + FTS5全文 + 概念图 + 四路检索 + 遗忘-重激活 + 30天浓缩
人格层   大五维度四通道 + 进化开关 + 一句话风格 + 关系级风格
自主层   主动思考(想念/关心/好奇/修复) + 自主目标 + tick()
对接层   SuperBrain门面 + 向量检索走多后端(faiss/usearch/rust)
```

```python
from superbrain2 import SuperBrain, VectorIndex

brain = SuperBrain.from_env()        # 完整 Agent（第一版全部能力）
reply = brain.chat("你好")
brain.remember("用户喜欢茶"); brain.recall("喜欢什么")
brain.apply_style("可爱"); brain.dream(); brain.tick()

idx = VectorIndex(dim=64, backend="auto")   # 多后端向量（记忆检索加速）
```

## 当前阶段

- ✅ **完整 Agent 内核已迁入**（第一版全部能力：认知/情绪/记忆/人格/自主/表达）
- ✅ Rust 内核：SIMD 点积 + HNSW
- ✅ **多后端向量**：faiss / usearch / rust / python 自动降级，召回可靠优先
- ✅ 测试 + 冒烟验证完整能力
- ⬜ 真实 embedding / PISA 强化 / 完整表达库（路线图）

## 路线图（第二版完整实现，不精简）

- [ ] **真实 embedding 模型**（替换 HashingEmbedder，可插拔）
- [ ] **完整 PISA / DSL schema 演化**
- [ ] **完整人性化表达元素库**（覆盖每种关系的表达模板）
- [ ] 记忆生命周期 / 人格 / 认知机制全量移植到 Rust 内核
- [ ] 三平台交叉编译 wheel（linux/mac/win）

## License

MIT