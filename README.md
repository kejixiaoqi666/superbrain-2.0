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

## 当前阶段

- ✅ Rust 内核骨架：SIMD 点积（AVX2+FMA）+ HNSW 层次图索引
- ✅ Python 双实现回退（有 Rust 用，无则纯 Python）
- ✅ Rust 单元测试 + Python 测试全过，基准验证 469x

## 路线图（第二版完整实现，不精简）

- [ ] **真实 embedding 模型**（替换 HashingEmbedder，可插拔）
- [ ] **完整 PISA / DSL schema 演化**
- [ ] **完整人性化表达元素库**（覆盖每种关系的表达模板）
- [ ] 记忆生命周期 / 人格 / 认知机制全量移植到 Rust 内核
- [ ] 三平台交叉编译 wheel（linux/mac/win）

## License

MIT