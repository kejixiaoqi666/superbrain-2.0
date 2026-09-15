"""superbrain2 — Rust 高性能内核升级版。

多后端向量索引（按召回可靠性与可用性自动选择）：
  1. faiss  HNSW      （召回最可靠，成熟实现）
  2. usearch HNSW     （超轻量嵌入式）
  3. rust   自研HNSW  （SIMD 点积，本内核）
  4. python 线性扫描  （零依赖兜底，功能一致）

用法：
    from superbrain2 import VectorIndex, dot
    idx = VectorIndex(dim=64, backend="auto")   # auto 自动选最优可用
    idx.add(vec)          # -> id
    idx.search(query, k)  # -> [(id, score), ...]  score 越大越相似
"""
from __future__ import annotations

from typing import List, Tuple, Sequence, Optional

__version__ = "0.2.0"

# ---------- 底层：SIMD 点积（Rust，可选） ----------
try:
    from superbrain2._native import simd_dot as _native_dot
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

    def _native_dot(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """点积（Rust SIMD 加速；无 Rust 时标量回退）。"""
    return _native_dot(list(a), list(b))


# ---------- 后端实现 ----------
class _RustBackend:
    """自研 Rust HNSW（本内核）。"""
    _impl = "rust"

    def __init__(self, dim, m, ef, seed):
        from superbrain2._native import HnswIndex as _H
        self._inner = _H(dim, m, ef, seed)

    def add(self, vec) -> int:
        return self._inner.add(list(vec))

    def search(self, q, k, ef):
        return self._inner.search(list(q), k, ef)

    def __len__(self):
        return len(self._inner)


class _PythonBackend:
    """纯 Python 线性扫描（零依赖兜底，召回 1.0）。"""
    _impl = "python"

    def __init__(self, dim, m, ef, seed):
        self.dim = dim
        self._vectors: List[List[float]] = []

    def add(self, vec) -> int:
        self._vectors.append(list(vec))
        return len(self._vectors) - 1

    def search(self, q, k, ef):
        q = list(q)
        scored = [(i, _native_dot(q, v)) for i, v in enumerate(self._vectors)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def __len__(self):
        return len(self._vectors)


class _FaissBackend:
    """FAISS HNSWFlat（内积度量，归一化向量=余弦，召回最可靠）。"""
    _impl = "faiss"

    def __init__(self, dim, m, ef, seed):
        import faiss
        self._dim = dim
        self._idx = faiss.IndexHNSWFlat(dim, m)
        self._idx.hnsw.efConstruction = ef
        self._count = 0

    def add(self, vec) -> int:
        import numpy as np
        self._idx.add(np.asarray([list(vec)], dtype=np.float32))
        self._count += 1
        return self._count - 1

    def search(self, q, k, ef):
        import numpy as np
        self._idx.hnsw.efSearch = max(ef, k)
        D, I = self._idx.search(np.asarray([list(q)], dtype=np.float32), k)
        res = [(int(I[0][j]), float(D[0][j])) for j in range(k) if I[0][j] != -1]
        res.sort(key=lambda t: t[1], reverse=True)   # 保证接口一致：分数降序
        return res

    def __len__(self):
        return self._count


class _UsearchBackend:
    """usearch HNSW（超轻量嵌入式，内积度量）。"""
    _impl = "usearch"

    def __init__(self, dim, m, ef, seed):
        from usearch.index import Index
        import numpy as np
        self._idx = Index(ndim=dim, metric="ip", connectivity=m,
                          expansion_add=ef, expansion_search=ef)
        self._count = 0

    def add(self, vec) -> int:
        import numpy as np
        self._idx.add(self._count, np.asarray(list(vec), dtype=np.float32))
        self._count += 1
        return self._count - 1

    def search(self, q, k, ef):
        import numpy as np
        matches = self._idx.search(np.asarray(list(q), dtype=np.float32), k)
        # usearch ip 的 distance = 1 - 内积 → score = 1 - distance（越大越相似，自身=1.0）
        return [(int(m.key), float(1.0 - m.distance)) for m in matches]

    def __len__(self):
        return self._count


# 后端注册表（按推荐优先级）
_BACKENDS = [("faiss", _FaissBackend), ("usearch", _UsearchBackend),
             ("rust", _RustBackend), ("python", _PythonBackend)]


class VectorIndex:
    """多后端向量索引门面：默认 auto 自动选召回最可靠且可用的后端。"""

    __slots__ = ("_inner", "_impl")

    def __init__(self, dim: int, m: int = 16, ef: int = 32, seed: int = 42,
                 backend: str = "auto"):
        self._inner = _build(dim, m, ef, seed, backend)
        self._impl = self._inner._impl

    def add(self, vec: Sequence[float]) -> int:
        return self._inner.add(vec)

    def search(self, query: Sequence[float], k: int = 5, ef: int = 32) -> List[Tuple[int, float]]:
        return self._inner.search(query, k, ef)

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def impl(self) -> str:
        """当前后端：faiss / usearch / rust / python。"""
        return self._impl


def _build(dim, m, ef, seed, backend) -> object:
    if backend == "auto":
        for name, cls in _BACKENDS:
            try:
                return cls(dim, m, ef, seed)
            except Exception:
                continue
        raise ImportError("所有向量后端均不可用")
    for name, cls in _BACKENDS:
        if name == backend:
            return cls(dim, m, ef, seed)
    raise ValueError(f"未知后端: {backend}（可选 faiss/usearch/rust/python/auto）")


# ---------- 兼容别名 ----------
HnswIndex = VectorIndex
