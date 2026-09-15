"""superbrain2 — Rust 高性能内核升级版。

双实现设计：
- 有编译好的 Rust 扩展（superbrain2._native）→ 用 HNSW/SIMD 高性能
- 没有 → 自动回退纯 Python（线性扫描 brute-force），功能一致

用法：
    from superbrain2 import HnswIndex, dot
    idx = HnswIndex(dim=64, m=16, ef=32)
    idx.add(vec)          # -> id
    idx.search(query, k)  # -> [(id, score), ...]
"""
from __future__ import annotations

from typing import List, Tuple, Sequence

__version__ = "0.1.0"

# ---------- 尝试加载 Rust 原生扩展 ----------
try:
    from superbrain2._native import HnswIndex as _NativeHnsw, simd_dot as _native_dot
    _HAS_NATIVE = True
    _IMPL = "rust"
except ImportError:  # 无 Rust 扩展 → 纯 Python 回退
    _HAS_NATIVE = False
    _IMPL = "python"

    def _native_dot(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    class _NativeHnsw:
        """纯 Python 回退：线性扫描 top-k（无 Rust 时保证可用）。"""

        def __init__(self, dim: int, m: int = 16, ef: int = 32, seed: int = 42):
            self.dim = dim
            self._vectors: List[List[float]] = []

        def add(self, vec: Sequence[float]) -> int:
            self._vectors.append(list(vec))
            return len(self._vectors) - 1

        def search(self, query: Sequence[float], k: int = 5, ef: int = 32) -> List[Tuple[int, float]]:
            q = list(query)
            scored = [(i, _native_dot(q, v)) for i, v in enumerate(self._vectors)]
            scored.sort(key=lambda t: t[1], reverse=True)
            return scored[:k]

        def __len__(self) -> int:
            return len(self._vectors)


# ---------- 统一导出（对上层透明，不 care 底层实现） ----------
class HnswIndex:
    """向量索引门面：内部自动选 Rust 或 Python 实现。"""

    __slots__ = ("_inner", "_impl")

    def __init__(self, dim: int, m: int = 16, ef: int = 32, seed: int = 42):
        self._inner = _NativeHnsw(dim, m, ef, seed)
        self._impl = _IMPL

    def add(self, vec: Sequence[float]) -> int:
        return self._inner.add(vec)

    def search(self, query: Sequence[float], k: int = 5, ef: int = 32) -> List[Tuple[int, float]]:
        return self._inner.search(list(query), k, ef)

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def impl(self) -> str:
        """当前实现：'rust'（HNSW/SIMD）或 'python'（线性回退）。"""
        return self._impl


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """SIMD 加速点积（无 Rust 时标量回退）。"""
    return _native_dot(list(a), list(b))
