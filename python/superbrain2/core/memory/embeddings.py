"""嵌入层：文本 → 语义向量 + 嵌入器一致性握手（MemoryBridge 精髓）。

默认零依赖哈希嵌入（跨进程/跨平台结果一致）；可换真实 embedding 模型，
但跨设备同步前必须用 embedder_identity 指纹握手（同模型同维度同 revision
才允许比对向量）。
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Protocol


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    # 内联点积（避免生成器函数调用开销）
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def dot_normalized(a: List[float], b: List[float]) -> float:
    """归一化向量的点积（已知 |a|=|b|=1 时等价 cosine，省两次开方）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def embedder_identity(emb) -> dict:
    """嵌入器自描述指纹：跨设备同步前用它做一致性握手。"""
    name = getattr(emb, "model", None) or f"hashing-{getattr(emb, 'dim', 0)}"
    dim = int(getattr(emb, "dim", 0) or 0)
    revision = str(getattr(emb, "revision", "") or "")
    basis = f"{type(emb).__name__}:{name}:{dim}" + (f":{revision}" if revision else "")
    fp = hashlib.blake2b(basis.encode("utf-8"), digest_size=8).hexdigest()
    return {"type": type(emb).__name__, "name": name, "dim": dim,
            "revision": revision, "fp": fp}


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]:  # pragma: no cover
        ...


class HashingEmbedder:
    """字符 n-gram 特征哈希嵌入：零依赖、跨进程/平台结果一致。

    关键：用 blake2b 哈希而非 Python 内置 hash()（后者有进程级随机化，
    同一文本不同进程嵌入不同，会导致记忆检索失效）。
    """

    def __init__(self, dim: int = 256, ngram: int = 2) -> None:
        self.dim = dim
        self.ngram = ngram

    def _grams(self, text: str) -> List[str]:
        t = "".join(text.lower().split())
        if not t:
            return []
        if len(t) <= self.ngram:
            return [t]
        return [t[i:i + self.ngram] for i in range(len(t) - self.ngram + 1)]

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for g in self._grams(text):
            h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
