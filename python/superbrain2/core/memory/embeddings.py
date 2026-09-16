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

    增强（可选，sync 时用增强词汇归一）：同义词归一到原型 token 再哈希，
    能在零依赖下突破部分"零字面重叠的同义词"天花板（如 卡顿↔不稳定）。
    strip_stop=True 时去掉中文高频虚词，减干扰。默认关闭以保持旧库向量一致。
    """

    _SYNONYMS = {
        "卡顿": ["卡顿", "不稳定", "很卡", "很慢", "慢", "延迟高", "卡死"],
        "查询": ["查", "查询", "查看", "查一下", "看看", "检索", "搜索"],
        "重启": ["重启", "重新启动", "重起", "reboot", "重启一下"],
        "启用": ["启用", "开通", "开启", "打开", "开启新"],
        "节点": ["节点", "服务器", "机器", "服务节点", "服务端"],
        "连接": ["连", "连接", "联网", "连上", "接入"],
        "失败": ["失败", "没成功", "上不去", "连不上", "不行", "挂了",
                 "出问题", "故障", "报错", "异常"],
        "退款": ["退款", "退单", "退回", "退钱", "撤销订单", "取消订单",
                "撤销购买", "退费"],
        "路由": ["路由", "线路", "转发规则", "转发", "代理规则", "新线路",
                "新节点", "通道", "入口"],
        "状态": ["状态", "情况", "状况", "运行情况", "health", "运行状态",
                "运行状况"],
    }
    _STOPWORDS = set("的了在是和你我他与它都也还不就从很能会把被对等但并且因为所以于是着过或又及既且然而")

    def __init__(self, dim: int = 256, ngram: int = 2,
                 enhance: bool = False, strip_stop: bool = False) -> None:
        self.dim = dim
        self.ngram = ngram
        self.enhance = enhance      # 默认 False=兼容旧库；True 用同义归一词典
        self.strip_stop = strip_stop  # 去掉中文高频虚词
        self.model = "hashing"
        self.revision = "v2-enhanced" if enhance else "v1"
        if enhance:
            self._fwd = {f: p for p, forms in self._SYNONYMS.items()
                         for f in forms}

    def _normalize(self, text: str) -> str:
        t = text.lower()
        if self.enhance:
            for f, p in self._fwd.items():
                t = t.replace(f, p)
        if self.strip_stop:
            t = "".join(ch for ch in t if ch not in self._STOPWORDS)
        return t

    def _grams(self, text: str) -> List[str]:
        t = "".join(self._normalize(text).split())
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
