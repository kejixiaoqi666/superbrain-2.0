"""Temporal versioning and contradiction resolution (zero dependencies)."""
from __future__ import annotations
import time
from typing import List, Tuple
from .node import MemoryNode, _gram_set
from .store import MemoryStore


def _sim(a: str, b: str) -> float:
    x, y = _gram_set(a), _gram_set(b)
    return len(x & y) / max(1, len(x | y))


def detect_conflicts(store: MemoryStore, node: MemoryNode,
                     threshold: float = 0.25) -> List[MemoryNode]:
    """检测与 node 冲突（同 kind、内容高度相似但表述不同）的已有记忆。

    commit_version 是写入操作（低频，非检索热路径），因此全量比对找冲突，
    不涉及检索性能。
    """
    out = []
    for n in store.all_nodes():
        if n.node_id == node.node_id:
            continue
        if n.kind != node.kind:
            continue
        s = _sim(n.content, node.content)
        # 冲突：相似但非完全相同（完全相同会被内容冻结去重）
        if 0.0 < s < 0.99 and s >= threshold:
            out.append(n)
    return out


def commit_version(store: MemoryStore, node: MemoryNode,
                   now: float | None = None, threshold: float = 0.25
                   ) -> Tuple[MemoryNode, List[str]]:
    """写入新版本，旧冲突记忆失效（set_valid_until）+ supersede 关联。

    返回 (新节点, 被失效的旧节点 id 列表)。内容冻结：旧内容不改写，
    只标 valid_until 失效时间 + supersede 边。
    """
    t = now or time.time()
    old = detect_conflicts(store, node, threshold)
    store.add(node)
    ids: List[str] = []
    for n in old:
        if n.created_at <= t and (not n.valid_until or n.valid_until > t):
            store.set_valid_until(n.node_id, t)
            store.supersede(n.node_id, node.node_id, evidence="temporal-conflict")
            ids.append(n.node_id)
    return node, ids
