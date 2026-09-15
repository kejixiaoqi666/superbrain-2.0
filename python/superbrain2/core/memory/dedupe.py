"""记忆去重/合并：长期记忆膨胀治理。

用嵌入相似度找重复/近似记忆，合并成一条，避免记忆库无限膨胀。
内容冻结原则下：合并 = 标记旧节点被取代（supersede 边），写一条新合并记忆。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .node import MemoryNode
from .store import MemoryStore
from .embeddings import cosine

# 相似度阈值：超过则视为重复/近似，可合并
MERGE_SIMILARITY = 0.85


def find_duplicates(store: MemoryStore, threshold: float = MERGE_SIMILARITY
                    ) -> List[Tuple[MemoryNode, MemoryNode, float]]:
    """找出相似度超过阈值的记忆对（用于合并）。"""
    nodes = store.all_nodes()
    # 只比对有向量的节点
    vec_nodes = [(n, n.embedding) for n in nodes if n.embedding]
    pairs = []
    for i in range(len(vec_nodes)):
        ni, vi = vec_nodes[i]
        for j in range(i + 1, len(vec_nodes)):
            nj, vj = vec_nodes[j]
            s = cosine(vi, vj)
            if s >= threshold:
                pairs.append((ni, nj, s))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


def merge_pair(store: MemoryStore, a: MemoryNode, b: MemoryNode,
               keep: str = "newer") -> Optional[MemoryNode]:
    """合并一对近似记忆。

    keep: "newer"（保留较新的）| "first"（保留 a）
    返回保留的节点（合并后）。被合并的节点用 supersede 边标记。
    """
    if keep == "first":
        kept, dropped = a, b
    else:  # newer
        kept, dropped = (a, b) if a.created_at >= b.created_at else (b, a)

    # 合并 tags（去重）
    merged_tags = list(dict.fromkeys(kept.tags + dropped.tags))
    kept.tags = merged_tags
    # 更新保留节点
    store.add(kept)
    # 标记 dropped 被 kept 取代（内容冻结，不改写）
    store.supersede(dropped.node_id, kept.node_id, evidence="merge")
    return kept


def deduplicate(store: MemoryStore, threshold: float = MERGE_SIMILARITY,
                limit: int = 50) -> int:
    """批量去重：合并所有近似记忆对，返回合并的对数。"""
    merged = 0
    for a, b, _ in find_duplicates(store, threshold)[:limit]:
        merge_pair(store, a, b)
        merged += 1
    return merged
