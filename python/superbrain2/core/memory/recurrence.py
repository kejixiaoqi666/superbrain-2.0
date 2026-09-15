"""Recurrence-gated consolidation (RecMem, Findings ACL 2026).

Keeps repeated raw interactions in a cheap buffer and only promotes a cluster
when recurrence is observed.  The gate is deterministic and dependency-free.
"""
from __future__ import annotations
from typing import List, Tuple
from .node import MemoryNode, _gram_set
from .store import MemoryStore

class RecurrenceGate:
    def __init__(self, store: MemoryStore, threshold: int = 2,
                 similarity_threshold: float = 0.7):
        self.store, self.threshold = store, max(2, threshold)
        self.similarity_threshold = similarity_threshold
        self._buffer: List[MemoryNode] = []

    def observe(self, node: MemoryNode) -> Tuple[bool, List[MemoryNode]]:
        """Buffer node; return (triggered, cluster), without duplicate writes.

        聚类用 Jaccard 相似度（重叠/并集），相对阈值严谨，避免长文本
        因绝对 n-gram 重叠数大而误聚语义无关的记忆。
        """
        self._buffer.append(node)
        grams = _gram_set(node.content)
        cluster = []
        for n in self._buffer:
            g2 = _gram_set(n.content)
            if not grams or not g2:
                continue
            jaccard = len(grams & g2) / len(grams | g2)
            if jaccard >= self.similarity_threshold:
                cluster.append(n)
        if len(cluster) < self.threshold:
            return False, cluster
        return True, cluster

    def consolidate(self, node: MemoryNode) -> MemoryNode:
        """Persist a compact semantic refinement derived from recurring facts."""
        triggered, cluster = self.observe(node)
        if not triggered:
            return node
        # preserve fine-grained repeated facts rather than lossy overwrite
        facts = []
        for n in cluster:
            for part in n.content.replace("。", "\n").splitlines():
                if part.strip() and part.strip() not in facts: facts.append(part.strip())
        out = MemoryNode(content="；".join(facts), tags=list(node.tags), scene=node.scene,
                         kind="fact", scope=node.scope, confidence=node.confidence,
                         retention_strength=node.retention_strength)
        self.store.add(out)
        self._buffer = [n for n in self._buffer if n not in cluster]
        return out
