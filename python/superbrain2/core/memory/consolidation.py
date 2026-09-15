"""Opt-in offline memory consolidation inspired by SCM/ACE (2025-26).

Never runs on the default retrieval path: callers explicitly invoke ``sleep``.
Uses only a supplied candidate set (or all nodes when offline maintenance is
requested), preserves frozen content, and records pruning as edges.
"""
from __future__ import annotations
import math, time
from typing import Iterable
from .node import MemoryNode
from .store import MemoryStore

class Consolidator:
    def __init__(self, store: MemoryStore):
        self.store = store

    @staticmethod
    def value(node: MemoryNode, now: float | None = None) -> float:
        now = now or time.time()
        age = max(0.0, now - (node.last_access or node.created_at))
        retention = math.exp(-age / max(node.retention_strength * 3600.0, 1.0))
        return retention * (1.0 + math.log1p(node.access_count)) * node.confidence

    def sleep(self, candidates: Iterable[MemoryNode] | None = None,
              max_items: int | None = None, forget_below: float = 0.0) -> dict:
        """NREM-like strengthening and active forgetting on an explicit batch.

        Returns ids, does not delete content: forgotten traces move to archival
        tier (provenance kept via event_log, not self-loop edges).
        """
        nodes = list(candidates) if candidates is not None else self.store.all_nodes()
        ranked = sorted(((self.value(n), n) for n in nodes), reverse=True,
                        key=lambda x: x[0])
        if max_items is not None:
            ranked = ranked[:max_items]
        forgotten = []
        strengthened = []
        for score, n in ranked:
            if score < forget_below and n.tier != "core":
                # 主动遗忘：降级 tier 到 archival（不删内容，不造自环边）
                self.store.promote(n.node_id, "archival")
                forgotten.append(n.node_id)
            elif score > 0:
                # 巩固：高价值晋升 core，普通保留 recall（复用已有 promote）
                self.store.promote(n.node_id, "core" if score >= 1.5 else "recall")
                strengthened.append(n.node_id)
        return {"considered": len(ranked), "strengthened": strengthened, "forgotten": forgotten}
