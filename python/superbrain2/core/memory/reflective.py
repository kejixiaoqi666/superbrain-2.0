"""Opt-in reflective memory: evidence-backed lessons and retrieval utility updates.
Inspired by RMM (ACL 2025): prospective topic summaries and retrospective citation feedback.
"""
from __future__ import annotations
from typing import Iterable
from .node import MemoryNode
from .store import MemoryStore

class ReflectiveMemory:
    """Low-frequency reflection; every mutation is persisted through MemoryStore."""
    def __init__(self, store: MemoryStore):
        self.store = store

    def prospective(self, nodes: Iterable[MemoryNode], topic: str, *, kind="fact") -> MemoryNode:
        """Create a compact topic memory from real nodes (caller supplies summary text)."""
        items = list(nodes)
        text = topic + "\n" + "\n".join(n.content for n in items)
        n = MemoryNode(content=text, kind=kind, tags=["reflection", topic])
        self.store.add(n)
        for source in items:
            if source.node_id != n.node_id:
                self.store.add_edge(n.node_id, source.node_id, 1.0, "semantic", "prospective")
        return n

    def retrospective(self, cited_ids: Iterable[str], *, success: bool, rate: float=.1) -> int:
        """Persist citation feedback with bounded convergence, not unbounded accumulation.

        强度成功/失败向语义一致的目标收敛（成功→3.0 温和值，失败→0.1 下限），
        由 update_confidence 一次写回 SQLite（不重复写库）。
        """
        changed = 0
        for node_id in cited_ids:
            n = self.store.get(node_id)
            if n is None:
                continue
            conf_target = 1.0 if success else 0.0
            new_conf = n.confidence + rate * (conf_target - n.confidence)
            new_conf = max(0.0, min(1.0, new_conf))
            # 强度：成功向温和目标(3.0)收敛，失败降到下限 0.1
            ret_target = 3.0 if success else 0.1
            new_ret = n.retention_strength + rate * (ret_target - n.retention_strength)
            new_ret = max(0.1, min(10.0, new_ret))
            self.store.update_confidence(node_id, new_conf, new_ret)
            changed += 1
        return changed
