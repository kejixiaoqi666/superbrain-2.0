"""Opt-in episodic indexing: time-aware gists linked to facts (REMem, 2026)."""
from __future__ import annotations
from typing import Iterable, List
from .node import MemoryNode
from .store import MemoryStore

_GIST_INDEX_KEY = "episodic:gist_ids"


class EpisodicIndex:
    """时间感知的事件索引：gist 事件节点通过 episode-fact 边关联事实节点。

    gist 节点 id 维护在 meta（持久化索引），between() 走索引而非全表扫 kind。
    """

    def __init__(self, store: MemoryStore):
        self.store = store

    def _gist_ids(self) -> List[str]:
        raw = self.store.get_meta(_GIST_INDEX_KEY)
        if not raw:
            return []
        import json
        try:
            ids = json.loads(raw)
            return ids if isinstance(ids, list) else []
        except (ValueError, TypeError):
            return []

    def _set_gist_ids(self, ids: List[str]) -> None:
        import json
        self.store.set_meta(_GIST_INDEX_KEY, json.dumps(ids, ensure_ascii=False))

    def add_episode(self, gist: MemoryNode, facts: Iterable[MemoryNode] = ()) -> str:
        """写入 gist 事件 + 关联事实，gist id 入索引。"""
        gist.kind = "gist"
        self.store.add(gist)
        ids = self._gist_ids()
        if gist.node_id not in ids:
            ids.append(gist.node_id)
            self._set_gist_ids(ids)
        for fact in facts:
            self.store.add(fact)
            self.store.add_edge(gist.node_id, fact.node_id, 1.0,
                                "episode-fact", "episodic-index")
        return gist.node_id

    def facts_of(self, gist_id: str) -> List[MemoryNode]:
        """返回某个 gist 关联的所有事实节点。"""
        out = []
        for a, b, w, k, e in self.store.all_edges():
            if a == gist_id and k == "episode-fact":
                n = self.store.get(b)
                if n is not None:
                    out.append(n)
        return out

    def between(self, start: float = 0.0, end: float = 0.0) -> List[MemoryNode]:
        """按时间窗口查 gist 事件（走 meta 索引，不全表扫）。"""
        out = []
        for gid in self._gist_ids():
            n = self.store.get(gid)
            if n is None:
                continue
            if start and n.created_at < start:
                continue
            if end and n.created_at > end:
                continue
            out.append(n)
        return out
