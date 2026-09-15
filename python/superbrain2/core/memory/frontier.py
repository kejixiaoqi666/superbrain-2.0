"""Opt-in frontier memory mechanisms from 2026 agent-memory research.

Implements engram maturation (dual-trace activation) and multi-cue retrieval.
State is persisted through MemoryStore's SQLite-backed meta API; the default
retrieval path is unchanged.
"""
from __future__ import annotations
import math, time
from typing import Iterable, List, Tuple, Optional
from .node import MemoryNode, _gram_set
from .store import MemoryStore

class EngramMaturation:
    """Sigmoid activation for a silent semantic engram (arXiv:2605.08538)."""
    def __init__(self, store: MemoryStore, half_life_hours: float = 168.0,
                 slope_hours: float = 48.0, threshold: float = .5):
        self.store, self.half_life, self.slope, self.threshold = store, half_life_hours, slope_hours, threshold
    def _key(self, node_id): return "engram:" + node_id
    def activation(self, node: MemoryNode, now: Optional[float] = None) -> float:
        raw = self.store.get_meta(self._key(node.node_id))
        started = float(raw) if raw else node.created_at
        age = max(0.0, (now or time.time()) - started) / 3600.0
        return 1.0 / (1.0 + math.exp(-(age - self.half_life) / self.slope))
    def register(self, node_id: str, created_at: Optional[float] = None) -> None:
        self.store.set_meta(self._key(node_id), str(created_at or time.time()))
    def mature(self, nodes: Optional[Iterable[MemoryNode]] = None, now: Optional[float] = None) -> List[str]:
        out=[]
        for n in (list(nodes) if nodes is not None else self.store.all_nodes()):
            if self.activation(n, now) >= self.threshold: out.append(n.node_id)
        return out
    def is_explicit(self, node: MemoryNode, now: Optional[float] = None) -> bool:
        return self.activation(node, now) >= self.threshold


def multi_cue_search(store: MemoryStore, query_vec: List[float], query_text: str,
                     k: int = 5, context: str = "", tags: Optional[Iterable[str]] = None,
                     maturation: Optional[EngramMaturation] = None) -> List[Tuple[MemoryNode, float, str]]:
    """Opt-in hybrid retrieval: vector + character n-gram + context + tag cues.

    Scores are normalized per cue then fused; explanation records contributing cues.
    性能：大库时先 FTS5 粗筛候选集，避免全表扫描（复用 CANDIDATE_THRESHOLD）。
    """
    from .retrieval import CANDIDATE_THRESHOLD, CANDIDATE_K

    qgrams = _gram_set(query_text); cgrams = _gram_set(context) if context else set()
    wanted = set(tags or ())
    # 候选集预筛：大库只对 FTS5 粗筛出的候选算，不扫全量
    nodes = store.all_nodes()
    if store.count_nodes() > CANDIDATE_THRESHOLD:
        cand = store.keyword_search(query_text, k=CANDIDATE_K)
        if cand:
            nodes = cand
    rows=[]
    for n in nodes:
        vec = store._sim(query_vec, n.embedding) if query_vec and n.embedding else 0.0
        kw = len(qgrams & _gram_set(n.content)) / max(1, len(qgrams | _gram_set(n.content)))
        ctx = len(cgrams & _gram_set(n.content)) / max(1, len(cgrams | _gram_set(n.content))) if cgrams else 0.0
        tag = len(wanted & set(n.tags)) / max(1, len(wanted)) if wanted else 0.0
        activation = maturation.activation(n) if maturation else 1.0
        score = (.45 * max(0.0, vec) + .30 * kw + .15 * ctx + .10 * tag) * (0.5 + 0.5 * activation)
        if score > 0:
            why=[name for name,val in (("vector",vec),("keyword",kw),("context",ctx),("tag",tag)) if val > 0]
            rows.append((n, score, "+".join(why)))
    rows.sort(key=lambda x: (-x[1], x[0].node_id))
    for n,_,_ in rows[:k]: store.touch(n.node_id)
    return rows[:k]
