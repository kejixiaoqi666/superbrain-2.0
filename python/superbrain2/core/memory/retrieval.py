"""检索层：四路混合检索 + RRF 融合 + 缺口发现 + 粒度自适应（MemGAS）。

四路：向量（余弦）+ 关键词（FTS5）+ 图谱（一跳邻居）+ 概念图（跨记忆）。
RRF 融合只看排名不看分数，天然奖励多路共识，无参数可调。
MemGAS：按查询熵选择检索粒度——模糊查询多路粗召回，精确查询单路精排。
"""

from __future__ import annotations

from typing import List, Tuple

from .node import MemoryNode
from .store import MemoryStore

RRF_K = 60
GRAPH_SEEDS = 2
GRAPH_FANOUT = 3
# 规模阈值：节点超过此数时，向量检索先 FTS5 粗筛候选集
CANDIDATE_THRESHOLD = 500
CANDIDATE_K = 100


def keyword_recall(store: MemoryStore, query: str) -> List[Tuple[MemoryNode, float]]:
    """关键词路：FTS5 全文检索（O(log n)，替代 n-gram 全表扫描）。"""
    nodes = store.keyword_search(query, k=20)
    return [(n, 1.0) for n in nodes]


def graph_recall(store: MemoryStore, seeds: List[MemoryNode]) -> List[Tuple[MemoryNode, float]]:
    """图谱路：种子的 SAN 一跳邻居按边权展开。"""
    seen = {s.node_id for s in seeds}
    scored = []
    for seed in seeds[:GRAPH_SEEDS]:
        for nbr, w in store.neighbors(seed.node_id)[:GRAPH_FANOUT]:
            if nbr.node_id in seen:
                continue
            seen.add(nbr.node_id)
            scored.append((nbr, w))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def concept_recall(store: MemoryStore, seeds: List[MemoryNode],
                   hops: int = 2) -> List[Tuple[MemoryNode, float]]:
    """概念图路（GraphRAG）：沿实体边跨记忆推理。

    记忆 → 实体 → 其它记忆（共享同一实体），找到向量/关键词检索
    找不到的跨记忆关联。entity 边是概念抽取建的桥梁。
    """
    seen = {s.node_id for s in seeds}
    scored: List[Tuple[MemoryNode, float]] = []
    frontier = list(seeds[:GRAPH_SEEDS])
    weight_decay = 1.0
    for _ in range(hops):
        nxt = []
        for node in frontier:
            for nbr, w in store.neighbors(node.node_id):
                if nbr.node_id in seen:
                    continue
                seen.add(nbr.node_id)
                # 只保留非实体节点（实体是桥梁，不是答案）
                if nbr.kind != "entity":
                    scored.append((nbr, w * weight_decay))
                nxt.append(nbr)
        frontier = nxt[:GRAPH_FANOUT]
        weight_decay *= 0.5
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def search(store: MemoryStore, query_vec: List[float], query_text: str,
           k: int = 5) -> List[Tuple[MemoryNode, float, str]]:
    """四路混合检索 + RRF，返回 (node, rrf_score, 命中路径)。

    四路：向量 + 关键词 + 图谱（一跳邻居）+ 概念图（实体边跨记忆推理）。
    """
    # 混合策略：大规模时 FTS5 粗筛候选集，小规模全量
    candidate_ids = None
    if store.count_nodes() > CANDIDATE_THRESHOLD:
        cand = store.keyword_search(query_text, k=CANDIDATE_K)
        candidate_ids = [n.node_id for n in cand]
    vec_hits = store.search(query_vec, k=k * 2, rel_floor=0.5,
                            candidate_ids=candidate_ids)
    kw_hits = keyword_recall(store, query_text)
    seeds = [n for n, _ in vec_hits[:GRAPH_SEEDS]] + [n for n, _ in kw_hits[:GRAPH_SEEDS]]
    named = [
        ("向量", vec_hits),
        ("关键词", kw_hits),
        ("图谱", graph_recall(store, seeds)),
        ("概念", concept_recall(store, seeds)),
    ]

    scores: dict = {}
    nodes: dict = {}
    why: dict = {}
    for name, route in named:
        for rank, (n, _) in enumerate(route):
            scores[n.node_id] = scores.get(n.node_id, 0.0) + 1.0 / (rank + RRF_K)
            nodes[n.node_id] = n
            seen = why.setdefault(n.node_id, [])
            if name not in seen:
                seen.append(name)

    fused = sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:k]
    hits = [(nodes[nid], s, "+".join(why.get(nid, []))) for nid, s in fused if nodes.get(nid)]

    if not hits:
        store.record_gap(query_text)  # 零命中记缺口，内容永远用户写
    return hits


# ---------- MemGAS：真正的多粒度熵路由器（跨粒度，只在小候选集上算） ----------

def query_entropy(query: str) -> float:
    """估算查询本身的不确定性（启发式，用于快速路径）。
    完整的多粒度熵见 granularity_entropy。"""
    q = query.strip()
    if not q:
        return 1.0
    length = len(q)
    precise_signals = sum(1 for c in q if c.isdigit() or c.isascii() and c.isalpha())
    precise_ratio = precise_signals / max(1, length)
    e = (1.0 - min(1.0, length / 30.0)) * 0.7 + (1.0 - precise_ratio) * 0.3
    return max(0.0, min(1.0, e))


def _shannon_entropy(scores: List[float], temperature: float = 1.0) -> float:
    """Shannon 熵：H = -Σ p_i log p_i，p_i = softmax(s_i/λ)。"""
    import math
    if not scores:
        return 1.0
    # softmax
    exps = [math.exp(s / temperature) for s in scores]
    z = sum(exps) or 1.0
    probs = [e / z for e in exps]
    h = -sum(p * math.log(p + 1e-12) for p in probs)
    return h


def granularity_weights(store: MemoryStore, query_vec: List[float],
                        query_text: str, candidates: List[MemoryNode]) -> dict:
    """跨粒度熵路由器（MemGAS 原文）。

    对候选集，按 4 个粒度各自算 query 到记忆的相似度分布熵 H^g，
    再按 w^g = (1/H^g) / Σ(1/H^g') 归一化——低熵（更确定）粒度权重高。

    只对 candidates（≤100）算，不扫全量，保证不损性能。
    """
    from .embeddings import cosine, dot_normalized

    if not candidates:
        return {}

    # 各粒度的相似度序列（候选集内）
    granularities = {
        "keyword": [],   # 关键词粒度：FTS5 命中
        "summary": [],   # 摘要粒度：首句
        "turn": [],      # 轮次粒度：整条内容
    }
    sim_fn = dot_normalized if store.normalized else cosine
    for n in candidates:
        # keyword 粒度：query 与内容 n-gram 重叠
        kw_sim = _keyword_sim(query_text, n.content)
        granularities["keyword"].append(kw_sim)
        # summary 粒度：query 与内容首句
        first_sent = n.content.split("。")[0]
        summary_sim = _keyword_sim(query_text, first_sent)
        granularities["summary"].append(summary_sim)
        # turn 粒度：向量相似度
        turn_sim = sim_fn(query_vec, n.embedding) if n.embedding else 0.0
        granularities["turn"].append(turn_sim)

    # 各粒度熵
    entropies = {}
    for g, scores in granularities.items():
        if scores:
            entropies[g] = _shannon_entropy(scores)

    if not entropies:
        return {}

    # w^g = (1/H^g) / Σ(1/H^g')
    inv = {g: 1.0 / max(h, 1e-6) for g, h in entropies.items()}
    z = sum(inv.values()) or 1.0
    return {g: v / z for g, v in inv.items()}


def _keyword_sim(a: str, b: str) -> float:
    """字符 n-gram 重叠相似度（轻量）。"""
    def grams(s, n=2):
        s = "".join(s.lower().split())
        return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else {s}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def adaptive_search(store: MemoryStore, query_vec: List[float], query_text: str,
                    k: int = 5, multi_granularity: bool = False) -> List[Tuple[MemoryNode, float, str]]:
    """自适应检索。multi_granularity=True 时启用完整多粒度熵路由（MemGAS）。

    默认 False：走原快速路径（不损性能）。
    开启时：仅对候选集算跨粒度熵，不扫全量。
    """
    if not multi_granularity:
        e = query_entropy(query_text)
        recall_k = max(k, int(k * (1.5 + e)))
        return search(store, query_vec, query_text, k=recall_k)[:k]

    # 多粒度路径：FTS5 粗筛候选 → 各粒度算相似度 → 按熵权重重排（MemGAS 落地）
    from .embeddings import cosine, dot_normalized
    cand = store.keyword_search(query_text, k=CANDIDATE_K)
    if not cand:
        return []
    weights = granularity_weights(store, query_vec, query_text, cand)
    sim_fn = dot_normalized if store.normalized else cosine
    scored = []
    for n in cand:
        kw = _keyword_sim(query_text, n.content)
        summary = _keyword_sim(query_text, n.content.split("。")[0])
        vec = sim_fn(query_vec, n.embedding) if n.embedding else 0.0
        # 低熵(更确定)粒度权重高 → 该粒度相似度贡献大；跨粒度加权融合
        gs = (weights.get("keyword", 0.0) * kw
              + weights.get("summary", 0.0) * summary
              + weights.get("turn", 0.0) * vec)
        scored.append((gs, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(n, s, "多粒度熵路由") for s, n in scored[:k]]
