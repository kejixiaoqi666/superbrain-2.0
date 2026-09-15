"""Opt-in adaptive memory control: utility-based forgetting and test-time feedback.

Inspired by MemoryAgentBench (2025) selective forgetting/test-time learning and
SF-AMS (2026 preprint): utility combines confidence, access recency, redundancy,
and importance; updates are persisted through MemoryStore.
"""
from __future__ import annotations
import math, time
from typing import List
from .node import MemoryNode
from .store import MemoryStore


def utility(node: MemoryNode, now: float | None = None, redundancy: float = 0.0) -> float:
    now = now or time.time()
    age = max(0.0, now - (node.last_access or node.created_at))
    recency = math.exp(-age / (86400.0 * max(.25, node.retention_strength)))
    return max(0.0, min(1.0, .55*node.confidence + .25*recency + .20*(1.0-redundancy)))


def schedule_forgetting(store: MemoryStore, budget: int, floor: float = .15) -> List[str]:
    """Keep the highest-utility memories; demote low utility ones durably."""
    nodes = store.all_nodes()
    ranked = sorted(nodes, key=utility, reverse=True)
    changed=[]
    for n in ranked[budget:]:
        if utility(n) < floor and n.tier != "archival":
            store.promote(n.node_id, "archival")
            changed.append(n.node_id)
    return changed


def test_time_update(store: MemoryStore, node_id: str, success: bool,
                     surprise: float = 0.0, learning_rate: float = .2) -> None:
    """Deployment-time feedback update; surprise lowers retention, success reinforces.

    方向（已修正）：
    - 置信度向"观察结果"收敛（success→1 方向，surprise 大→下调）
    - retention：surprise 大→下降（记忆被动摇）；success→向温和目标收敛（EMA），
      不无限累加（避免长期所有成功记忆都顶到上限）。
    """
    n = store.get(node_id)
    if n is None:
        return
    error = max(0.0, min(1.0, surprise))

    # 置信度：向目标收敛（success 目标=1，否则目标=当前-误差惩罚）
    target_conf = (1.0 if success else n.confidence) - error * 0.5
    confidence = n.confidence + learning_rate * (target_conf - n.confidence)
    confidence = max(0.0, min(1.0, confidence))

    # retention：surprise 大→动摇下降；success→向温和目标(如 3.0)收敛，不无限累加
    strength = n.retention_strength * (1.0 - learning_rate * error)
    if success:
        target_strength = 3.0  # 温和上限，避免所有成功记忆都顶到 10
        strength = strength + learning_rate * (target_strength - strength)
    strength = max(0.1, min(10.0, strength))

    store.update_confidence(node_id, confidence, strength)
