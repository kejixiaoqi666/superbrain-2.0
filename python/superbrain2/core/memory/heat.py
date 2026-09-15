"""热度与遗忘：艾宾浩斯遗忘曲线 + 记忆价值（PowerMem 精髓）。

热度 = 遗忘曲线保留率 R = e^(-t/S) × 频次增益 × 置信度。
S 是记忆强度（retention_strength），随复习增强，遗忘变慢——这是
PowerMem 的艾宾浩斯遗忘曲线，比简单的 exp(-α·t) 更符合认知科学。
"""

from __future__ import annotations

import math
import time
from typing import List

from .node import MemoryNode
from .store import MemoryStore

THETA_HOT = 0.4
PRELOAD_BUDGET = 8


def retention(node: MemoryNode, now: float | None = None) -> float:
    """艾宾浩斯保留率 R = e^(-t/S)。

    t = 距上次访问的小时数；S = 记忆强度（retention_strength）。
    S 越大，遗忘越慢（复习会增强 S）。
    """
    t = ((now if now is not None else time.time()) - node.last_access) / 3600.0
    t = max(0.0, t)
    s = max(0.1, node.retention_strength)
    return math.exp(-t / s)


def heat(node: MemoryNode, now: float | None = None) -> float:
    """热度 = 保留率（艾宾浩斯）× 频次增益 × 置信度。"""
    frequency = 1.0 + math.log1p(node.access_count)
    return retention(node, now) * frequency * node.confidence


def preload_candidates(store: MemoryStore, k: int = PRELOAD_BUDGET) -> List[MemoryNode]:
    nodes = store.all_nodes()
    ranked = sorted(nodes, key=heat, reverse=True)
    return ranked[:k]


# ---------- 间隔重复（艾宾浩斯复习区间） ----------

REVIEW_INTERVALS_HOURS = [1.0, 9.0, 24.0, 48.0, 144.0, 336.0, 720.0]  # 1h/9h/1d/2d/6d/14d/30d


def due_for_review(node: MemoryNode, now: float | None = None) -> bool:
    """是否到复习时间（间隔重复：按 review_count 决定下次复习间隔）。"""
    idx = min(node.review_count, len(REVIEW_INTERVALS_HOURS) - 1)
    interval = REVIEW_INTERVALS_HOURS[idx] * 3600.0
    t = (now if now is not None else time.time()) - node.last_access
    return t >= interval


# ---------- 重要度分级（MemoryBridge sync_agent 精髓） ----------

IMPORTANT_CONFIDENCE = 0.8
IMPORTANT_ACCESS = 2
IMPORTANT_TAGS = ("important", "重要")


def is_important(node: MemoryNode) -> bool:
    if node.confidence >= IMPORTANT_CONFIDENCE:
        return True
    if node.access_count >= IMPORTANT_ACCESS:
        return True
    return any(t.lower() in IMPORTANT_TAGS for t in node.tags)
