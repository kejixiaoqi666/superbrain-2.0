"""全局工作空间 GWT（Global Workspace Theory 工程化）。

意识涌现的关键：多模块竞争注意力，只有最强的"广播"到全局工作区，
再驱动统一行动。替代超脑现在的"单线程顺序处理"。

模块（子系统）各自独立产生候选内容，带 urgency(紧迫度)+salience(显著性)，
竞争一个"广播权"——胜者进入工作区，controller(LLM/策略)据此统一行动。

参考：GWA (arXiv 2604.08206) — 竞争广播避免同质死锁
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Module:
    """一个认知子系统。"""
    name: str
    produce: Callable[[], dict]        # 返回候选 {content, urgency, salience}
    weight: float = 1.0                # 模块固有权重


@dataclass
class Bcast:
    """一次广播的全局工作区内容。"""
    module: str
    content: str
    urgency: float
    salience: float
    timestamp: float = field(default_factory=time.time)

    def score(self) -> float:
        return self.urgency * 2.0 + self.salience  # 竞争分值


class GlobalWorkspace:
    """全局工作空间：多模块竞争 → 广播 → 统一行动。"""

    def __init__(self, controller: Optional[Callable[[str], str]] = None) -> None:
        self.modules: Dict[str, Module] = {}
        self.controller = controller          # 统一行动决策器（可用 LLM）
        self.workspace: Optional[Bcast] = None
        self.history: List[Bcast] = []        # 广播历史（trace）
        self.gamma = 0.6                      # 注意力惯性：上次广播抑制（避免死锁）

    def register(self, name: str, produce: Callable[[], dict],
                 weight: float = 1.0) -> None:
        self.modules[name] = Module(name, produce, weight)

    def _apply_inertia(self, candidate: dict) -> float:
        """注意力惯性：与上次广播同源的候选被抑制（鼓励切换，避免死锁）。"""
        if self.workspace and candidate.get("module") == self.workspace.module:
            return -self.gamma
        return 0.0

    def step(self) -> Optional[Bcast]:
        """一次竞争广播：所有模块出候选 → 算分 → 最强进入工作区。"""
        contenders = []
        for name, mod in self.modules.items():
            try:
                cand = mod.produce() or {}
                urgency = float(cand.get("urgency", 0.0))
                salience = float(cand.get("salience", 0.0))
                content = str(cand.get("content", ""))
                if not content:
                    continue
            except Exception:
                continue
            inertia = self._apply_inertia({"module": name})
            score = urgency * 2.0 + salience * mod.weight + inertia
            contenders.append((score, Bcast(module=name, content=content,
                                            urgency=urgency, salience=salience)))
        if not contenders:
            return None
        contenders.sort(key=lambda t: t[0], reverse=True)
        winner = contenders[0][1]
        self.workspace = winner
        self.history.append(winner)
        return winner

    def act(self, query_hint: str = "") -> Optional[str]:
        """广播后统一行动：工作区内容交给 controller 决策，返回动作描述。"""
        bcast = self.step()
        if bcast is None:
            return None
        if self.controller:
            return self.controller(f"[工作区·{bcast.module}] {bcast.content}")
        return f"由模块 {bcast.module} 驱动：{bcast.content}"

    def clear(self) -> None:
        self.workspace = None

    def top_modules(self, k: int = 5) -> List[str]:
        """最近广播里最活跃的模块（诊断用）。"""
        from collections import Counter
        return [m for m, _ in Counter(b.module for b in self.history[-20:]).most_common(k)]