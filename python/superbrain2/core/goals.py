"""目标系统：需求 → 目标 → 拆解（补「方向」）。

需求是内在驱动力（"我想变强"），目标系统把它转成具体可执行目标，
再拆解成子任务。这是让"自主"真正成立的承上启下环节。
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .cognition.needs import NeedType


class GoalStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class Task:
    id: str = ""
    description: str = ""
    status: str = "pending"
    result: str = ""
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "description": self.description,
                "status": self.status, "result": self.result,
                "dependencies": self.dependencies}


@dataclass
class Goal:
    id: str = ""
    description: str = ""
    # 由哪个需求驱动（需求→目标的转化记录）
    driven_by: str = ""  # NeedType 名
    priority: float = 0.0
    status: str = "pending"
    tasks: List[Task] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # 来源：need(即时需求派生) | autonomous(自主目标采纳)
    source: str = "need"
    source_id: str = ""  # 对应 AutonomousGoal.id，用于去重

    def to_dict(self) -> dict:
        return {"id": self.id, "description": self.description,
                "driven_by": self.driven_by, "priority": self.priority,
                "status": self.status,
                "tasks": [t.to_dict() for t in self.tasks],
                "created_at": self.created_at,
                "source": self.source, "source_id": self.source_id}


class GoalManager:
    """目标管理器：需求→目标→拆解→推进。"""

    def __init__(self) -> None:
        self._goals: List[Goal] = []

    def derive_goal(self, dominant_need: NeedType, drive: float,
                    capabilities: Optional[List[str]] = None) -> Optional[Goal]:
        """从主导需求派生目标。

        每个需求有对应的默认目标模板。drive 越高优先级越高。
        """
        if drive < 0.2:
            return None  # 驱动太弱，不主动设目标
        templates = {
            NeedType.CERTAINTY: "探索新信息，减少对当前环境的不确定性",
            NeedType.COMPETENCE: "练习技能，完成一个能提升能力的挑战",
            NeedType.AUTONOMY: "独立完成一个任务，证明自我决定能力",
            NeedType.RELATEDNESS: "与用户建立更有意义的连接",
            NeedType.ENERGY: "优化资源使用，提升效率",
        }
        desc = templates.get(dominant_need, "执行当前最高优先级任务")
        goal = Goal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            description=desc,
            driven_by=dominant_need.value,
            priority=drive,
        )
        self._goals.append(goal)
        return goal

    def decompose(self, goal: Goal, steps: List[str]) -> Goal:
        """拆解目标为子任务（steps 可由 LLM 生成或规则生成）。"""
        for i, step in enumerate(steps):
            task = Task(id=f"task_{uuid.uuid4().hex[:6]}", description=step)
            if i > 0:
                task.dependencies.append(goal.tasks[i - 1].id)
            goal.tasks.append(task)
        if goal.tasks:
            goal.status = GoalStatus.IN_PROGRESS.value
        return goal

    def next_task(self, goal: Goal) -> Optional[Task]:
        """返回下一个可执行任务（依赖已满足）。"""
        done = {t.id for t in goal.tasks if t.status == GoalStatus.COMPLETED.value}
        for t in goal.tasks:
            if t.status == "pending" and all(d in done for d in t.dependencies):
                return t
        return None

    def complete_task(self, goal: Goal, task_id: str, result: str = "") -> None:
        for t in goal.tasks:
            if t.id == task_id:
                t.status = GoalStatus.COMPLETED.value
                t.result = result
                break
        if all(t.status == GoalStatus.COMPLETED.value for t in goal.tasks):
            goal.status = GoalStatus.COMPLETED.value

    def active_goals(self) -> List[Goal]:
        return [g for g in self._goals if g.status in
                (GoalStatus.PENDING.value, GoalStatus.IN_PROGRESS.value)]

    def has_source(self, source_id: str) -> bool:
        """是否已采纳过某自主目标（去重，防重复采纳）。"""
        return any(g.source_id == source_id for g in self._goals)

    def adopt_autonomous(self, ag) -> Optional[Goal]:
        """把一条自主目标（中长期意图）采纳为目标系统里的可执行 Goal。

        鸭子类型接受 AutonomousGoal（id/content/horizon/driven_by/urgency）。
        返回新 Goal；已采纳过（source_id 重复）则返回 None。
        """
        if ag is None:
            return None
        sid = getattr(ag, "id", "") or ""
        content = getattr(ag, "content", "") or ""
        if not sid or not str(content).strip():
            return None  # 无 id 或无实质内容的脏目标不采纳
        if self.has_source(sid):
            return None
        goal = Goal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            description=content,
            driven_by=getattr(ag, "driven_by", "") or "autonomous",
            priority=getattr(ag, "urgency", 0.5) or 0.5,
            source="autonomous",
            source_id=sid,
        )
        self._goals.append(goal)
        return goal

    def summary(self) -> List[dict]:
        return [g.to_dict() for g in self._goals]
