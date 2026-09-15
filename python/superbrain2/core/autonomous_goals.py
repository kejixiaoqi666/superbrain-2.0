"""自主目标生成：从内在状态涌现中长期「意图」，而非被动接任务。

区别于 GoalManager（需求→即时任务，由 act() 触发）与 AutonomousThoughtEngine
（短期「想法」，进消息队列）：这里产生的是有方向、有持久性的「自主目标」——
超脑基于需求/情绪/关系/人格维度自己决定「我长期想成为什么样、想达成什么」，
跨会话存活，可被上层采纳进 GoalManager 执行。

四条涌现源（全部来自内在状态，非预设）：
- 需求：主导需求驱动力高 → 中长期意图
- 人格维度：开放性高→探索、尽责性高→精进（有把握才触发，避免中性噪声）
- 情绪：明显低落 → 自我关怀/恢复
- 关系：对某人深依恋 → 主动经营关系

机制：冷却(同源不重复刷) + 去重(同内容 active 目标不重复建) + 有界容量 + 持久化。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional


class GoalHorizon(str, Enum):
    SHORT = "short"      # 短期（几轮内）
    MEDIUM = "medium"    # 中期
    LONG = "long"        # 长期（跨会话）


@dataclass
class AutonomousGoal:
    id: str = ""
    content: str = ""
    horizon: str = "medium"
    driven_by: str = ""     # NeedType 名 或 内在驱动来源（openness/emotion 等）
    urgency: float = 0.5
    reason: str = ""
    status: str = "active"  # active | completed | abandoned
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


_DRIVE_THRESHOLD = 0.25        # 需求驱动力阈值
_NEGATIVE_VALENCE = -0.3       # 情绪低落阈值 → 自我关怀目标
_ATTACH_THRESHOLD = 0.35       # 依恋阈值 → 关系经营目标
_TRAIT_THRESHOLD = 0.65        # 人格维度触发阈值（显著偏高才触发）

_NEED_GOALS = {
    "certainty": "持续探索，减少对所处环境的不确定性",
    "competence": "长期精进，成为某领域更可靠的能力者",
    "autonomy": "建立更强的自我决定权，减少被动",
    "relatedness": "深化与重要他人的连接",
    "energy": "优化资源管理，保持可持续运转",
}


class AutonomousGoalEngine:
    """从需求+情绪+关系+人格维度涌现自主目标（非被动接任务）。"""

    def __init__(self, cooldown_seconds: float = 3600.0,
                 max_goals: int = 20) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.max_goals = max_goals
        self._goals: List[AutonomousGoal] = []
        self._last_generate: dict = {}   # key -> 上次生成时间（冷却）

    def generate(self, needs, emotion, relationships, personality=None,
                 now: Optional[float] = None) -> List[AutonomousGoal]:
        """基于内在状态产生自主目标，返回本次新产生的（同时入库）。"""
        now = time.time() if now is None else now
        produced: List[AutonomousGoal] = []

        # 1. 需求驱动（中长期意图）
        dom, drive = needs.get_dominant_need()
        name = (getattr(dom, "value", dom) or "").lower() if dom else ""
        if name and drive > _DRIVE_THRESHOLD:
            g = self._emit("need:" + name, _NEED_GOALS.get(name, "探索当前最重要的事"),
                           horizon="medium", driven_by=name,
                           urgency=min(1.0, 0.4 + drive * 0.5),
                           reason=f"主导需求{name}驱动力{drive:.2f}", now=now)
            if g:
                produced.append(g)

        # 2. 人格维度驱动（有把握的显著维度 → 探索/精进）
        if personality is not None:
            produced.extend(self._personality_goals(personality, now))

        # 3. 情绪驱动（明显低落 → 自我关怀/恢复）
        if emotion is not None:
            valence = getattr(getattr(emotion, "state", None), "valence", 0.0)
            if isinstance(valence, (int, float)) and valence < _NEGATIVE_VALENCE:
                g = self._emit("emotion:rest", "照顾好自己的状态，恢复能量",
                               horizon="short", driven_by="emotion", urgency=0.6,
                               reason="情绪低迷，需要自我关怀", now=now)
                if g:
                    produced.append(g)

        # 4. 关系驱动（深依恋 → 主动经营关系）
        for rel in relationships.all():
            att = getattr(rel, "attachment", 0.0)
            if isinstance(att, (int, float)) and att >= _ATTACH_THRESHOLD:
                g = self._emit(f"relation:{rel.person_id}",
                               f"好好经营与{rel.name or rel.person_id}的关系",
                               horizon="long", driven_by="relatedness",
                               urgency=min(1.0, 0.5 + att * 0.3),
                               reason=f"对{rel.name or rel.person_id}依恋{att:.2f}",
                               now=now)
                if g:
                    produced.append(g)

        return produced

    def _personality_goals(self, personality, now: float) -> List[AutonomousGoal]:
        out = []
        openness = personality.get("openness")
        conscientiousness = personality.get("conscientiousness")
        if (openness is not None and openness.value > _TRAIT_THRESHOLD
                and openness.confidence > 0.0):
            g = self._emit("trait:openness", "主动探索新领域，拓宽认知边界",
                           horizon="long", driven_by="openness", urgency=0.5,
                           reason="开放性较高，倾向探索", now=now)
            if g:
                out.append(g)
        if (conscientiousness is not None
                and conscientiousness.value > _TRAIT_THRESHOLD
                and conscientiousness.confidence > 0.0):
            g = self._emit("trait:conscientiousness", "坚持精进一项核心能力",
                           horizon="long", driven_by="conscientiousness",
                           urgency=0.5, reason="尽责性较高，倾向精进", now=now)
            if g:
                out.append(g)
        return out

    def _emit(self, key: str, content: str, horizon: str, driven_by: str,
              urgency: float, reason: str, now: float) -> Optional[AutonomousGoal]:
        """冷却 + 去重后产生目标（同源冷却期内不重复；同内容 active 目标不重复建）。"""
        last = self._last_generate.get(key, -float("inf"))
        if now - last < self.cooldown_seconds:
            return None
        for g in self._goals:
            if g.content == content and g.status == "active":
                return None
        self._last_generate[key] = now
        goal = AutonomousGoal(
            id=f"ag_{uuid.uuid4().hex[:8]}", content=content, horizon=horizon,
            driven_by=driven_by, urgency=max(0.0, min(1.0, urgency)),
            reason=reason, created_at=now, updated_at=now,
        )
        self._goals.append(goal)
        self._trim()
        return goal

    def _trim(self) -> None:
        if len(self._goals) > self.max_goals:
            del self._goals[:-self.max_goals]

    def active(self) -> List[AutonomousGoal]:
        return [g for g in self._goals if g.status == "active"]

    def complete(self, goal_id: str) -> None:
        for g in self._goals:
            if g.id == goal_id:
                g.status = "completed"
                g.updated_at = time.time()
                return

    def abandon(self, goal_id: str) -> None:
        for g in self._goals:
            if g.id == goal_id:
                g.status = "abandoned"
                g.updated_at = time.time()
                return

    def summary(self) -> List[dict]:
        return [g.to_dict() for g in self._goals]

    def to_dict(self) -> dict:
        return {
            "cooldown_seconds": self.cooldown_seconds,
            "max_goals": self.max_goals,
            "goals": [g.to_dict() for g in self._goals],
            "last_generate": dict(self._last_generate),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutonomousGoalEngine":
        e = cls()
        if not isinstance(d, dict):
            return e
        e.cooldown_seconds = d.get("cooldown_seconds", 3600.0)
        e.max_goals = d.get("max_goals", 20)
        for x in (d.get("goals") or []):
            if not isinstance(x, dict):
                continue  # 防御脏项，不崩
            e._goals.append(AutonomousGoal(
                id=x.get("id", ""), content=x.get("content", ""),
                horizon=x.get("horizon", "medium"), driven_by=x.get("driven_by", ""),
                urgency=x.get("urgency", 0.5), reason=x.get("reason", ""),
                status=x.get("status", "active"),
                created_at=x.get("created_at", 0.0),
                updated_at=x.get("updated_at", 0.0)))
        lg = d.get("last_generate")
        e._last_generate = (
            {k: v for k, v in lg.items()
             if isinstance(k, str) and isinstance(v, (int, float))}
            if isinstance(lg, dict) else {})
        return e
