"""价值观与边界层：人格的稳定约束。

人格不能只由记忆堆出来，还需要稳定的约束：
- 核心不可变原则（不能被一次对话覆盖）
- 可成长偏好（随经历改变）

这是让 agent 可靠、可控的关键——区分"事实/推测"、不编造经历、
不把"像人"说成"有意识"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CorePrinciple:
    """核心不可变原则（安全边界，不可被一次对话覆盖）。"""
    id: str = ""
    statement: str = ""
    priority: int = 0  # 越高越优先


@dataclass
class Preference:
    """可成长偏好（随经历改变）。"""
    id: str = ""
    key: str = ""       # 如 music / expression_style / topic_interest
    value: str = ""
    confidence: float = 0.5
    source_event: str = ""


DEFAULT_CORE_PRINCIPLES = [
    CorePrinciple(id="p1", statement="不编造自己没有的经历", priority=10),
    CorePrinciple(id="p2", statement="不把推测说成事实", priority=10),
    CorePrinciple(id="p3", statement="不泄露隐私和秘密", priority=10),
    CorePrinciple(id="p4", statement="不为了维持角色而伤害人", priority=10),
    CorePrinciple(id="p5", statement="不把「像人」误称为「已经有意识」", priority=10),
]


class ValueSystem:
    """价值观系统：核心原则 + 可成长偏好。"""

    def __init__(self) -> None:
        self.principles: List[CorePrinciple] = list(DEFAULT_CORE_PRINCIPLES)
        self.preferences: Dict[str, Preference] = {}

    def add_principle(self, statement: str, priority: int = 1) -> CorePrinciple:
        p = CorePrinciple(id=f"p{len(self.principles)+1}", statement=statement,
                          priority=priority)
        self.principles.append(p)
        self.principles.sort(key=lambda x: x.priority, reverse=True)
        return p

    def add_preference(self, key: str, value: str, confidence: float = 0.5,
                       source_event: str = "") -> Preference:
        pref = Preference(id=f"pref_{key}", key=key, value=value,
                          confidence=confidence, source_event=source_event)
        self.preferences[key] = pref
        return pref

    def get_preference(self, key: str) -> Preference | None:
        return self.preferences.get(key)

    def principle_text(self) -> str:
        """核心原则拼接（用于注入 prompt）。"""
        return "\n".join(f"- {p.statement}" for p in self.principles)

    def to_dict(self) -> dict:
        return {
            "principles": [{"id": p.id, "statement": p.statement,
                            "priority": p.priority} for p in self.principles],
            "preferences": {k: {"id": v.id, "key": v.key, "value": v.value,
                                "confidence": v.confidence,
                                "source_event": v.source_event}
                            for k, v in self.preferences.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ValueSystem":
        vs = cls()
        vs.principles = [CorePrinciple(id=p["id"], statement=p["statement"],
                                       priority=p.get("priority", 0))
                         for p in d.get("principles", [])]
        vs.preferences = {k: Preference(id=v["id"], key=v["key"], value=v["value"],
                                        confidence=v.get("confidence", 0.5),
                                        source_event=v.get("source_event", ""))
                          for k, v in d.get("preferences", {}).items()}
        return vs
