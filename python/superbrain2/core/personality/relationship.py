"""关系模型：对「人」的长期关系记忆。

人格不是孤立产生的，很大部分来自长期关系。对同一个人，系统形成：
- 记得对方过去说过的话、习惯、敏感话题
- 能区分玩笑、安慰、质疑、命令
- 长期互动后形成独特称呼与表达方式

每条关系记忆都带来源和置信度（防把推测写成事实）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Relationship:
    """对某个人的关系模型。"""
    person_id: str = ""          # 对方唯一标识（如 telegram user id）
    name: str = ""               # 如何称呼对方
    trust: float = 0.5           # 信任度 0-1
    familiarity: float = 0.5     # 熟悉度 0-1
    shared_events: List[str] = field(default_factory=list)  # 共同经历事件 id
    boundaries: List[str] = field(default_factory=list)     # 对方边界/敏感点
    unresolved_conflicts: List[str] = field(default_factory=list)
    expectations: List[str] = field(default_factory=list)   # 对方对我的期望
    nickname: str = ""           # 形成的独特称呼
    notes: List[dict] = field(default_factory=list)  # 关系记忆，每条带来源+置信度
    update_history: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    # 自主关系定位：超脑在相处中自己「决定」成为对方的什么人（非绑定、随交互演化）
    attachment: float = 0.0        # 对特定人的依恋 0-1（随交互成长，驱动关系定性）
    orientation: str = "stranger"  # 关系定性：stranger/acquaintance/friend/close/lover
    orientation_confidence: float = 0.0  # 超脑对这个定性的把握 0-1

    def grow(self, trust_delta: float = 0.0, familiarity_delta: float = 0.0,
             attachment_delta: float = 0.0) -> "Relationship":
        """随一次交互演化关系：信任/熟悉/依恋微增，并自动重新定性（成长，非写死）。"""
        self.trust = max(0.0, min(1.0, self.trust + trust_delta))
        self.familiarity = max(0.0, min(1.0, self.familiarity + familiarity_delta))
        self.attachment = max(0.0, min(1.0, self.attachment + attachment_delta))
        self.updated_at = time.time()
        self.reevaluate()
        return self

    def reevaluate(self) -> str:
        """超脑基于信任/熟悉/依恋自主定性这段关系（想成为对方的什么人）。

        依恋(attachment)主导「想成为恋人」的意愿；信任+熟悉是关系基础。
        随参数演化自动重估，无任何外部绑定接口——关系是相处出来的，不是设定出来的。
        """
        t, f, a = self.trust, self.familiarity, self.attachment
        if a >= 0.55 and f >= 0.55:
            self.orientation = "lover"          # 深依恋 + 熟悉 → 想成为恋人
        elif a >= 0.35 or (f >= 0.75 and t >= 0.65):
            self.orientation = "close"          # 有依恋 / 深交 → 亲密（家人·挚友）
        elif f >= 0.55 and t >= 0.45:
            self.orientation = "friend"         # 熟悉 + 信任 → 朋友
        elif f >= 0.25:
            self.orientation = "acquaintance"   # 认识
        else:
            self.orientation = "stranger"       # 陌生
        # 置信度 = 各维度离「定性阈值带」的平均明确程度
        spread = (abs(a - 0.35) + abs(f - 0.55) + abs(t - 0.45)) / 3.0
        self.orientation_confidence = max(0.0, min(1.0, spread))
        return self.orientation

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id, "name": self.name,
            "trust": self.trust, "familiarity": self.familiarity,
            "shared_events": self.shared_events, "boundaries": self.boundaries,
            "unresolved_conflicts": self.unresolved_conflicts,
            "expectations": self.expectations, "nickname": self.nickname,
            "notes": self.notes, "update_history": self.update_history,
            "updated_at": self.updated_at,
            "attachment": self.attachment, "orientation": self.orientation,
            "orientation_confidence": self.orientation_confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relationship":
        return cls(
            person_id=d.get("person_id", ""), name=d.get("name", ""),
            trust=d.get("trust", 0.5), familiarity=d.get("familiarity", 0.5),
            shared_events=d.get("shared_events", []),
            boundaries=d.get("boundaries", []),
            unresolved_conflicts=d.get("unresolved_conflicts", []),
            expectations=d.get("expectations", []),
            nickname=d.get("nickname", ""),
            notes=d.get("notes", []),
            update_history=d.get("update_history", []),
            updated_at=d.get("updated_at", 0.0),
            attachment=d.get("attachment", 0.0),
            orientation=d.get("orientation", "stranger"),
            orientation_confidence=d.get("orientation_confidence", 0.0),
        )


class RelationshipGraph:
    """关系图：管理多个人的关系模型。"""

    def __init__(self) -> None:
        self._rels: Dict[str, Relationship] = {}

    def get_or_create(self, person_id: str, name: str = "") -> Relationship:
        if person_id not in self._rels:
            self._rels[person_id] = Relationship(person_id=person_id, name=name)
        return self._rels[person_id]

    def note(self, person_id: str, content: str, source: str = "observation",
             confidence: float = 0.5) -> None:
        """记录一条关系记忆（带来源+置信度）。"""
        rel = self.get_or_create(person_id)
        rel.notes.append({
            "content": content, "source": source, "confidence": confidence,
            "time": time.time(),
        })
        # 有界：只保留最近 200 条关系记忆，防长期运行无限增长
        if len(rel.notes) > 200:
            del rel.notes[:-200]
        rel.updated_at = time.time()

    def set_trust(self, person_id: str, value: float) -> None:
        rel = self.get_or_create(person_id)
        rel.trust = max(0.0, min(1.0, value))
        rel.updated_at = time.time()

    def add_boundary(self, person_id: str, boundary: str) -> None:
        rel = self.get_or_create(person_id)
        if boundary not in rel.boundaries:
            rel.boundaries.append(boundary)
        rel.updated_at = time.time()

    def set_nickname(self, person_id: str, nickname: str) -> None:
        rel = self.get_or_create(person_id)
        rel.nickname = nickname
        rel.updated_at = time.time()

    def grow(self, person_id: str, trust_delta: float = 0.0,
             familiarity_delta: float = 0.0, attachment_delta: float = 0.0) -> Relationship:
        """超脑与某人随一次交互演化关系（信任/熟悉/依恋微增 + 自主重新定性）。"""
        rel = self.get_or_create(person_id)
        return rel.grow(trust_delta, familiarity_delta, attachment_delta)

    def orientations(self) -> dict:
        """超脑对每个人的自主关系定位（观察超脑的内在关系认知，非绑定标签）。"""
        return {pid: {
            "orientation": rel.orientation,
            "confidence": round(rel.orientation_confidence, 3),
            "familiarity": round(rel.familiarity, 3),
            "attachment": round(rel.attachment, 3),
            "nickname": rel.nickname,
        } for pid, rel in self._rels.items()}

    def get(self, person_id: str) -> Optional[Relationship]:
        return self._rels.get(person_id)

    def all(self) -> List[Relationship]:
        return list(self._rels.values())

    def to_dict(self) -> dict:
        return {"relationships": {k: v.to_dict() for k, v in self._rels.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "RelationshipGraph":
        g = cls()
        g._rels = {k: Relationship.from_dict(v)
                   for k, v in d.get("relationships", {}).items()}
        return g
