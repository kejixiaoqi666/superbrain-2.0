"""种子机制：经历 → 行为倾向（人格系统的核心）。

种子 = 一次经历留下的、会持续影响未来行为的"倾向"，不是死记忆。
它带触发条件，会被后续经历强化、修正或随时间衰减。

关键设计（文档强调）：
- 事实、解释、推测分开存（provenance）
- 每条种子带来源 + 置信度
- 可修订、衰减、撤销（避免一次误会永久改变人格）
- 高影响种子需更高置信度
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Seed:
    """一条倾向种子。"""
    id: str = ""
    content: str = ""            # 这个倾向是什么（如"完整表达通常能获得耐心回应"）
    strength: float = 0.5        # 当前影响强度 0-1
    confidence: float = 0.5      # 可信度 0-1
    valence: float = 0.0         # 情绪价 -1..1
    triggers: List[str] = field(default_factory=list)  # 什么情境触发
    behavior_bias: str = ""      # 推向什么行为
    source_event: str = ""       # 来源事件 id
    provenance: str = "observation"  # observation | statement | inference
    decay_rate: float = 0.01     # 衰减率
    reinforcement_count: int = 0
    revision: int = 1
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "content": self.content,
            "strength": self.strength, "confidence": self.confidence,
            "valence": self.valence, "triggers": self.triggers,
            "behavior_bias": self.behavior_bias, "source_event": self.source_event,
            "provenance": self.provenance, "decay_rate": self.decay_rate,
            "reinforcement_count": self.reinforcement_count,
            "revision": self.revision, "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Seed":
        return cls(
            id=d.get("id", ""), content=d.get("content", ""),
            strength=d.get("strength", 0.5), confidence=d.get("confidence", 0.5),
            valence=d.get("valence", 0.0), triggers=d.get("triggers", []),
            behavior_bias=d.get("behavior_bias", ""),
            source_event=d.get("source_event", ""),
            provenance=d.get("provenance", "observation"),
            decay_rate=d.get("decay_rate", 0.01),
            reinforcement_count=d.get("reinforcement_count", 0),
            revision=d.get("revision", 1),
            created_at=d.get("created_at", 0.0),
            last_updated=d.get("last_updated", 0.0),
        )

    def effective_strength(self, now: float | None = None) -> float:
        """当前有效强度 = strength × 置信度 × 时间衰减。"""
        t = now or time.time()
        elapsed_hours = (t - self.last_updated) / 3600.0
        decay = max(0.0, 1.0 - self.decay_rate * elapsed_hours)
        return self.strength * self.confidence * decay


class SeedBank:
    """种子库：管理倾向种子的增删改查、强化、衰减、修正。"""

    def __init__(self) -> None:
        self._seeds: Dict[str, Seed] = {}
        self._by_trigger: Dict[str, List[str]] = {}

    def add(self, content: str, triggers: List[str] = None,
            behavior_bias: str = "", valence: float = 0.0,
            confidence: float = 0.5, provenance: str = "observation",
            source_event: str = "") -> Seed:
        sid = f"seed_{int(time.time()*1000)}"
        seed = Seed(id=sid, content=content, triggers=triggers or [],
                    behavior_bias=behavior_bias, valence=valence,
                    confidence=confidence, provenance=provenance,
                    source_event=source_event)
        self._seeds[sid] = seed
        for t in seed.triggers:
            self._by_trigger.setdefault(t, []).append(sid)
        return seed

    def find_similar(self, content: str, threshold: float = 0.6) -> Optional[Seed]:
        """找内容相似（字符 n-gram 重叠）的已有种子，用于去重强化。"""
        def grams(s: str, n: int = 2):
            s = "".join(s.lower().split())
            return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else {s}
        g1 = grams(content)
        if not g1:
            return None
        best, best_score = None, 0.0
        for s in self._seeds.values():
            g2 = grams(s.content)
            if not g2:
                continue
            overlap = len(g1 & g2) / max(1, len(g1 | g2))
            if overlap > best_score:
                best, best_score = s, overlap
        return best if best_score >= threshold else None

    def add_or_reinforce(self, content: str, triggers: List[str] = None,
                         behavior_bias: str = "", valence: float = 0.0,
                         confidence: float = 0.5, provenance: str = "observation",
                         source_event: str = "",
                         min_confidence: float = 0.3,
                         similarity_threshold: float = 0.6) -> Optional[Seed]:
        """智能沉淀：低置信度不入库；相似种子强化而非新增。

        - confidence < min_confidence：不写（避免低质量内容污染人格）
        - 有相似种子：强化它（新证据巩固旧判断）
        - 否则：新建
        """
        if confidence < min_confidence:
            return None
        similar = self.find_similar(content, similarity_threshold)
        if similar is not None:
            self.reinforce(similar.id, amount=0.1)
            # 合并触发词
            for t in (triggers or []):
                if t not in similar.triggers:
                    similar.triggers.append(t)
            return similar
        return self.add(content, triggers=triggers, behavior_bias=behavior_bias,
                        valence=valence, confidence=confidence,
                        provenance=provenance, source_event=source_event)

    def reinforce(self, seed_id: str, amount: float = 0.1) -> None:
        """强化种子（被验证的经历）。"""
        s = self._seeds.get(seed_id)
        if s is None:
            return
        s.strength = min(1.0, s.strength + amount)
        s.confidence = min(1.0, s.confidence + amount * 0.5)
        s.reinforcement_count += 1
        s.last_updated = time.time()

    def correct(self, seed_id: str, new_content: str, reason: str = "") -> None:
        """修正种子（新证据推翻旧判断），保留修订版本。"""
        s = self._seeds.get(seed_id)
        if s is None:
            return
        s.content = new_content
        s.revision += 1
        s.strength = max(0.1, s.strength * 0.7)  # 修正后降强度，需重新验证
        s.confidence = max(0.1, s.confidence * 0.7)
        s.last_updated = time.time()

    def revoke(self, seed_id: str) -> None:
        """撤销种子（证明错误）。"""
        self._seeds.pop(seed_id, None)
        for t, ids in self._by_trigger.items():
            if seed_id in ids:
                ids.remove(seed_id)

    def decay_all(self, now: float | None = None) -> None:
        """衰减：低于阈值的种子自动移除（避免负面记忆无限强化）。"""
        t = now or time.time()
        to_remove = []
        for sid, s in self._seeds.items():
            if s.effective_strength(t) < 0.05:
                to_remove.append(sid)
        for sid in to_remove:
            self._seeds.pop(sid, None)

    def match(self, context: str, k: int = 5) -> List[Seed]:
        """按触发条件/内容匹配种子，返回最相关的（影响当前行为）。"""
        scored = []
        cl = context.lower()
        for s in self._seeds.values():
            score = 0.0
            # 触发词匹配
            for t in s.triggers:
                if t.lower() in cl:
                    score += 1.0
            # 内容关键词匹配
            for w in s.content:
                if w in cl:
                    score += 0.1
            if score > 0:
                scored.append((s, score * s.effective_strength()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:k]]

    def dominant(self, context: str) -> Optional[Seed]:
        """当前情境下最强的主导种子。"""
        matched = self.match(context, k=1)
        return matched[0] if matched else None

    def all(self) -> List[Seed]:
        return list(self._seeds.values())

    def to_dict(self) -> dict:
        return {"seeds": {k: v.to_dict() for k, v in self._seeds.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "SeedBank":
        bank = cls()
        bank._seeds = {k: Seed.from_dict(v) for k, v in d.get("seeds", {}).items()}
        bank._reindex()
        return bank

    def _reindex(self) -> None:
        self._by_trigger = {}
        for sid, s in self._seeds.items():
            for t in s.triggers:
                self._by_trigger.setdefault(t, []).append(sid)
