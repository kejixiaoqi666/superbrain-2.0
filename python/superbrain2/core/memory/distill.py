"""经验/技能蒸馏 + Wiki 持久经验层 + 回滚（WikiSkill + MemSkill 精髓）。

三层结构（WikiSkill）：
- Raw 轨迹：原始执行日志（event_log 已存）
- Wiki 经验：结构化、可持续积累的失败/成功模式（永不随 Skill 回滚丢失）
- Skill 规则：可执行、可回滚的技能

关键：Skill 更新带版本，验证变好才合并，性能退化回滚到上一版；
但 Wiki 经验持续保留（失败模式和成功策略不随会话结束丢失）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Experience:
    """一条经验：具体教训。"""
    id: str = ""
    task: str = ""          # 任务描述
    context: str = ""       # 情境
    outcome: str = ""       # 结果（成功/失败）
    lesson: str = ""        # 提炼的教训
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"id": self.id, "task": self.task, "context": self.context,
                "outcome": self.outcome, "lesson": self.lesson,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        return cls(id=d.get("id", ""), task=d.get("task", ""),
                   context=d.get("context", ""), outcome=d.get("outcome", ""),
                   lesson=d.get("lesson", ""), created_at=d.get("created_at", 0.0))


@dataclass
class Skill:
    """一个可复用技能：从多次经验沉淀，带版本可回滚。"""
    name: str = ""
    procedure: str = ""     # 步骤（怎么做）
    proficiency: float = 0.0  # 熟练度 0-1
    success_count: int = 0
    fail_count: int = 0
    related_experiences: List[str] = field(default_factory=list)
    version: int = 1        # 版本（回滚用）
    history: List[str] = field(default_factory=list)  # 历史 procedure 快照（回滚栈）
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"name": self.name, "procedure": self.procedure,
                "proficiency": self.proficiency, "success_count": self.success_count,
                "fail_count": self.fail_count,
                "related_experiences": self.related_experiences,
                "version": self.version, "history": self.history,
                "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(name=d.get("name", ""), procedure=d.get("procedure", ""),
                   proficiency=d.get("proficiency", 0.0),
                   success_count=d.get("success_count", 0),
                   fail_count=d.get("fail_count", 0),
                   related_experiences=d.get("related_experiences", []),
                   version=d.get("version", 1), history=d.get("history", []),
                   updated_at=d.get("updated_at", 0.0))


@dataclass
class WikiEntry:
    """一条 Wiki 持久经验（结构化失败/成功模式，永不随 Skill 回滚丢失）。"""
    id: str = ""
    pattern: str = ""       # 模式描述（如"用户说'稍等'时不应立即追问"）
    kind: str = "lesson"    # lesson | failure | success | heuristic
    evidence: str = ""      # 证据（来自哪个 raw 轨迹）
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    reinforced: int = 0     # 被后续经验强化的次数

    def to_dict(self) -> dict:
        return {"id": self.id, "pattern": self.pattern, "kind": self.kind,
                "evidence": self.evidence, "confidence": self.confidence,
                "created_at": self.created_at, "reinforced": self.reinforced}

    @classmethod
    def from_dict(cls, d: dict) -> "WikiEntry":
        return cls(id=d.get("id", ""), pattern=d.get("pattern", ""),
                   kind=d.get("kind", "lesson"), evidence=d.get("evidence", ""),
                   confidence=d.get("confidence", 0.5),
                   created_at=d.get("created_at", 0.0),
                   reinforced=d.get("reinforced", 0))


class Distiller:
    """经验/技能双层蒸馏引擎。"""

    def __init__(self) -> None:
        self._experiences: Dict[str, Experience] = {}
        self._skills: Dict[str, Skill] = {}
        self._wiki: Dict[str, WikiEntry] = {}  # Wiki 持久经验层
        self._hard_cases: List[dict] = []      # Designer 演化用 hard cases

    # ---------- 记录经验 ----------

    def record_experience(self, task: str, context: str, outcome: str,
                          lesson: str) -> Experience:
        eid = f"exp_{int(time.time()*1000)}"
        exp = Experience(id=eid, task=task, context=context,
                         outcome=outcome, lesson=lesson)
        self._experiences[eid] = exp
        return exp

    # ---------- 蒸馏技能 ----------

    def distill_skill(self, name: str, procedure: str,
                      success: bool = True) -> Skill:
        """沉淀/更新一个技能：成功则熟练度上升，失败则下降。"""
        sk = self._skills.get(name) or Skill(name=name, procedure=procedure)
        sk.procedure = procedure or sk.procedure
        if success:
            sk.success_count += 1
        else:
            sk.fail_count += 1
        total = sk.success_count + sk.fail_count
        sk.proficiency = sk.success_count / total if total else 0.0
        sk.updated_at = time.time()
        self._skills[name] = sk
        return sk

    def best_skills(self, k: int = 5) -> List[Skill]:
        """熟练度最高的技能（可复用工作流）。"""
        return sorted(self._skills.values(), key=lambda s: s.proficiency, reverse=True)[:k]

    # ---------- Wiki 持久经验层（WikiSkill） ----------

    def record_wiki(self, pattern: str, kind: str = "lesson",
                    evidence: str = "", confidence: float = 0.5) -> WikiEntry:
        """记录/强化一条 Wiki 经验（持久，不随 Skill 回滚丢失）。"""
        import hashlib
        key = hashlib.blake2b(pattern.encode("utf-8"), digest_size=8).hexdigest()
        if key in self._wiki:
            # 强化已有模式
            w = self._wiki[key]
            w.reinforced += 1
            w.confidence = min(1.0, w.confidence + 0.05)
            return w
        w = WikiEntry(id=key, pattern=pattern, kind=kind,
                      evidence=evidence, confidence=confidence)
        self._wiki[key] = w
        return w

    def wiki_patterns(self, kind: Optional[str] = None) -> List[WikiEntry]:
        """取 Wiki 经验（可按 kind 过滤）。"""
        items = list(self._wiki.values())
        if kind:
            items = [w for w in items if w.kind == kind]
        return sorted(items, key=lambda w: w.confidence, reverse=True)

    # ---------- Skill 回滚（MemSkill） ----------

    def update_skill(self, name: str, new_procedure: str) -> Skill:
        """带版本更新技能：把旧 procedure 压入回滚栈。"""
        sk = self._skills.get(name) or Skill(name=name, procedure="")
        if sk.procedure and sk.procedure != new_procedure:
            sk.history.append(sk.procedure)  # 保存旧版到回滚栈
            sk.version += 1
        sk.procedure = new_procedure
        sk.updated_at = time.time()
        self._skills[name] = sk
        return sk

    def rollback_skill(self, name: str) -> Optional[Skill]:
        """回滚技能到上一版（性能退化时）。Wiki 经验不受影响。"""
        sk = self._skills.get(name)
        if sk is None or not sk.history:
            return None
        sk.procedure = sk.history.pop()  # 弹出上一版
        sk.version += 1
        sk.proficiency = max(0.0, sk.proficiency * 0.8)  # 回滚降熟练度
        sk.updated_at = time.time()
        return sk

    # ---------- Designer：从 hard case 演化 skill 集（MemSkill） ----------

    def record_hard_case(self, skill_name: str, task: str, problem: str) -> None:
        """记录 hard case（选错 skill 导致记忆错误/不全的样本）。"""
        self._hard_cases.append({
            "skill": skill_name, "task": task, "problem": problem,
            "at": time.time(),
        })

    def design_skills(self, refine_fn) -> List[str]:
        """Designer：审 hard cases，提议改进/新 skill。

        refine_fn(hard_case) -> 返回 (skill_name, new_procedure) 或 None。
        返回本次演化涉及的操作描述列表。
        """
        results = []
        for case in list(self._hard_cases):
            out = refine_fn(case)
            if out:
                name, new_proc = out
                self.update_skill(name, new_proc)  # 带版本，可回滚
                results.append(f"演化 skill '{name}'")
        self._hard_cases.clear()
        return results

    # ---------- 序列化 ----------

    def to_dict(self) -> dict:
        return {
            "experiences": {k: v.to_dict() for k, v in self._experiences.items()},
            "skills": {k: v.to_dict() for k, v in self._skills.items()},
            "wiki": {k: v.to_dict() for k, v in self._wiki.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Distiller":
        dist = cls()
        dist._experiences = {k: Experience.from_dict(v) for k, v in d.get("experiences", {}).items()}
        dist._skills = {k: Skill.from_dict(v) for k, v in d.get("skills", {}).items()}
        dist._wiki = {k: WikiEntry.from_dict(v) for k, v in d.get("wiki", {}).items()}
        return dist

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Distiller":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
