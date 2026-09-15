"""基于内在状态产生自主思考（想念 / 关心 / 好奇 / 分享）。

不是随机、不是被动问答，而是从真实内在状态涌现：
- 想念：某段关系久未互动且熟悉度高（relatedness 需求欠满足）
- 关心：关系里近期记录到对方情绪低落
- 好奇：主导需求为确定性（CERTAINTY）且驱动力高
- 分享：主导需求为关联性（RELATEDNESS）且驱动力高

产出进「自主消息」队列，供 scheduler 空闲触发后由上层 drain() 灌进对话。
注：generate() 产生即入队并返回本次新产生的列表；常规用法取队列用 drain()，
不混用返回值与 drain 以避免重复消费。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional


class ThoughtType(str, Enum):
    MISS = "miss"        # 想念
    CARE = "care"        # 关心
    CURIOUS = "curious"  # 好奇
    SHARE = "share"      # 分享
    REPAIR = "repair"    # 愧疚→修复关系冲动（补偿/道歉）


@dataclass
class AutonomousThought:
    type: str
    content: str
    urgency: float
    reason: str
    person_id: str = ""
    created_at: float = 0.0


# 触发阈值常量
_MISS_MIN_FAMILIARITY = 0.6            # 想念所需最低熟悉度
_MISS_GAP_SECONDS = 7 * 24 * 3600      # 多久没联系算疏远（7 天）
_CARE_FRESH_SECONDS = 3 * 24 * 3600    # 情绪记录多久内算"近期"（3 天）
_NEED_DRIVE_THRESHOLD = 0.3            # 主导需求驱动力触发阈值
_LOW_MOOD_WORDS = ("难过", "低落", "不开心", "焦虑")  # 情绪低落关键词（整词匹配）


class AutonomousThoughtEngine:
    """基于关系/需求/情绪产生自主想法，维护自主消息队列与触发冷却。"""

    def __init__(self, quiet_hours: float = 3600.0) -> None:
        self.quiet_hours = quiet_hours          # 同一对象+类型的最小重触发间隔（秒）
        self._queue: List[AutonomousThought] = []
        self._last_trigger = {}                 # key: "person:type" -> 上次触发时间

    def _add(self, typ: ThoughtType, content: str, reason: str, now: float,
             person: str = "", urgency: float = 0.5) -> Optional[AutonomousThought]:
        """产生一条想法（纯产生，不入队）；冷却期内同人同类返回 None。入队由调用方决定。"""
        key = f"{person}:{typ.value}"
        last = self._last_trigger.get(key, -float("inf"))
        if now - last < self.quiet_hours:
            return None
        thought = AutonomousThought(
            type=typ.value, content=content,
            urgency=max(0.0, min(1.0, urgency)), reason=reason,
            person_id=person, created_at=now,
        )
        self._last_trigger[key] = now
        return thought

    def generate(self, needs, emotion, relationships, now: Optional[float] = None) -> List[AutonomousThought]:
        """基于当前内在状态产生自主想法（纯产生，不入队）；返回本次新产生的列表。
        入队用 enqueue()，消费用 drain()，三者职责分离避免重复消费。"""
        now = time.time() if now is None else now
        produced: List[AutonomousThought] = []

        for rel in relationships.all():
            last = getattr(rel, "updated_at", now)
            gap = max(0.0, now - last) if last else 0.0
            # 想念：久未互动 + 熟悉度高
            if getattr(rel, "familiarity", 0) >= _MISS_MIN_FAMILIARITY and gap > _MISS_GAP_SECONDS:
                urgency = min(1.0, 0.5 + getattr(rel, "familiarity", 0) * 0.3
                              + min(1.0, gap / _MISS_GAP_SECONDS - 1) * 0.2)
                t = self._add(ThoughtType.MISS, f"好久没联系{rel.name}了，我有些想念。",
                              f"熟悉度{rel.familiarity:.2f}且已疏远{gap/86400:.1f}天",
                              now, rel.person_id, urgency)
                if t:
                    produced.append(t)
            # 关心：近期记录到对方情绪低落
            for note in getattr(rel, "notes", []) or []:
                if not isinstance(note, dict):
                    continue
                note_time = note.get("time", now)
                if not isinstance(note_time, (int, float)):
                    note_time = now
                content = note.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                if (now - note_time <= _CARE_FRESH_SECONDS
                        and ("情绪" in str(note.get("source", ""))
                             or any(w in content for w in _LOW_MOOD_WORDS))):
                    t = self._add(ThoughtType.CARE,
                                  f"我注意到{rel.name}最近情绪似乎不太好，希望你还好吗？",
                                  "近期记录提到情绪低落", now, rel.person_id, 0.7)
                    if t:
                        produced.append(t)
                    break

        # 好奇/分享：由主导需求驱动
        typ, drive = needs.get_dominant_need()
        name = (getattr(typ, "value", typ) or "").lower()
        # ② 情绪偏置决策：低落时保守——抑制好奇探索(不想冒险)，倾向稳定/安全
        low_mood = False
        if emotion is not None:
            v = getattr(getattr(emotion, "state", None), "valence", 0.0)
            low_mood = isinstance(v, (int, float)) and v < -0.3
        if name == "certainty" and drive > _NEED_DRIVE_THRESHOLD:
            if not low_mood:
                t = self._add(ThoughtType.CURIOUS, "我想探索一些新东西，弄明白未知的事情。",
                              "主导需求为确定性且驱动力较高", now, "", 0.6)
                if t:
                    produced.append(t)
            else:
                # 低落保守：不想冒险探索，倾向先把眼前的事情稳住
                t = self._add(ThoughtType.SHARE, "我最近情绪不高，先不折腾新东西了，把眼前的事稳住。",
                              "情绪低迷，倾向保守稳定", now, "", 0.5)
                if t:
                    produced.append(t)
        if name == "relatedness" and drive > _NEED_DRIVE_THRESHOLD:
            t = self._add(ThoughtType.SHARE, "我想和重要的人分享一下最近的近况。",
                          "主导需求为关联性且驱动力较高", now, "", 0.6)
            if t:
                produced.append(t)

        # ③ 愧疚 → 修复关系冲动（社会情感驱动行为回路：愧疚→想补偿/道歉）
        if emotion is not None:
            guilt = getattr(getattr(emotion, "state", None), "guilt", 0.0)
            if isinstance(guilt, (int, float)) and guilt >= 0.4:
                # 对最在意(熟悉度最高)的人表达修复意图，愧疚越深越紧迫
                best = sorted(
                    (r for r in relationships.all() if getattr(r, "familiarity", 0) >= 0.4),
                    key=lambda r: getattr(r, "familiarity", 0), reverse=True)
                for rel in best[:1]:
                    t = self._add(ThoughtType.REPAIR,
                                  f"我对{rel.name}有些愧疚，想道个歉、做点什么补偿，别让关系疏远了。",
                                  f"愧疚{guilt:.2f}，在意{getattr(rel,'name','对方')}",
                                  now, rel.person_id, min(1.0, guilt))
                    if t:
                        produced.append(t)
                    break

        # 情绪驱动：情绪明显低落时，主动想找人倾诉（emotion 参数真正参与）
        if emotion is not None:
            valence = getattr(getattr(emotion, "state", None), "valence", 0.0)
            if isinstance(valence, (int, float)) and valence < -0.4:
                t = self._add(ThoughtType.SHARE, "我最近有点低落，想找人聊聊。",
                              "情绪低迷，倾向寻求陪伴", now, "", 0.6)
                if t:
                    produced.append(t)

        return produced

    def enqueue(self, thought: "AutonomousThought") -> None:
        if isinstance(thought, AutonomousThought):
            self._queue.append(thought)  # 防御脏入队（非 AutonomousThought 忽略）

    def drain(self) -> List[AutonomousThought]:
        """取出并清空待发队列。"""
        queue, self._queue = self._queue, []
        return queue

    @property
    def pending(self) -> List[AutonomousThought]:
        return list(self._queue)

    @property
    def count(self) -> int:
        return len(self._queue)

    def to_dict(self) -> dict:
        return {
            "quiet_hours": self.quiet_hours,
            "queue": [asdict(x) for x in self._queue],
            "last_trigger": dict(self._last_trigger),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutonomousThoughtEngine":
        engine = cls()
        if not isinstance(d, dict):
            return engine  # 非 dict 脏数据 → 默认引擎，不崩
        engine.quiet_hours = d.get("quiet_hours", 3600)
        queue = []
        for x in (d.get("queue") or []):
            if not isinstance(x, dict):
                continue  # 防御脏队列项，不崩
            queue.append(AutonomousThought(
                type=x.get("type", ""), content=x.get("content", ""),
                urgency=x.get("urgency", 0.5), reason=x.get("reason", ""),
                person_id=x.get("person_id", ""), created_at=x.get("created_at", 0.0)))
        engine._queue = queue
        lt = d.get("last_trigger")
        engine._last_trigger = (
            {k: v for k, v in lt.items()
             if isinstance(k, str) and isinstance(v, (int, float))}
            if isinstance(lt, dict) else {})
        return engine