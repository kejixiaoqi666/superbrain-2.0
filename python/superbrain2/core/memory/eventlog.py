"""事件日志：原始对话/事件的追加写日志（溯源基础）。

文档第12节「存储建议」：原始对话应追加写日志。
这是 provenance（来源）的基础——每条种子/记忆/关系都能追溯到原始事件，
避免"把推测写成事实"。

事件日志是只追加的（append-only），永不改写，是"事实的原始证据"。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Event:
    id: str
    time: float
    kind: str                  # conversation | action | learning | system
    content: str
    participants: List[str] = field(default_factory=list)
    emotion: Dict[str, float] = field(default_factory=dict)
    provenance: str = "direct_observation"  # direct_observation | statement | inference
    revision: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id, "time": self.time, "kind": self.kind,
            "content": self.content, "participants": self.participants,
            "emotion": self.emotion, "provenance": self.provenance,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            id=d.get("id", ""), time=d.get("time", 0.0), kind=d.get("kind", ""),
            content=d.get("content", ""), participants=d.get("participants", []),
            emotion=d.get("emotion", {}), provenance=d.get("provenance", "direct_observation"),
            revision=d.get("revision", 1),
        )


class EventLog:
    """追加写的事件日志。"""

    def __init__(self, max_events: int = 10000) -> None:
        self.max_events = max_events
        self._events: List[Event] = []

    def append(self, kind: str, content: str, participants: List[str] = None,
               emotion: Dict[str, float] = None,
               provenance: str = "direct_observation") -> Event:
        e = Event(
            id=f"evt_{uuid.uuid4().hex[:12]}",
            time=time.time(), kind=kind, content=content,
            participants=participants or [], emotion=emotion or {},
            provenance=provenance,
        )
        self._events.append(e)
        # 容量控制：超出裁剪最旧的
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]
        return e

    def recent(self, k: int = 20) -> List[Event]:
        return self._events[-k:]

    def since(self, ts: float) -> List[Event]:
        return [e for e in self._events if e.time >= ts]

    def to_dict(self) -> dict:
        return {"events": [e.to_dict() for e in self._events]}

    @classmethod
    def from_dict(cls, d: dict) -> "EventLog":
        log = cls()
        log._events = [Event.from_dict(e) for e in d.get("events", [])]
        return log

    def __len__(self) -> int:
        return len(self._events)
