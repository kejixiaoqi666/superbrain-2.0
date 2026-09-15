"""工作记忆：当前任务的短期记忆（显式建模，容量受限）。

模拟人脑工作记忆（约 7±2 项）：存当前任务、最近事件、未完成事项、当前情绪。
与长期记忆的区别：工作记忆是"当下在脑子里"的，容量小、易被覆盖，不持久化。

文档四类记忆中的「工作记忆」：
- 当前正在聊什么
- 最近几轮发生了什么
- 当前任务、情绪和未完成事项
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class WorkingMemoryItem:
    content: str
    kind: str = "event"  # event | task | pending | emotion


class WorkingMemory:
    """容量受限的工作记忆（默认 9 项，超出挤掉最旧的）。"""

    def __init__(self, capacity: int = 9) -> None:
        self.capacity = capacity
        self.items: Deque[WorkingMemoryItem] = deque(maxlen=capacity)
        self.current_task: str = ""          # 当前任务
        self.pending: List[str] = []         # 未完成事项
        self.current_emotion: str = ""       # 当前情绪简述

    def add(self, content: str, kind: str = "event") -> None:
        self.items.append(WorkingMemoryItem(content=content, kind=kind))

    def set_task(self, task: str) -> None:
        self.current_task = task

    def add_pending(self, item: str) -> None:
        if item not in self.pending:
            self.pending.append(item)

    def done_pending(self, item: str) -> None:
        if item in self.pending:
            self.pending.remove(item)

    def recent(self, k: int = 8) -> List[str]:
        """最近 k 条（用于注入上下文）。"""
        return [i.content for i in list(self.items)[-k:]]

    def summary(self) -> str:
        """工作记忆摘要（注入 prompt）。"""
        parts = []
        if self.current_task:
            parts.append(f"当前任务：{self.current_task}")
        if self.pending:
            parts.append("未完成：" + "、".join(self.pending[:3]))
        if self.items:
            parts.append("最近：" + "；".join(self.recent(3)))
        return "。".join(parts) if parts else ""

    def clear(self) -> None:
        self.items.clear()
        self.current_task = ""
        self.pending.clear()
        self.current_emotion = ""

    def to_dict(self) -> dict:
        return {
            "items": [{"content": i.content, "kind": i.kind} for i in self.items],
            "current_task": self.current_task,
            "pending": self.pending,
            "current_emotion": self.current_emotion,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingMemory":
        wm = cls()
        wm.items = deque([WorkingMemoryItem(i["content"], i.get("kind", "event"))
                          for i in d.get("items", [])], maxlen=wm.capacity)
        wm.current_task = d.get("current_task", "")
        wm.pending = d.get("pending", [])
        wm.current_emotion = d.get("current_emotion", "")
        return wm
