"""睡眠计算 dream()（Letta 精髓）：空闲时回顾近期对话，提炼教训写回记忆。

不改变用户可见行为，只在后台把碎片交互沉淀为经验和技能，
让 agent 越用越强。反思闭环的落地。
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from .store import MemoryStore
from .node import MemoryNode
from .distill import Distiller


class DreamEngine:
    """睡眠计算引擎：回顾近期记忆 → 提炼 → 写回。"""

    def __init__(self, store: MemoryStore, distiller: Distiller) -> None:
        self.store = store
        self.distiller = distiller
        self._last_dream_at = 0.0

    def dream(self, extract: Optional[Callable[[str], str]] = None,
              window_hours: float = 24.0) -> List[str]:
        """回顾近 window_hours 的记忆，提炼教训。

        extract: 可选的外部提炼函数（如 LLM），输入一条记忆内容，输出一条教训。
        缺省用规则提炼（关键词启发）。
        返回本次写回的经验 id 列表。
        """
        now = time.time()
        recent = [n for n in self.store.all_nodes()
                  if (now - n.last_access) / 3600.0 <= window_hours]
        lessons: List[str] = []
        for n in recent:
            lesson = extract(n.content) if extract else self._rule_extract(n)
            if lesson:
                eid = self._remember_lesson(n, lesson)
                lessons.append(eid)
        self._last_dream_at = now
        return lessons

    def _remember_lesson(self, node: MemoryNode, lesson: str) -> str:
        exp = self.distiller.record_experience(
            task="回顾", context=node.content[:60],
            outcome="沉淀", lesson=lesson,
        )
        return exp.id

    @staticmethod
    def _rule_extract(node: MemoryNode) -> str:
        """规则提炼（无 LLM 时）：把带结论性关键词的记忆转成教训。"""
        c = node.content
        for kw in ("教训", "应该", "以后", "记住", "注意", "避免", "结论", "要"):
            if kw in c:
                return c[:120]
        return ""

    @property
    def last_dream_at(self) -> float:
        return self._last_dream_at
