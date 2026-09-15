"""自我模型：让 agent 知道自己处于什么状态。

可计算的自我表征，让行为更连贯：
- identity（我是谁）
- current_goal（我现在要完成什么）
- capability（我能做什么）
- limitation（我做不到什么）
- relationship（我与谁是什么关系）
- continuity（哪些过去经历仍影响现在）
- uncertainty（哪些判断不确定）
- next_intention（下一步想做什么）

这是「自我叙事」的工程基础——能描述过去/现在/将来的我。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class SelfModel:
    identity: str = ""           # 我是谁（一句话自我定位）
    current_goal: str = ""       # 当前目标
    capability: List[str] = field(default_factory=list)    # 我能做什么
    limitation: List[str] = field(default_factory=list)    # 我做不到什么
    relationships: List[str] = field(default_factory=list) # 我与谁是什么关系
    continuity: List[str] = field(default_factory=list)    # 仍影响现在的过去经历
    uncertainty: List[str] = field(default_factory=list)   # 不确定的判断
    next_intention: str = ""     # 下一步想做什么
    updated_at: float = field(default_factory=time.time)

    def narrative(self) -> str:
        """生成自我叙事（过去/现在/将来）。"""
        parts = []
        if self.identity:
            parts.append(f"我是{self.identity}")
        if self.continuity:
            parts.append("过去：" + "；".join(self.continuity[:3]))
        if self.current_goal:
            parts.append(f"现在：{self.current_goal}")
        if self.next_intention:
            parts.append(f"将来：{self.next_intention}")
        if self.uncertainty:
            parts.append("不确定：" + "；".join(self.uncertainty[:3]))
        return "。".join(parts) if parts else ""

    def to_dict(self) -> dict:
        return {
            "identity": self.identity, "current_goal": self.current_goal,
            "capability": self.capability, "limitation": self.limitation,
            "relationships": self.relationships, "continuity": self.continuity,
            "uncertainty": self.uncertainty, "next_intention": self.next_intention,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfModel":
        return cls(
            identity=d.get("identity", ""), current_goal=d.get("current_goal", ""),
            capability=d.get("capability", []), limitation=d.get("limitation", []),
            relationships=d.get("relationships", []), continuity=d.get("continuity", []),
            uncertainty=d.get("uncertainty", []), next_intention=d.get("next_intention", ""),
            updated_at=d.get("updated_at", 0.0),
        )
