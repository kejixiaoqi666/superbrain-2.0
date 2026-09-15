"""身份层：稳定身份锚点。

避免每轮对话都像重新加载一个陌生模型。身份锚点跨会话稳定：
- name（名称）
- core_identity（核心身份定位，不是一次性聊天机器人）
- expression_style（表达风格：语气、口头禅、表达习惯）
- boundaries（边界：什么话题/行为不参与）

与 SelfModel 的区别：SelfModel 描述"状态"（过去/现在/将来的我），
Identity 是"我是谁"的稳定锚点（不随对话变化）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Identity:
    name: str = ""
    core_identity: str = ""           # 核心身份定位
    expression_style: str = ""        # 语气/表达习惯
    boundaries: List[str] = field(default_factory=list)  # 不参与的话题/行为
    # 时间身份：能区分"过去的我、现在的我、将来的我"
    past_self: str = ""               # 过去的我
    present_self: str = ""            # 现在的我
    future_self: str = ""             # 将来的我

    def anchor_text(self) -> str:
        """生成稳定身份锚点（注入 prompt 的开头，每轮不变）。"""
        parts = []
        if self.name:
            parts.append(f"你是{self.name}")
        if self.core_identity:
            parts.append(self.core_identity)
        if self.expression_style:
            parts.append(f"表达风格：{self.expression_style}")
        return "。".join(parts) if parts else ""

    def temporal_text(self) -> str:
        """时间身份：过去/现在/将来的我。"""
        parts = []
        if self.past_self:
            parts.append(f"过去的我：{self.past_self}")
        if self.present_self:
            parts.append(f"现在的我：{self.present_self}")
        if self.future_self:
            parts.append(f"将来的我：{self.future_self}")
        return "；".join(parts) if parts else ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "core_identity": self.core_identity,
            "expression_style": self.expression_style, "boundaries": self.boundaries,
            "past_self": self.past_self, "present_self": self.present_self,
            "future_self": self.future_self,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Identity":
        return cls(
            name=d.get("name", ""), core_identity=d.get("core_identity", ""),
            expression_style=d.get("expression_style", ""),
            boundaries=d.get("boundaries", []),
            past_self=d.get("past_self", ""), present_self=d.get("present_self", ""),
            future_self=d.get("future_self", ""),
        )
