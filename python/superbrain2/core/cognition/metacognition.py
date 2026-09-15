"""元认知：思考自己的思考（决策质量反思闭环）。

与 self_model 的区别：
- SelfModel 描述"我是谁"（静态自我表征）
- 元认知是"我的思考质量如何、要不要换策略"（动态反思）

闭环：
行动/决策 → 评估结果 → 反思"我当时为什么这样想？质量如何？"
  → 更新内部策略（是高估了？太草率？）→ 影响下次决策
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Reflection:
    """一次反思结果。"""
    decision: str = ""            # 反思的决策
    reasoning_quality: float = 0.5  # 当时推理质量 0-1
    confidence: float = 0.5       # 当时多自信
    outcome: Optional[float] = None  # 结果反馈 (-1坏..1好)
    insight: str = ""             # 得到的洞见
    timestamp: float = field(default_factory=time.time)


class Metacognition:
    """元认知引擎：反思决策质量，调整内部策略。"""

    def __init__(self) -> None:
        self._reflections: deque = deque(maxlen=100)  # 有上限，防无限增长
        # 内部策略（会被反思调整）
        self.cautiousness: float = 0.5      # 谨慎度
        self.strategy_bias: Dict[str, float] = {}  # 各策略的表现得分

    def reflect(self, decision: str, reasoning_quality: float,
                confidence: float, outcome: Optional[float] = None,
                insight: str = "") -> Reflection:
        """反思一次决策，更新内部策略。返回本次反思。"""
        r = Reflection(decision=decision, reasoning_quality=reasoning_quality,
                       confidence=confidence, outcome=outcome, insight=insight)
        self._reflections.append(r)
        # 依据结果调整策略
        if outcome is not None:
            delta = outcome * 0.1
            # 高自信但结果差 → 提高谨慎度（下次别太莽）
            if confidence > 0.7 and outcome < 0:
                self.cautiousness = min(1.0, self.cautiousness + 0.1)
                r.insight = (r.insight + " 高估了自己，" if not r.insight
                             else r.insight + "；高估了自己")
            rst = self.strategy_bias.get(decision, 0.5)
            self.strategy_bias[decision] = max(0.0, min(1.0, rst + delta))
        return r

    def reasoning_trend(self, window: int = 5) -> float:
        """近期推理质量趋势（反思→改进）。"""
        recent = list(self._reflections)[-window:]
        if not recent:
            return 0.5
        return sum(r.reasoning_quality for r in recent) / len(recent)

    def assess_pending(self, confidence: float) -> dict:
        """对即将做的事做元认知评估（实时）。"""
        return {
            "should_pause": confidence < 0.3,            # 太不确定该暂停
            "should_be_cautious": self.cautiousness > 0.7,  # 要谨慎
            "past_performance": self.reasoning_trend(),
            "suggestions": self._suggest(),
        }

    def _suggest(self) -> List[str]:
        out = []
        if self.cautiousness > 0.7:
            out.append("近期多次高估，建议更保守")
        if self.reasoning_trend() < 0.4:
            out.append("推理质量偏低，建议换思路")
        return out

    def to_dict(self) -> dict:
        return {
            "cautiousness": round(self.cautiousness, 3),
            "strategy_bias": {k: round(v, 3) for k, v in self.strategy_bias.items()},
            "reflections": [
                {"decision": r.decision, "quality": round(r.reasoning_quality, 2),
                 "outcome": r.outcome, "insight": r.insight}
                for r in list(self._reflections)[-10:]
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Metacognition":
        m = cls()
        m.cautiousness = d.get("cautiousness", 0.5)
        m.strategy_bias = {k: v for k, v in d.get("strategy_bias", {}).items()}
        for rd in d.get("reflections", []):
            m._reflections.append(Reflection(
                decision=rd.get("decision", ""),
                reasoning_quality=rd.get("quality", 0.5),
                outcome=rd.get("outcome"),
                insight=rd.get("insight", ""),
            ))
        return m