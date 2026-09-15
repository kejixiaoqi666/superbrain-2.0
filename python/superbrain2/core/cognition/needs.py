"""认知层：五大需求驱动。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class NeedType(Enum):
    CERTAINTY = "certainty"    # 确定性：对环境预测能力的需求
    COMPETENCE = "competence"  # 胜任感：能力成长
    AUTONOMY = "autonomy"      # 自主性：自我决定
    RELATEDNESS = "relatedness"  # 归属感：社交连接
    ENERGY = "energy"          # 能量：资源获取


@dataclass
class Need:
    type: NeedType
    current_level: float = 0.5
    target_level: float = 0.8
    decay_rate: float = 0.01
    importance: float = 1.0
    volatility: float = 0.05

    def compute_drive(self) -> float:
        return max(0.0, self.target_level - self.current_level) * self.importance

    def tick(self, dt: float = 1.0) -> float:
        decay = self.decay_rate * dt
        noise = random.gauss(0, self.volatility * dt)
        self.current_level = min(1.0, max(0.0, self.current_level - decay + noise))
        return self.current_level

    def satisfy(self, amount: float) -> None:
        self.current_level = min(1.0, self.current_level + amount)

    @property
    def deficit(self) -> float:
        return max(0.0, self.target_level - self.current_level)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "current": round(self.current_level, 3),
            "target": self.target_level,
            "drive": round(self.compute_drive(), 3),
            "deficit": round(self.deficit, 3),
        }


class NeedDriveSystem:
    """需求驱动系统：五大需求随时间衰减，被满足则回升。"""

    def __init__(self) -> None:
        self.needs: Dict[NeedType, Need] = {}
        self.tick_count = 0
        defaults = {
            NeedType.CERTAINTY: (0.6, 0.8, 0.008, 1.2, 0.04),
            NeedType.COMPETENCE: (0.4, 0.9, 0.012, 1.5, 0.06),
            NeedType.AUTONOMY: (0.5, 0.7, 0.005, 1.0, 0.03),
            NeedType.RELATEDNESS: (0.5, 0.7, 0.010, 0.8, 0.05),
            NeedType.ENERGY: (0.7, 0.8, 0.015, 1.3, 0.04),
        }
        for nt, (cur, tgt, dec, imp, vol) in defaults.items():
            self.needs[nt] = Need(nt, cur, tgt, dec, imp, vol)

    def tick(self, dt: float = 1.0) -> Dict[NeedType, float]:
        self.tick_count += 1
        result = {nt: self.needs[nt].tick(dt) for nt in self.needs}
        self._learn_importance()
        return result

    def _learn_importance(self) -> None:
        """需求重要性习得：相对持续高缺口的需要重要性上升(形成个性偏好)，
        长期不缺的回落。需求的优先级是"长出来的"，而非固定设定。
        """
        deficits = [n.deficit for n in self.needs.values()]
        avg = sum(deficits) / len(deficits)
        for need in self.needs.values():
            if need.deficit > avg + 0.05:
                need.importance = min(2.0, need.importance * (1.0 + 0.002))
            elif need.deficit < avg - 0.05:
                need.importance = max(0.5, need.importance * (1.0 - 0.001))

    def satisfy(self, need_type: NeedType, amount: float) -> None:
        if need_type in self.needs:
            self.needs[need_type].satisfy(amount)

    def get_dominant_need(self) -> Tuple[Optional[NeedType], float]:
        best_type, best_drive = None, -1.0
        for need in self.needs.values():
            d = need.compute_drive()
            if d > best_drive:
                best_drive, best_type = d, need.type
        return best_type, best_drive

    @property
    def mean_satisfaction(self) -> float:
        vals = [self.needs[nt].current_level for nt in self.needs]
        return sum(vals) / len(vals)

    def get_profile(self) -> dict:
        return {nt.value: self.needs[nt].to_dict() for nt in self.needs}
