"""认知层：情绪梯度系统（内部状态 ≠ 对外表达）。

情绪 = 需求满足率的微分信号，不是标签，是连续梯度场。

关键设计：区分「内部真实状态」和「对外表达」——
同一个"我没事"可能是平静的没事、强撑的没事、或不想让对方担心。
内部估计 ≠ 对外说法，且每个状态带来源、置信度和衰减时间。
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EmotionalState:
    valence: float = 0.0       # 愉悦度 -1..1（瞬时事件驱动）
    arousal: float = 0.5       # 唤醒度 0..1
    dominance: float = 0.5     # 掌控感 0..1
    confidence: float = 0.5    # 自信 0..1
    # 扩展维度
    certainty: float = 0.5     # 确定感 0..1
    safety: float = 0.5        # 安全感 0..1
    fatigue: float = 0.0       # 疲惫度 0..1
    attachment: float = 0.5    # 依恋 0..1
    # 心境：慢变量基线 -1..1（随瞬时情绪长期累积缓慢漂移，瞬时情绪会向它回归）
    mood: float = 0.0
    # 社会情绪：自豪/愧疚（基于行为与自我标准的道德情感）
    pride: float = 0.0      # 自豪度 0..1（做成好事/符合标准/帮助到人）
    guilt: float = 0.0      # 愧疚度 0..1（违背标准/伤害关系/失责）

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "dominance": round(self.dominance, 3),
            "confidence": round(self.confidence, 3),
            "certainty": round(self.certainty, 3),
            "safety": round(self.safety, 3),
            "fatigue": round(self.fatigue, 3),
            "attachment": round(self.attachment, 3),
            "mood": round(self.mood, 3),
            "pride": round(self.pride, 3),
            "guilt": round(self.guilt, 3),
        }


@dataclass
class Expression:
    """对外表达（可能与内部状态不一致）。"""
    text: str = ""             # 对外说法（如"我没事"）
    intensity: float = 0.5     # 实际对外表达强度 0..1
    deviation: float = 0.0     # 与内部状态的偏差（>0 表示压抑/强撑）
    reason: str = ""           # 为何这样表达（如"不想让对方担心"）


class EmotionGradient:
    """情绪从需求满足率的变化率自然涌现，用指数平滑保持连续感。"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.8,
                 smoothing: float = 0.3) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smoothing = smoothing
        self.state = EmotionalState()
        self.expression = Expression()
        self._need_history: deque = deque(maxlen=100)   # 有上限，防无限增长
        self._reward_history: deque = deque(maxlen=200)  # 有上限，防无限增长

    def update(self, satisfactions: Dict[str, float],
               task_success: Optional[float] = None,
               novelty: Optional[float] = None,
               contagion: float = 0.0) -> EmotionalState:
        """需求满足 → 瞬时情绪更新。contagion(他人情绪感染 -1..1) 额外拉低/抬高自身。"""
        self._need_history.append(satisfactions)
        avg = statistics.mean(satisfactions.values())
        v_target = max(-1.0, min(1.0, 2.0 * avg - 1.0))
        if contagion:
            # 情绪感染：自身情绪被对方同化——对方低落(contagion<0)拖低、高涨(>0)提振
            v_target = max(-1.0, min(1.0, v_target + contagion * 0.4))
        self.state.valence = self._smooth(self.state.valence, v_target)
        # 心境(mood)慢变量：随瞬时情绪长期累积缓慢漂移（乐观/悲观基线，很慢）
        self.state.mood = self._approach(self.state.mood, v_target * 0.6, 0.02)

        if len(self._need_history) >= 2:
            prev = self._need_history[-2]
            curr = self._need_history[-1]
            deltas = [curr[k] - prev.get(k, 0.5) for k in curr]
            drive = min(1.0, abs(statistics.mean(deltas)) * 3)
        else:
            drive = 0.3
        self.state.arousal = self._smooth(self.state.arousal, 0.3 + 0.7 * drive)

        if task_success is not None:
            self.state.dominance = self._smooth(self.state.dominance, 0.2 + 0.8 * task_success)
        if novelty is not None:
            self.state.confidence = self._smooth(self.state.confidence, max(0.0, min(1.0, 1.0 - novelty)))
        return self.state

    def elapse(self, dt_hours: float = 1.0, tau_hours: float = 6.0) -> EmotionalState:
        """情绪随时间的动力学：瞬时情绪向心境(mood)基线指数回归（事件后的高涨/低落会回落）。

        tau_hours = 情绪回归到基线的时长常数；dt_hours = 流逝时间。模拟"事件会过去，
        但心境底色还在"——情绪不是停留在事件峰值，而是按衰减曲线回到慢变量基线。
        """
        if not isinstance(dt_hours, (int, float)) or dt_hours <= 0:
            return self.state
        rate = 1.0 - math.exp(-float(dt_hours) / max(tau_hours, 0.5))
        self.state.valence = self._approach(self.state.valence, self.state.mood, rate)
        return self.state

    def social_update(self, pride: float = 0.0, guilt: float = 0.0) -> EmotionalState:
        """社会情绪更新：自豪 / 愧疚（基于行为与自我标准，道德情感）。

        自豪(做成好事/帮到人/符合标准)：valence 上行 + dominance 上行 + pride 累积；
        愧疚(违背标准/伤害关系/失责)：valence 下行 + dominance 下行 + guilt 累积，
        且驱动修复关系(关系层)的内在动机。
        """
        if pride:
            p = max(0.0, min(1.0, pride))
            self.state.pride = max(0.0, min(1.0, self.state.pride + p * 0.3))
            self.state.guilt = max(0.0, self.state.guilt - p * 0.1)  # 自豪减轻愧疚
            self.state.valence = self._approach(self.state.valence, 1.0, p * 0.4)
            self.state.dominance = self._approach(self.state.dominance, 1.0, p * 0.3)
        if guilt:
            g = max(0.0, min(1.0, guilt))
            self.state.guilt = max(0.0, min(1.0, self.state.guilt + g * 0.3))
            self.state.valence = self._approach(self.state.valence, -1.0, g * 0.4)
            self.state.dominance = self._approach(self.state.dominance, 0.0, g * 0.3)
        return self.state

    def set_expression(self, text: str, intensity: Optional[float] = None,
                       reason: str = "") -> None:
        """设置对外表达，自动计算与内部状态的偏差（识别"强撑"）。"""
        self.expression.text = text
        self.expression.intensity = intensity if intensity is not None else \
            (self.state.valence + 1.0) / 2.0
        # 偏差 = 对外表达的积极程度 vs 内部真实 valence
        expressed_positivity = self.expression.intensity
        internal_positivity = (self.state.valence + 1.0) / 2.0
        self.expression.deviation = abs(expressed_positivity - internal_positivity)
        self.expression.reason = reason

    @property
    def masking(self) -> bool:
        """是否在压抑情绪（对外积极但内部消极）。"""
        return (self.expression.deviation > 0.3 and
                self.state.valence < 0 and
                self.expression.intensity > 0.5)

    def compute_intrinsic_reward(self) -> float:
        vd = 0.0
        if len(self._need_history) >= 2:
            vd = statistics.mean(list(self._need_history[-1].values())) \
                 - statistics.mean(list(self._need_history[-2].values()))
        r = max(-1.0, min(1.0, self.alpha * vd + self.beta * (1.0 - self.state.confidence)
                          + self.gamma * self.state.arousal))
        self._reward_history.append(r)
        return r

    @property
    def mean_reward(self, window: int = 20) -> float:
        r = list(self._reward_history)[-window:] if self._reward_history else [0.0]
        return statistics.mean(r)

    def _smooth(self, old: float, new: float) -> float:
        return old * self.smoothing + new * (1.0 - self.smoothing)

    @staticmethod
    def _approach(old: float, target: float, rate: float) -> float:
        """按速率向目标靠拢（rate 0..1，小=慢）。"""
        return old + rate * (target - old)

    def reset(self) -> None:
        self.state = EmotionalState()
        self.expression = Expression()
        self._need_history.clear()
        self._reward_history.clear()
