"""认知层：EG-MRSI 完整版（元认知自改进）。

基于 EG-MRSI 框架（元认知 · 情绪梯度 · 递归自改进）的完整实现：

核心三主体：
1. 元认知向量 m = (c, e, n)
   - c certainty   : 对当前策略的把握
   - e effectiveness: 近期有效程度
   - n novelty     : 探索的新颖度
2. 可微内在奖励梯度：基于需求满足 + 元认知分量构成的内驱
3. 递归自改进：提案 → 沙盒评估 → 采纳/拒绝，带安全约束

升级自原简化版（只有 exploration/learning rate）为完整元认知模型。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger("superbrain.self_tune")


@dataclass
class MetaCognitionVector:
    """元认知向量 m=(certainty, effectiveness, novelty)。"""
    certainty: float = 0.5      # 对策略的把握 0..1
    effectiveness: float = 0.5  # 近期有效性 0..1
    novelty: float = 0.3        # 探索新颖度 0..1


@dataclass
class TuningState:
    exploration_rate: float = 0.15
    learning_rate: float = 0.1
    need_weights: Dict[str, float] = field(default_factory=dict)
    meta: MetaCognitionVector = field(default_factory=MetaCognitionVector)
    fitness_history: deque = field(default_factory=lambda: deque(maxlen=50))  # 有上限，防无限增长
    acceptance_ratio: float = 0.5   # 近期采纳率（安全约束参考）
    accepted: int = 0
    rejected: int = 0
    step: int = 0

    def snapshot(self) -> dict:
        return dict(
            exploration_rate=self.exploration_rate,
            learning_rate=self.learning_rate,
            need_weights=dict(self.need_weights),
            meta=self.meta.__dict__,
        )

    def restore(self, snap: dict) -> None:
        self.exploration_rate = snap["exploration_rate"]
        self.learning_rate = snap["learning_rate"]
        self.need_weights = dict(snap.get("need_weights", {}))
        m = snap.get("meta", {})
        self.meta = MetaCognitionVector(
            certainty=m.get("certainty", 0.5),
            effectiveness=m.get("effectiveness", 0.5),
            novelty=m.get("novelty", 0.3),
        )


class SelfTuner:
    """EG-MRSI 完整版：元认知向量驱动 + 内在奖励梯度 + 安全约束自改进。"""

    EXPLOIT_BASE = 0.15   # 开发主导时的探索率下限
    EXPLORE_MAX = 0.5     # 探索爆发时的上限

    def __init__(self, interval: int = 20, min_improvement: float = 0.05) -> None:
        self.interval = interval
        self.min_improvement = min_improvement
        self.state = TuningState()

    def should_tune(self) -> bool:
        return self.state.step > 0 and self.state.step % self.interval == 0

    # ---- 元认知向量更新 ----

    def update_meta(self, mean_reward: float, success_rate: float,
                    novelty: float = 0.0) -> None:
        """按近期表现更新元认知向量。
        - effectiveness ← 近3步成功率（EMA）
        - certainty ← 最近轨迹稳定性（波动小→确定）
        - novelty ← 传入的新颖度/自发探索信号
        """
        m = self.state.meta
        m.effectiveness = m.effectiveness * 0.7 + success_rate * 0.3
        # 确定性：Reward 波动小则确定（用历史标准差的反向）
        if len(self.state.fitness_history) >= 4:
            recent = list(self.state.fitness_history)[-4:]
            spread = max(recent) - min(recent)
            m.certainty = max(0.05, min(0.95, 0.9 - spread))
        m.novelty = max(0.0, min(1.0, novelty))

    # ---- 内在奖励（可微梯度信号）----

    def intrinsic_reward(self, need_satisfaction: float,
                         novelty: float = 0.0) -> float:
        """内驱奖励 = α·需求满足 + β·好奇(novelty) + γ·探索期激励。"""
        m = self.state.meta
        alpha, beta, gamma = 1.0, 0.5, 0.3
        r = alpha * need_satisfaction + beta * novelty
        # 低把握且高有效时给探索激励（避免过早收敛）
        if m.certainty < 0.4 and m.effectiveness > 0.4:
            r += gamma * (1.0 - m.certainty)
        return max(-1.0, min(1.0, r))

    # ---- 元认知驱动的探索率 ----

    def _target_exploration(self) -> float:
        """应由元认知驱动的探索率：不确定且有效→多探索；确定→少探索。"""
        m = self.state.meta
        if m.effectiveness < 0.5:
            return self.EXPLORE_MAX                  # 效果差·要赶紧探索
        if m.certainty > 0.8:
            return self.EXPLOIT_BASE                 # 很确定→稳定开发
        return self.EXPLOIT_BASE + (1 - m.certainty) * (self.EXPLORE_MAX - self.EXPLOIT_BASE)

    def current_fitness(self, mean_reward: float, success_rate: float,
                        need_satisfaction: float = 0.0) -> float:
        """复合适应度 = 0.5×奖励 + 0.3×成功率 + 0.2×内驱与元认知协调。"""
        intrinsic = self.intrinsic_reward(need_satisfaction)
        m = self.state.meta
        metacog_align = m.effectiveness * (1 - abs(m.certainty - 0.7))  # 有效且不太飘
        return 0.5 * mean_reward + 0.3 * success_rate + 0.2 * (intrinsic + metacog_align) / 2

    def propose(self, novelty: float = 0.0) -> dict:
        """生成候选调参（探索率按元认知动态目标，学习率平滑）。"""
        m = self.state.meta
        target = self._target_exploration()
        # 逼近目标的梯度步长
        cur = self.state.exploration_rate
        new_e = cur + (target - cur) * 0.3
        # 学习率：有效时微调、无效时回调
        new_lr = self.state.learning_rate * (1.15 if m.effectiveness > 0.5 else 0.85)
        return {"exploration_rate": round(new_e, 4), "learning_rate": round(new_lr, 4)}

    def evaluate_and_apply(self, before_fitness: float, after_fitness: float,
                           proposal: dict, novelty: float = 0.0) -> dict:
        """对比适应度采纳/拒绝，加安全约束（防止探索率失控）。"""
        # 安全约束：探索率必须保持在下限内
        safe_e = max(self.EXPLOIT_BASE * 0.9, min(self.EXPLORE_MAX, proposal["exploration_rate"]))
        proposal["exploration_rate"] = safe_e
        m = self.state.meta
        # 动态阈值：把握高→要求更高提升；把握低→容忍更大
        threshold = self.min_improvement * (0.7 + m.certainty * 0.6)
        delta = after_fitness - before_fitness
        accepted = (delta > threshold and delta > 0)
        if accepted:
            self.state.exploration_rate = safe_e
            self.state.learning_rate = proposal["learning_rate"]
            self.state.accepted += 1
            m.certainty = min(0.95, m.certainty + 0.03)      # 有效则更确定
        else:
            self.state.exploration_rate = max(self.EXPLOIT_BASE * 0.5,
                                              self.state.exploration_rate * 0.9)
            self.state.rejected += 1
            m.novelty = min(1.0, m.novelty + 0.05)           # 失败则更想探索
        # 维护采纳率（安全约束参考）
        total = self.state.accepted + self.state.rejected
        if total > 0:
            self.state.acceptance_ratio = self.state.accepted / total
        self.state.step += 1
        self.state.fitness_history.append(after_fitness)
        return {"accepted": accepted, "delta": round(delta, 4),
                "exploration_rate": round(self.state.exploration_rate, 4),
                "certainty": round(m.certainty, 4),
                "novelty": round(m.novelty, 4)}

    def report(self) -> dict:
        m = self.state.meta
        return {
            "exploration_rate": round(self.state.exploration_rate, 4),
            "learning_rate": round(self.state.learning_rate, 4),
            "meta": {"certainty": round(m.certainty, 4),
                     "effectiveness": round(m.effectiveness, 4),
                     "novelty": round(m.novelty, 4)},
            "acceptance_ratio": round(self.state.acceptance_ratio, 4),
            "accepted": self.state.accepted,
            "rejected": self.state.rejected,
            "step": self.state.step,
        }