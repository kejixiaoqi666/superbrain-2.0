"""Opt-in prediction-error reconsolidation（预测误差门控的记忆再巩固）。

ZenBrain 精髓：检索使记忆痕迹不稳定，只有"意外反馈"（预测误差超阈值）
才改变强度/置信度；误差小则不动，避免无意义漂移。

修正点（相对子agent初版）：
- 写回 store 持久化（初版只改内存对象，重启丢失）
- retention 方向改对：误差大 → 稳定性下降（记忆被"动摇"），误差小才稳定
- 内容冻结：绝不改 content，只调 confidence/retention
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ReconsolidationResult:
    prediction_error: float
    updated: bool
    new_confidence: float
    new_retention: float


def reconsolidate(store, node_id: str, observed_confidence: float,
                  threshold: float = 0.25, learning_rate: float = 0.2) -> ReconsolidationResult:
    """预测误差门控的再巩固，写回 store 持久化。

    误差 ≤ 阈值：不动（记忆稳定）。
    误差 > 阈值：置信度向观察值移动，retention 随误差大小下调
    （误差大 = 记忆不可靠 = 稳定性降，符合"意外反馈动摇记忆"）。
    """
    node = store.get(node_id)
    if node is None:
        return ReconsolidationResult(0.0, False, 0.0, 0.0)

    observed = max(0.0, min(1.0, float(observed_confidence)))
    error = abs(node.confidence - observed)

    if error <= threshold:
        return ReconsolidationResult(error, False, node.confidence,
                                     node.retention_strength)

    new_conf = node.confidence + learning_rate * (observed - node.confidence)
    new_conf = max(0.0, min(1.0, new_conf))
    # 误差大 → 记忆被动摇 → retention 下降（但保底 0.1）
    new_ret = node.retention_strength * (1.0 - error * learning_rate)
    new_ret = max(0.1, new_ret)

    store.update_confidence(node_id, new_conf, new_ret)
    return ReconsolidationResult(error, True, new_conf, new_ret)
