"""多尺度记忆表达层：记忆的尺度特征决定「如何开口」。

检索到的记忆，不止要"注入内容"，还要让 agent 根据记忆的尺度特征
校准表达方式——确定性、时态、语气、熟悉度。

尺度维度：
- 时间尺度：最新 vs 久远 → 表达的新鲜感/时态
- 置信度尺度：高 vs 低 → 表达的确定性（断言 vs 保留）
- 情绪尺度：记忆的情绪价 → 语气基调
- 熟悉度尺度：记忆数量/热度 → 熟练 vs 谨慎
- 粒度尺度：精确 vs 模糊（检索熵）→ 具体 vs 概括
"""

from __future__ import annotations

import time
from typing import List, Tuple

from .node import MemoryNode


def analyze_memory_scale(hits: List[Tuple[MemoryNode, float, str]]) -> dict:
    """分析检索记忆的多尺度特征，返回一个表达校准信号。

    Args:
        hits: [(node, score, why)] 检索命中的记忆。
    Returns:
        {
          "familiarity": 0-1  熟悉度（命中数量/质量）
          "confidence": 0-1    平均置信度 → 表达确定性
          "freshness": 0-1     时间新鲜度（1=刚刚，0=久远）
          "valence": -1..1     情绪基调（记忆情绪价均值）
          "granularity": 0-1   粒度（1=具体，0=概括）
        }
    """
    if not hits:
        return {"familiarity": 0.0, "confidence": 0.5, "freshness": 0.0,
                "valence": 0.0, "granularity": 0.5}

    n = len(hits)
    now = time.time()
    confs = [node.confidence for node, _, _ in hits if node.confidence > 0]
    vals = [node.valence for node, _, _ in hits]
    # 时间新鲜度：最近一条记忆距现在
    ages_h = [(now - node.last_access) / 3600.0 for node, _, _ in hits]
    newest_h = min(ages_h)
    freshness = max(0.0, 1.0 - newest_h / 168.0)  # 7天为满新鲜
    # 熟悉度：命中数量 + 分数
    avg_score = sum(s for _, s, _ in hits) / n
    familiarity = min(1.0, 0.3 * min(n, 5) / 5 + 0.7 * avg_score)

    return {
        "familiarity": round(familiarity, 3),
        "confidence": round(sum(confs) / len(confs) if confs else 0.5, 3),
        "freshness": round(freshness, 3),
        "valence": round(sum(vals) / n if vals else 0.0, 3),
        "granularity": 0.5,  # 默认中性，由调用方按检索熵覆盖
    }


def expression_guide(scale: dict) -> str:
    """根据多尺度特征生成「开口指导」，注入 prompt 校准表达。

    返回一段简短的自然语言指导，让 agent 的表达方式和记忆状态对齐。
    """
    fam = scale["familiarity"]
    conf = scale["confidence"]
    fresh = scale["freshness"]
    val = scale["valence"]

    # 完全无记忆（familiarity=0）时不给开口指导
    if fam <= 0.0:
        return ""

    parts = []
    # 确定性：置信度决定断言 vs 保留
    if conf >= 0.75:
        parts.append("你对此有高置信记忆，可以确定地表达")
    elif conf <= 0.4:
        parts.append("你对此记忆置信较低，表达时应带保留，不要断言")
    else:
        parts.append("你对此有一定了解，表达时适度肯定")

    # 熟悉度：熟练 vs 谨慎
    if fam >= 0.7:
        parts.append("这是你熟悉的话题")
    elif fam <= 0.25:
        parts.append("你对此较陌生，表达宜谨慎")

    # 新鲜度：时态/新鲜感
    if fresh >= 0.7:
        parts.append("这是最近发生/最近的记忆")
    elif fresh <= 0.2:
        parts.append("这是较早的记忆，注意区分过去的我")

    # 情绪基调
    if val >= 0.3:
        parts.append("相关记忆偏积极，可带正面情绪")
    elif val <= -0.3:
        parts.append("相关记忆偏消极，语气宜克制/共情")

    if not parts:
        return ""
    return "【开口指导】" + "；".join(parts)
