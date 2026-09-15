"""时间体感：时间戳 → 模糊时间描述（"刚才/昨天/小时候"）。

真实大脑有时间流逝的体感，不是冷冰冰的秒数。
把时间戳转成自然的模糊时间，让记忆/事件带"时间感"。

与遗忘曲线的区别：遗忘曲线处理"时间如何衰减记忆强度"，
这里处理"时间如何被体感描述"——互补，不重复。
"""

from __future__ import annotations

import time
from typing import Optional


def fuzzy_time(ts: Optional[float], now: Optional[float] = None) -> str:
    """把时间戳转成模糊时间描述。"""
    if ts is None or ts <= 0:
        return "未知时间"
    t = now or time.time()
    delta = t - ts  # 秒
    if delta < 0:
        return "将来"
    if delta < 60:
        return "刚才"
    if delta < 3600:
        return f"{int(delta // 60)}分钟前"
    if delta < 86400:
        return f"{int(delta // 3600)}小时前"
    days = delta / 86400
    if days < 1:
        return "今天"
    if days < 2:
        return "昨天"
    if days < 7:
        return f"{int(days)}天前"
    if days < 30:
        return f"{int(days // 7)}周前"
    if days < 365:
        return f"{int(days // 30)}个月前"
    years = days / 365
    if years < 10:
        return f"{int(years)}年前"
    return "很久以前"


def temporal_context(ts: Optional[float]) -> str:
    """时间上下文描述（用于记忆注入，带体感）。"""
    if ts is None or ts <= 0:
        return ""
    return f"（{fuzzy_time(ts)}）"


def age_in_words(created_at: float, now: Optional[float] = None) -> str:
    """记忆的年龄描述，带成长感（用于自我叙事的时间身份）。"""
    t = now or time.time()
    days = (t - created_at) / 86400
    if days < 0:
        return "刚开始"
    if days < 1:
        return "刚开始"
    if days < 7:
        return "刚认识不久"
    if days < 30:
        return f"认识了{int(days)}天"
    if days < 365:
        return f"认识了{int(days // 30)}个月"
    return f"认识了{int(days // 365)}年"
