"""底层表达元素库：分类的简单言语模板。

定位：这是「言语底层框架」，不是锁死的人设。
提供简单、可组合的表达元素（称呼/语气/情感/关心/深情），
超脑根据**自主关系定位(orientation)+情绪**灵活选用，不强制、不锁定。

元素内容只是简单的模板词汇；具体什么时候用、怎么组合，由超脑自主决定。
关系越深（orientation 越高），可选用的元素类别越丰富——不是"设定成恋人就必须说情话"，
而是"关系相处到 lover 级，超脑自然可以选用深情/亲昵的表达元素"。
"""

from __future__ import annotations

from typing import Dict, List

# 分类的底层表达元素（简单的言语模板）
_EXPRESSION_ELEMENTS: Dict[str, List[str]] = {
    "endearment": ["宝宝", "亲爱的", "宝贝"],                    # 亲昵称呼
    "coquetry": ["哼", "人家", "要抱抱", "不许不理我"],           # 撒娇语气
    "care": ["要照顾好自己", "有没有想我", "今天累不累"],          # 关心
    "affection": ["你来了灯就亮了", "有你在真好", "我只想陪着你"],   # 深情
    "intimacy": ["贴贴", "抱抱", "牵牵手"],                      # 亲昵动作
    "joy": ["嘿嘿", "开心", "好耶"],                              # 欢快
    "sadness": ["呜呜", "难过", "委屈"],                          # 低落
    "casual": ["哈哈哈", "笑死", "真有你的"],                      # 轻松
}

# 关系定位 → 可选用的元素类别（关系越深，可选越丰富；这是「范围」不是「强制」）
_ORIENTATION_SCOPE: Dict[str, List[str]] = {
    "stranger": [],
    "acquaintance": ["joy", "sadness"],
    "friend": ["joy", "sadness", "casual", "care"],
    "close": ["joy", "sadness", "casual", "care", "endearment", "intimacy"],
    "lover": ["joy", "sadness", "casual", "care", "endearment",
              "intimacy", "coquetry", "affection"],
}


def all_categories() -> List[str]:
    """返回所有底层表达元素类别（框架清单，供上层查看有哪些元素类型）。"""
    return list(_EXPRESSION_ELEMENTS.keys())


def all_elements() -> Dict[str, List[str]]:
    """返回全部分类元素（不区分关系定位），供上层查看完整底层模板清单。"""
    return {cat: list(items) for cat, items in _EXPRESSION_ELEMENTS.items()}


def categories_for(orientation: str) -> List[str]:
    """返回某关系定位下可选用的元素类别（超脑自主选用的范围）。"""
    if not isinstance(orientation, str):
        return []  # 脏 orientation（如 list/dict 不可 hash）防御
    return list(_ORIENTATION_SCOPE.get(orientation, []))


def elements_for(orientation: str) -> Dict[str, List[str]]:
    """返回某关系定位下可用的表达元素（类别→元素列表）。

    这是「可用范围」：超脑可以从中自主选用，而非被强制套用。
    """
    if not isinstance(orientation, str):
        return {}  # 脏 orientation 防御
    return {cat: list(_EXPRESSION_ELEMENTS[cat])
            for cat in _ORIENTATION_SCOPE.get(orientation, [])}


def pick(orientation: str, category: str, index: int = 0) -> str:
    """超脑按关系定位+类别自主选用一个底层表达元素。

    index 用于确定性选择（默认 0=第一个）；上层可用随机 index 体现「自主」。
    关系定位不匹配该类别时返回空串（越界不自作主张）。
    """
    if not isinstance(orientation, str) or not isinstance(category, str):
        return ""  # 脏 orientation/category 防御
    if category not in _ORIENTATION_SCOPE.get(orientation, []):
        return ""
    items = _EXPRESSION_ELEMENTS.get(category, [])
    if not items:
        return ""
    return items[index % len(items)]
