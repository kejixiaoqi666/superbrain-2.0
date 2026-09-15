"""上下文清洗：进上下文前清理结构噪声（TokenPilot Ingestion-Aware Compaction）。

上下文污染让模型迷失的根源：
- 工具输出的 JSON/日志/堆栈（结构噪声）原样进上下文
- 报错/失败消息反复累积
- 旧的工具结果一直留在历史里

在"进上下文"这一刻清洗，而不是等上下文爆炸后再压缩。
"""

from __future__ import annotations

import json
from typing import Optional


def clean_tool_output(text: str, max_chars: int = 400) -> str:
    """清洗工具输出：剥离结构噪声，保留价值密度高的部分。

    - JSON → 提取关键字段摘要
    - 堆栈/日志 → 截断到前几行
    - 空/纯空白 → 丢弃
    - 过长 → 截断 + 标记
    """
    if not text or not text.strip():
        return ""

    t = text.strip()

    # 错误堆栈：只保留最后一行（真正的错误信息）
    if "Traceback" in t or "Error:" in t or "Exception" in t:
        lines = [l for l in t.splitlines() if l.strip()]
        # 取最后 1-2 行（通常是真正的错误原因）
        tail = lines[-2:] if len(lines) > 1 else lines
        return "[错误] " + " | ".join(l.strip() for l in tail)[:max_chars]

    # JSON：尝试解析，提取摘要
    if t.startswith("{") or t.startswith("["):
        try:
            d = json.loads(t)
            return _summarize_json(d, max_chars)
        except json.JSONDecodeError:
            pass

    # 普通文本：截断
    if len(t) > max_chars:
        return t[:max_chars] + f"...（已截断 {len(t)} 字符）"
    return t


def _summarize_json(d, max_chars: int) -> str:
    """JSON 摘要：只保留关键字段，避免整段 JSON 污染上下文。"""
    if isinstance(d, dict):
        # 挑少量关键字段
        keys = list(d.keys())[:8]
        parts = []
        for k in keys:
            v = d[k]
            if isinstance(v, (dict, list)):
                v = f"<{type(v).__name__}>"
            elif isinstance(v, str) and len(v) > 50:
                v = v[:50] + "..."
            parts.append(f"{k}={v}")
        return "{" + ", ".join(parts) + "}"
    elif isinstance(d, list):
        n = len(d)
        sample = ", ".join(str(x)[:40] for x in d[:3])
        return f"[{n}项: {sample}" + (", ...]" if n > 3 else "]")
    return str(d)[:max_chars]


def clean_history_message(content: str, role: str, max_chars: int = 300) -> Optional[str]:
    """清洗历史消息：旧消息降级，剥离噪声。

    - tool 消息：用 clean_tool_output 清洗
    - 超长消息：截断
    - 纯噪声：返回 None（丢弃）
    """
    if role == "tool":
        return clean_tool_output(content, max_chars)
    if not content or not content.strip():
        return None
    if len(content) > max_chars * 2:  # 历史长消息更激进截断
        return content[:max_chars] + "..."
    return content


def is_noise(content: str) -> bool:
    """判断是否纯噪声（无信息量，可丢弃）。"""
    if not content or not content.strip():
        return True
    t = content.strip().lower()
    # 纯标点/纯空行/极短无意义
    if len(t) <= 2:
        return True
    # 纯重复字符
    if len(set(t)) <= 2 and len(t) > 3:
        return True
    return False
