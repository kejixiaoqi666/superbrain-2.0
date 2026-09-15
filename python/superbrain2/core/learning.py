"""自动学习：从对话抽取种子/关系/事实（事件→种子闭环的自动化）。

让 agent 每轮对话后自动沉淀该记住的东西，而不是靠外部手动调用：
- facts：值得记住的事实
- seeds：经历 → 行为倾向（遇到 X 应该 Y）
- relationships：对说话者的了解

由 LLM 抽取结构化学习信号，框架层写入。可关闭（省 token）。
"""

from __future__ import annotations

import json

from .llm import LLMProvider


EXTRACT_PROMPT = """你是记忆抽取器。从下面这段对话中，提取值得长期保存的学习信号。

只提取【确凿】的内容，不推断、不编造、不把推测当事实。

输出严格 JSON，格式：
{
  "facts": ["事实1", "事实2"],
  "seeds": [{"content": "遇到X应该Y的倾向", "trigger": "触发词", "bias": "行为方向", "valence": 0.5}],
  "relationships": [{"person": "说话者id", "note": "对TA的了解"}]
}

没有可提取的就返回空数组。只返回 JSON，不要其他文字。"""


class LearningExtractor:
    """用 LLM 从对话抽取学习信号。"""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def extract(self, user_msg: str, assistant_reply: str) -> dict:
        """抽取学习信号，返回 {facts, seeds, relationships}。"""
        convo = f"用户：{user_msg}\n助手：{assistant_reply}"
        try:
            resp = self.llm.chat([
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": convo},
            ], max_tokens=2000)
            text = (resp.content or "").strip()
            # 剥离可能的 markdown 代码块围栏
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            return self._parse(text)
        except Exception:
            return {"facts": [], "seeds": [], "relationships": []}

    @staticmethod
    def _parse(text: str) -> dict:
        """宽容解析：截取第一个 { 到最后一个 }，容错 JSON。"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {"facts": [], "seeds": [], "relationships": []}
