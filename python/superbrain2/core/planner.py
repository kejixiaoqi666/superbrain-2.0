"""规划器：LLM 驱动规划。

区别于规则拆解，让 LLM 根据目标 + 可用工具动态生成可执行步骤列表。
标准做法：先 plan（出步骤），再 execute（逐步执行）。
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from .goals import Goal

logger = logging.getLogger("superbrain.planner")


PLAN_PROMPT = """你是任务规划器。根据目标，生成可执行步骤列表。

目标：{goal}

可用工具：{tools}

请只输出 JSON 数组，每个元素是一个步骤描述字符串，例如：
["步骤1", "步骤2", "步骤3"]

步骤要求：
- 具体、可执行
- 按依赖顺序排列
- 3-6 步为佳
- 不要输出 JSON 以外的内容"""


class LLMPlanner:
    """LLM 驱动的任务规划器。"""

    def __init__(self, llm) -> None:
        self.llm = llm

    def plan(self, goal: Goal, tool_names: Optional[List[str]] = None) -> List[str]:
        """用 LLM 生成步骤列表。失败时回退到规则拆解。"""
        tools = ", ".join(tool_names) if tool_names else "无（纯对话）"
        prompt = PLAN_PROMPT.format(goal=goal.description, tools=tools)
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}])
            steps = self._parse_steps(resp.content)
            if steps:
                return steps
        except Exception:
            logger.debug("规划 LLM 失败，回退规则拆解", exc_info=True)
        # 回退：规则拆解
        return ["分析需求", "执行核心动作", "验证结果"]

    @staticmethod
    def _parse_steps(content: str) -> List[str]:
        """从 LLM 输出解析步骤列表（容错 JSON）。"""
        text = content.strip()
        # 去掉可能的 markdown 代码块包裹
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        # 提取 JSON 数组
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            arr = json.loads(text[start:end + 1])
            if isinstance(arr, list):
                return [str(s) for s in arr if str(s).strip()]
        except json.JSONDecodeError:
            pass
        return []
