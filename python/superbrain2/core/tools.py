"""行动层：工具系统 + 权限（让超脑有「手」）。

Tool 抽象 + 注册表 + 权限门禁。工具是 agent 作用于世界的能力，
没有它，需求驱动是空的（算出了"想做"却无法"去做"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    """一个可调用的工具。"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Callable
    category: str = "general"
    # 权限：是否需要用户批准才执行（危险操作）
    requires_approval: bool = False
    # 副作用声明
    side_effects: str = "none"  # none | read | write | exec

    def call(self, **kwargs) -> Any:
        return self.handler(**kwargs)

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list(self) -> List[Tool]:
        return list(self._tools.values())

    def call(self, name: str, **kwargs) -> Any:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"未知工具: {name}")
        return tool.call(**kwargs)

    def openai_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]


@dataclass
class PermissionPolicy:
    """权限策略：哪些工具需要批准。"""
    require_approval: set = field(default_factory=lambda: {"exec", "write_file", "delete"})

    def needs_approval(self, tool_name: str, category: str, side_effects: str) -> bool:
        if tool_name in self.require_approval:
            return True
        if side_effects in ("exec", "write"):
            return True
        return False


# ---------- 内置工具 ----------

def _tool_exec(command: str) -> str:
    """执行 shell 命令（危险，需批准）。"""
    import subprocess
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "[超时] 命令执行超过 30 秒"
    except Exception as e:
        return f"[错误] {e}"


def _tool_read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:5000]
    except Exception as e:
        return f"[错误] {e}"


def _tool_write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}"
    except Exception as e:
        return f"[错误] {e}"


def register_builtin_tools(registry: ToolRegistry) -> None:
    """注册内置基础工具。"""
    registry.register(Tool(
        name="exec", description="执行 shell 命令",
        parameters={"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"]},
        handler=_tool_exec, category="system", requires_approval=True, side_effects="exec",
    ))
    registry.register(Tool(
        name="read_file", description="读取文件内容",
        parameters={"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]},
        handler=_tool_read_file, category="file", side_effects="read",
    ))
    registry.register(Tool(
        name="write_file", description="写入文件",
        parameters={"type": "object", "properties": {"path": {"type": "string"},
                    "content": {"type": "string"}}, "required": ["path", "content"]},
        handler=_tool_write_file, category="file", requires_approval=True, side_effects="write",
    ))
