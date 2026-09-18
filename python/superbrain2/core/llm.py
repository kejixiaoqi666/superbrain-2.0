"""LLM Provider 抽象层：多通道可插拔，默认 DeepSeek，支持 function calling。

function-calling 协议：
- 请求带 tools 参数（JSON Schema）
- 模型返回 tool_calls（finish_reason="tool_calls"）
- 工具结果作为 role="tool" 消息回传
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToolCall:
    """一次工具调用。"""
    id: str
    name: str
    arguments: str  # JSON 字符串

    def args_dict(self) -> dict:
        try:
            return json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            return {}


@dataclass
class LLMResponse:
    content: str
    reasoning: str = ""
    finish_reason: str = "stop"
    tool_calls: List[ToolCall] = field(default_factory=list)
    # 真实 token 用量（provider 返回的 usage）——"当轮真实输入"的权威计量源，
    # 供上游 super-agent 判断是否自动续接新会话。缺省 0 = 提供方未返回。
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMProvider:
    """聊天补全客户端（支持 function calling）。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.7) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
             max_tokens: int = 2048) -> LLMResponse:
        """对话，可带 tools（function calling）。

        messages 支持标准角色：system/user/assistant/tool（tool 需带 tool_call_id）。
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        choice = data["choices"][0]
        msg = choice.get("message", {})
        tool_calls = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=fn.get("arguments", "{}"),
            ))
        return LLMResponse(
            content=msg.get("content", "") or "",
            reasoning=msg.get("reasoning_content", "") or "",
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=tool_calls,
            prompt_tokens=int((data.get("usage") or {}).get("prompt_tokens", 0) or 0),
            completion_tokens=int((data.get("usage") or {}).get("completion_tokens", 0) or 0),
            total_tokens=int((data.get("usage") or {}).get("total_tokens", 0) or 0),
        )


def from_env() -> "LLMProvider":
    """从环境变量构造默认 DeepSeek 通道。"""
    base = os.environ.get("SUPERBRAIN_LLM_BASE", "https://zzzzz.dpdns.org/v1")
    key = os.environ.get("SUPERBRAIN_LLM_KEY", os.environ.get("DPDNS_DEEPSEEK_API_KEY", ""))
    model = os.environ.get("SUPERBRAIN_LLM_MODEL", "deepseek-v4-pro")
    if not key:
        raise RuntimeError("缺少 LLM key：请设置 SUPERBRAIN_LLM_KEY 或 DPDNS_DEEPSEEK_API_KEY")
    return LLMProvider(base, key, model)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文≈1，英文≈4字符1个）。"""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return int(cjk + (len(text) - cjk) / 4)


def _content_text(content) -> str:
    """从消息 content 取纯文本（兼容多模态 list）。"""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def image_content_part(data_url: str) -> dict:
    """构造 OpenAI 兼容的多模态图片片段（base64 data URL）。"""
    return {"type": "image_url", "image_url": {"url": data_url}}


def multimodal_message(text: str, image_data_urls: List[str]) -> dict:
    """构造带图的多模态 user 消息：text + 若干 image_url 片段。"""
    parts: List[dict] = []
    if text:
        parts.append({"type": "text", "text": text})
    for u in image_data_urls:
        parts.append(image_content_part(u))
    return {"role": "user", "content": parts}


def to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """图片字节 → base64 data URL（直接喂模型，无需上传/OCR）。"""
    import base64
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"
