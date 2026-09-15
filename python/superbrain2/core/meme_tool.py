"""表情包搜索工具：优先在线检索公开 memegen 模板库，失败/无匹配时降级到本地索引。

在线源为 memegen.link 的模板目录 API `GET https://api.memegen.link/templates`
（返回全量模板数组，含 name/keywords/blank），按 query 在模板名与关键词里匹配；
中文关键词在英文模板库里通常无命中，会自动降级到内置 twemoji 本地索引返回真实图片直链。
零依赖（仅 urllib），模板列表模块级缓存 10 分钟，网络异常优雅降级。

用法（供上层 agent 拿图发出）：
    from superbrain.core.meme_tool import search_meme
    hits = search_meme("cat", limit=3)   # [{"url","title","source"}, ...]
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import List

_LOCAL_INDEX = {
    "开心": [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f604.png", "title": "开心笑脸", "source": "twemoji"}],
    "哈哈": [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f606.png", "title": "大笑", "source": "twemoji"}],
    "笑":   [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f606.png", "title": "大笑", "source": "twemoji"}],
    "哭":   [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f62d.png", "title": "大哭", "source": "twemoji"}],
    "难过": [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f62d.png", "title": "大哭", "source": "twemoji"}],
    "生气": [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f620.png", "title": "生气脸", "source": "twemoji"}],
    "猫":   [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f431.png", "title": "猫咪表情", "source": "twemoji"}],
    "狗":   [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f436.png", "title": "狗狗表情", "source": "twemoji"}],
    "点赞": [{"url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f44d.png", "title": "点赞", "source": "twemoji"}],
}

_TEMPLATES_URL = "https://api.memegen.link/templates"
_CACHE = {"ts": 0.0, "templates": []}
_TTL = 600.0  # 模板列表缓存 10 分钟，避免每次搜索全量拉取


def _local(query: str, limit: int) -> List[dict]:
    found = []
    for keyword, items in _LOCAL_INDEX.items():
        if keyword in query:
            # 条目可能重复命中不同关键词，去重保序
            for it in items:
                if not any(it["url"] == e["url"] for e in found):
                    found.append(it)
    return found[:max(0, limit)]


def _fetch_templates(timeout: float) -> List[dict]:
    """拉取并缓存 memegen 模板列表；格式兼容 dict{results|templates}/list 两种响应。"""
    now = time.time()
    if _CACHE["templates"] and now - _CACHE["ts"] < _TTL:
        return _CACHE["templates"]
    # memegen 会 403 拒绝 urllib 默认 UA，需带浏览器 UA
    req = urllib.request.Request(_TEMPLATES_URL, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("templates") or payload.get("results") or []
    templates = payload if isinstance(payload, list) else []
    if templates:
        _CACHE["ts"] = now
        _CACHE["templates"] = templates
    return templates


def search_meme(query: str, limit: int = 5, timeout: float = 8.0) -> List[dict]:
    """搜索表情包；优先在线 memegen 模板库（name/keywords 匹配），失败/无命中降级内置索引。"""
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(limit, int) or limit <= 0:
        return []  # 脏 limit（如字符串）防御，不崩
    hits: List[dict] = []
    try:
        templates = _fetch_templates(timeout)
        q = query.strip().lower()
        matched = [t for t in templates if isinstance(t, dict) and (
            q in str(t.get("name") or "").lower()
            or any(q in str(kw or "").lower() for kw in (t.get("keywords") or [])))]
        for t in matched:
            url = t.get("blank") or (t.get("example") or {}).get("url")
            if url:
                hits.append({"url": url, "title": t.get("name", ""), "source": "memegen"})
    except Exception:
        pass  # 网络失败静默降级，不抛错
    if hits:
        return hits[:limit]
    return _local(query, limit)


def register_meme_tool(registry) -> None:
    """向工具注册表注册只读表情包搜索工具。"""
    from .tools import Tool

    def handler(query="", limit=5):
        return json.dumps(search_meme(query, limit), ensure_ascii=False)

    registry.register(Tool(name="search_meme",
                           description="搜索表情包并返回图片直链、描述和来源。",
                           parameters={"type": "object", "properties": {
                               "query": {"type": "string"},
                               "limit": {"type": "integer", "default": 5}},
                               "required": ["query"]},
                           handler=handler, category="media",
                           requires_approval=False, side_effects="read"))