"""超脑 SuperBrain —— 记忆节点、边类型与嵌入工具。

融合四大记忆框架的精髓字段：
- scope (Mem0): user / session / agent 三作用域
- tier (Letta): core / recall / archival 三层
- valid_from / valid_until (Zep): 时序有效窗口
- 边类型枚举 (supersede 取代等)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List


# 边类型枚举（内容冻结下用边表达关系，不改写原始记忆）
EDGE_KINDS = ("semantic", "cooccur", "entity", "handoff", "supersede")

# 三作用域（Mem0）：user=跨会话持久 / session=单会话 / agent=智能体自身
SCOPES = ("user", "session", "agent")

# 三层（Letta OS 式）：core=常驻 / recall=最近访问 / archival=长期归档
TIERS = ("core", "recall", "archival")


@dataclass
class MemoryNode:
    """一条记忆。内容冻结：content 一经写入永不改写。"""

    node_id: str = ""
    content: str = ""
    embedding: List[float] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    scene: str = "personal"
    confidence: float = 1.0
    kind: str = "fact"  # fact | procedure | handover
    # Mem0 作用域 + Letta 分层 + Zep 时序窗口
    scope: str = "user"          # user | session | agent
    tier: str = "recall"         # core | recall | archival
    valence: float = 0.0         # 情绪标签 -1..1（Sentipolis 情绪-记忆耦合）
    valid_from: float = 0.0      # 生效时间戳（0=永久）
    valid_until: float = 0.0     # 失效时间戳（0=永久有效）
    # 艾宾浩斯遗忘曲线参数（PowerMem）
    retention_strength: float = 1.0  # 记忆强度 S，随复习增强
    review_count: int = 0            # 复习次数（间隔重复）
    created_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "content": self.content,
            "embedding": self.embedding, "tags": self.tags,
            "scene": self.scene, "confidence": self.confidence, "kind": self.kind,
            "scope": self.scope, "tier": self.tier,
            "valence": self.valence,
            "valid_from": self.valid_from, "valid_until": self.valid_until,
            "retention_strength": self.retention_strength,
            "review_count": self.review_count,
            "created_at": self.created_at, "last_access": self.last_access,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNode":
        return cls(
            node_id=d.get("node_id", ""), content=d.get("content", ""),
            embedding=d.get("embedding", []), tags=d.get("tags", []),
            scene=d.get("scene", "personal"), confidence=d.get("confidence", 1.0),
            kind=d.get("kind", "fact"),
            scope=d.get("scope", "user"), tier=d.get("tier", "recall"),
            valence=d.get("valence", 0.0),
            valid_from=d.get("valid_from", 0.0), valid_until=d.get("valid_until", 0.0),
            retention_strength=d.get("retention_strength", 1.0),
            review_count=d.get("review_count", 0),
            created_at=d.get("created_at", 0.0), last_access=d.get("last_access", 0.0),
            access_count=d.get("access_count", 0),
        )

    def is_valid(self, now: float = 0.0) -> bool:
        """时序有效性（Zep）：当前时间戳是否在有效窗口内。"""
        import time as _time
        t = now or _time.time()
        if self.valid_from and t < self.valid_from:
            return False
        if self.valid_until and t > self.valid_until:
            return False
        return True


def fingerprint(content: str) -> str:
    """节点语义指纹：空白与大小写归一后的内容哈希（内容冻结的去重键）。"""
    normalized = re.sub(r"\s+", "", content.lower())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def _gram_set(text: str, n: int = 3) -> set:
    """字符 n-gram 集合（关键词检索的零依赖实现）。"""
    t = re.sub(r"\s+", "", text.lower())
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}
