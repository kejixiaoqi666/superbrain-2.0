"""记忆 Schema 演化（PISA / 皮亚杰图式的工程化）。

记忆不只是积累节点，还要让"结构"随经验自适应地演化。皮亚杰的三模态：
- 同化(assimilate)：新信息并入已有 schema（归类到现成模板）
- 顺应(accommodate)：新信息不匹配 → 扩展/修正 schema（加字段、调关系）
- 创造(create)：完全没有 schema → 建新的 schema（新模式）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Schema:
    """一个记忆图式（模板）：某类经验的抽象结构。"""
    id: str
    name: str                        # 如"用户偏好"、"服务节点"、"支付流程"
    fields: Dict[str, str] = field(default_factory=dict)  # 字段名->类型
    exemplars: List[str] = field(default_factory=list)    # 实例 node_id
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    update_count: int = 0

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "fields": self.fields,
                "exemplars": self.exemplars, "confidence": self.confidence,
                "created_at": self.created_at, "update_count": self.update_count}

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(id=d["id"], name=d.get("name", ""), fields=d.get("fields", {}),
                   exemplars=d.get("exemplars", []), confidence=d.get("confidence", 0.5),
                   created_at=d.get("created_at", 0.0), update_count=d.get("update_count", 0))


class SchemaEngine:
    """PISA 三模态演化引擎。"""

    SIMILARITY_FIELDS = 0.5   # 判断"匹配已有 schema"的字段重叠阈值

    def __init__(self) -> None:
        self.schemas: Dict[str, Schema] = {}

    @staticmethod
    def _grams(s: str, n: int = 2):
        s = "".join(s.lower().split())
        return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else {s}

    def _keywords(self, content: str) -> set:
        """粗提取关键词（前置词、域名、人名等出现的高价值词）。"""
        stopwords = {"的", "了", "是", "在", "吗", "啊", "我", "你", "它", "我们",
                     "一个", "这个", "那个", "什么", "怎么", "为什么", "关于"}
        tokens = [w for w in content.replace("，", " ").replace("。", " ").split() if w]
        return {w for w in tokens if w not in stopwords and len(w) > 1}

    def find_best_schema(self, content: str) -> Optional[Schema]:
        """找与内容最匹配的已有 schema（按关键词重叠）。"""
        kw = self._keywords(content)
        if not kw:
            return None
        best, best_score = None, 0.0
        for sch in self.schemas.values():
            s_kw = {"".join(field.split()) for field in sch.fields}
            overlap = len(kw & s_kw) / max(1, len(kw))
            if overlap > best_score:
                best, best_score = sch, overlap
        return best if best_score >= self.SIMILARITY_FIELDS else None

    def ingest(self, content: str, node_id: str = "") -> Tuple[str, str, str]:
        """三模态入口，返回 (mode, schema_id, schema_name)。
        - assimilate : 并入已有 schema
        - accommodate: 修正已有 schema（加字段）
        - create     : 建新 schema
        """
        best = self.find_best_schema(content)
        if best is None:
            # create：建新 schema，用内容头2词命名
            kw = sorted(self._keywords(content))[:2]
            name = "".join(kw) or "经验"
            sch = Schema(id=f"sch_{int(time.time()*1000)}", name=name,
                         fields={w: "keyword" for w in kw})
            self.schemas[sch.id] = sch
            sch.exemplars.append(node_id)
            return ("create", sch.id, sch.name)
        # 加入实例
        if node_id and node_id not in best.exemplars:
            best.exemplars.append(node_id)
        new_kw = self._keywords(content)
        existing = set(best.fields)
        # accommodate：若新内容带来不在 schema 里的关键词 → 扩展字段
        novel = new_kw - existing
        if novel:
            for w in novel:
                best.fields[w] = "keyword"
            best.update_count += 1
            best.confidence = min(1.0, best.confidence + 0.05)
            return ("accommodate", best.id, best.name)
        # assimilate：纯归类，轻微增强置信度
        best.confidence = min(1.0, best.confidence + 0.02)
        best.update_count += 1
        return ("assimilate", best.id, best.name)

    def all(self) -> List[Schema]:
        return list(self.schemas.values())

    def to_dict(self) -> dict:
        return {"schemas": {k: v.to_dict() for k, v in self.schemas.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaEngine":
        se = cls()
        se.schemas = {k: Schema.from_dict(v) for k, v in d.get("schemas", {}).items()}
        return se