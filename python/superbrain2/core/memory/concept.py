"""语义概念抽取 + 关系建边（GraphRAG 思路，真增量）。

让 LLM 从记忆内容里自动抽取实体/概念，并在概念间建立关系边，
提升跨记忆推理能力——不只靠向量相似度，还能靠"概念图"找到
「A 和 B 都涉及同一个实体」这类向量检索找不到的关联。
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple


EXTRACT_PROMPT = """从下面这段记忆里抽取关键实体/概念，以及它们之间的关系。

记忆：{content}

只输出 JSON，格式：
{{"entities": ["实体1", "实体2"], "relations": [["实体1", "关系", "实体2"]]}}

要求：
- 实体是人名、项目名、工具名、技术名词、地点等关键概念
- 关系是动词短语（如"使用""负责""属于""依赖"）
- 实体 1-5 个，关系 0-5 个
- 不要输出 JSON 以外的内容"""


class ConceptExtractor:
    """语义概念抽取器（LLM 驱动，失败回退规则）。"""

    def __init__(self, llm) -> None:
        self.llm = llm

    def extract(self, content: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """抽取实体列表 + 关系三元组。失败回退空。"""
        try:
            prompt = EXTRACT_PROMPT.format(content=content[:500])
            resp = self.llm.chat([{"role": "user", "content": prompt}])
            return self._parse(resp.content)
        except Exception:
            return [], []

    @staticmethod
    def _parse(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return [], []
        try:
            d = json.loads(text[start:end + 1])
            entities = [str(e) for e in d.get("entities", []) if str(e).strip()]
            relations = []
            for r in d.get("relations", []):
                if isinstance(r, list) and len(r) == 3:
                    relations.append((str(r[0]), str(r[1]), str(r[2])))
            return entities, relations
        except (json.JSONDecodeError, AttributeError):
            return [], []


class ConceptGraph:
    """概念图：实体节点 + 关系边，叠加在 MemoryStore 之上。

    每个实体是 store 里的一个 MemoryNode（kind=entity），
    关系是 edges（kind=entity 边，evidence 存关系名）。
    """

    def __init__(self, store) -> None:
        self.store = store

    def index_memory(self, node, entities: List[str],
                     relations: List[Tuple[str, str, str]]) -> int:
        """把一条记忆的概念写进图：实体→节点，关系→边。返回建的关系数。"""
        entity_ids: Dict[str, str] = {}
        for e in entities:
            enode = self._ensure_entity(e)
            entity_ids[e] = enode.node_id
            # 记忆 ↔ 实体 关联边
            self.store.add_edge(node.node_id, enode.node_id, 1.0, kind="entity", evidence="mentions")
        count = 0
        for s, rel, o in relations:
            sid = entity_ids.get(s)
            oid = entity_ids.get(o)
            if sid and oid and sid != oid:
                self.store.add_edge(sid, oid, 1.0, kind="entity", evidence=rel)
                count += 1
        return count

    def _ensure_entity(self, name: str):
        """确保实体节点存在（用指纹去重）。"""
        from .node import MemoryNode, fingerprint
        fp = fingerprint(name)[:16]
        existing = self.store.get(fp)
        if existing is not None:
            return existing
        node = MemoryNode(node_id=fp, content=name, kind="entity",
                          tier="core", confidence=1.0)
        self.store.add(node)
        return node

    def expand(self, node_id: str, hops: int = 1) -> List:
        """从一条记忆出发，沿概念图扩展一跳，找相关记忆（跨记忆推理）。"""
        seen = {node_id}
        result = []
        frontier = [node_id]
        for _ in range(hops):
            nxt = []
            for nid in frontier:
                for neighbor, w in self.store.neighbors(nid):
                    if neighbor.node_id in seen:
                        continue
                    seen.add(neighbor.node_id)
                    result.append((neighbor, w))
                    nxt.append(neighbor.node_id)
            frontier = nxt
        return result
