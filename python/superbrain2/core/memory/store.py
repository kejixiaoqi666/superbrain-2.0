"""存储层：SQLite 单文件持久化（内容冻结 + scope/tier/时序/遗忘曲线字段）。"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import time
import heapq
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .node import MemoryNode, fingerprint
from .embeddings import cosine, dot_normalized

GAP_LIMIT = 20


def default_db_path() -> str:
    return os.environ.get("SUPERBRAIN_DB") or str(Path.home() / ".superbrain" / "memory.db")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id            TEXT PRIMARY KEY,
    content            TEXT NOT NULL,
    embedding          BLOB NOT NULL,
    tags               TEXT NOT NULL DEFAULT '[]',
    scene              TEXT NOT NULL DEFAULT 'personal',
    confidence         REAL NOT NULL DEFAULT 1.0,
    kind               TEXT NOT NULL DEFAULT 'fact',
    scope              TEXT NOT NULL DEFAULT 'user',
    tier               TEXT NOT NULL DEFAULT 'recall',
    valence            REAL NOT NULL DEFAULT 0,
    valid_from         REAL NOT NULL DEFAULT 0,
    valid_until        REAL NOT NULL DEFAULT 0,
    retention_strength REAL NOT NULL DEFAULT 1.0,
    review_count       INTEGER NOT NULL DEFAULT 0,
    created_at         REAL NOT NULL,
    last_access        REAL NOT NULL,
    access_count       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edges (
    src      TEXT NOT NULL,
    dst      TEXT NOT NULL,
    weight   REAL NOT NULL,
    kind     TEXT NOT NULL DEFAULT '',
    evidence TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (src, dst)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    content, node_id UNINDEXED,
    tokenize='trigram'
);
"""

_NODE_COLS = (
    "node_id, content, embedding, tags, scene, confidence, kind, scope, tier, "
    "valence, valid_from, valid_until, retention_strength, review_count, "
    "created_at, last_access, access_count"
)


def _pack_vec(vec: List[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_vec(blob: bytes) -> List[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class MemoryStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or default_db_path()
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._migrate_schema()   # 旧版库补列，保证升到新版不崩
        self._vec_cache: Dict[str, Tuple[bytes, List[float]]] = {}
        self._tx_depth = 0
        # 向量是否已归一化（HashingEmbedder 输出已归一化 → 用点积快路径）
        self.normalized = False
        self._node_count_cache: Optional[int] = None  # 节点数缓存（省 SQL COUNT）
        # 记忆库物理上限（护栏，默认 1GB，可配置）——配合对话浓缩机制，实际很难触达
        self.max_db_bytes: int = 1024 ** 3

    def db_size(self) -> int:
        """记忆库当前磁盘占用（字节）。"""
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def over_size_limit(self) -> bool:
        """是否超过记忆库物理上限（触发强制浓缩/遗忘）。"""
        return self.db_size() > self.max_db_bytes

    def _migrate_schema(self) -> None:
        """旧版库 schema 迁移：为缺失的新列补 ALTER TABLE（不重建表，保留数据）。

        `CREATE TABLE IF NOT EXISTS` 不会给已存在的旧表加列；旧版 brain.db 缺
        scope/tier/valence 等新列时，新版代码查询会报 `no such column`。这里逐列
        PRAGMA 检查，缺的用 ALTER TABLE ADD COLUMN 补上（带默认值）。
        """
        try:
            existing = {r[1] for r in
                        self.conn.execute("PRAGMA table_info(nodes)").fetchall()}
            # (列名, 完整定义) 与 _SCHEMA 保持一致
            needed = {
                "scope": "TEXT NOT NULL DEFAULT 'user'",
                "tier": "TEXT NOT NULL DEFAULT 'recall'",
                "valence": "REAL NOT NULL DEFAULT 0",
                "valid_from": "REAL NOT NULL DEFAULT 0",
                "valid_until": "REAL NOT NULL DEFAULT 0",
                "retention_strength": "REAL NOT NULL DEFAULT 1.0",
                "review_count": "INTEGER NOT NULL DEFAULT 0",
            }
            for col, ddl in needed.items():
                if col not in existing:
                    self.conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} {ddl}")
            self.conn.commit()
        except Exception:
            # 迁移失败不阻塞连接；极端情况下可重建库（数据迁移由上层负责）
            pass

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._tx_depth += 1
        try:
            yield self.conn
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.commit()
        except Exception:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.rollback()
            raise

    # ---------- 节点 ----------

    def add(self, node: MemoryNode) -> MemoryNode:
        fp = fingerprint(node.content)
        if self.get_by_fingerprint(fp) is not None:
            return node
        if not node.node_id:
            node.node_id = fp[:16]
        if not node.created_at:
            node.created_at = time.time()
        if not node.last_access:
            node.last_access = node.created_at
        blob = _pack_vec(node.embedding) if node.embedding else b""
        self.conn.execute(
            f"INSERT OR REPLACE INTO nodes ({_NODE_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                node.node_id, node.content, sqlite3.Binary(blob),
                json.dumps(node.tags, ensure_ascii=False), node.scene,
                node.confidence, node.kind, node.scope, node.tier,
                node.valence,
                node.valid_from, node.valid_until, node.retention_strength,
                node.review_count, node.created_at, node.last_access, node.access_count,
            ),
        )
        if node.embedding:
            self._vec_cache[node.node_id] = (blob, list(node.embedding))
        # 同步 FTS 全文索引
        self.conn.execute(
            "INSERT INTO nodes_fts(content, node_id) VALUES (?,?)",
            (node.content, node.node_id),
        )
        # 更新节点计数缓存
        if self._node_count_cache is not None:
            self._node_count_cache += 1
        self._commit_if_autonomous()
        return node

    def add_many(self, nodes: Iterable[MemoryNode]) -> List[MemoryNode]:
        """Add nodes in one transaction, avoiding one SQLite commit per node."""
        added = []
        with self.transaction():
            for node in nodes:
                added.append(self.add(node))
        return added

    def get_by_fingerprint(self, fp: str) -> Optional[MemoryNode]:
        """按内容指纹查节点（改为用 node_id 前缀直接查，O(1)）。"""
        return self.get(fp[:16])

    def keyword_search(self, query: str, k: int = 10) -> List[MemoryNode]:
        """FTS5 全文检索（替代 n-gram 全表扫描，O(log n)）。

        短查询（<3 字符）FTS5 trigram 无法建索引，降级为 LIKE 兜底。
        """
        # 短查询降级：FTS5 trigram 需要 ≥3 字符
        if len(query.strip()) < 3:
            rows = self.conn.execute(
                "SELECT node_id FROM nodes WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", k),
            ).fetchall()
        else:
            try:
                rows = self.conn.execute(
                    "SELECT node_id FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT ?",
                    (query, k),
                ).fetchall()
            except Exception:
                # FTS5 查询语法错误时降级为 LIKE
                rows = self.conn.execute(
                    "SELECT node_id FROM nodes WHERE content LIKE ? LIMIT ?",
                    (f"%{query}%", k),
                ).fetchall()
        out = []
        for (node_id,) in rows:
            n = self.get(node_id)
            if n is not None:
                out.append(n)
        return out

    def is_concept_indexed(self, node_id: str) -> bool:
        """节点是否已做概念抽取（有 entity 出边即已索引）。"""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src=? AND kind='entity'",
            (node_id,),
        ).fetchone()
        return row[0] > 0

    def get(self, node_id: str) -> Optional[MemoryNode]:
        row = self.conn.execute(
            f"SELECT {_NODE_COLS} FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def all_nodes(self) -> List[MemoryNode]:
        return [self._row_to_node(r) for r in self.conn.execute(
            f"SELECT {_NODE_COLS} FROM nodes").fetchall()]

    def count_nodes(self) -> int:
        """节点数，带内存缓存（省每轮 SQL COUNT 开销）。"""
        if self._node_count_cache is None:
            self._node_count_cache = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        return self._node_count_cache

    def touch(self, node_id: str) -> None:
        """访问一次记忆：更新时间、次数、复习（艾宾浩斯间隔重复）。"""
        node = self.get(node_id)
        if node is None:
            return
        node.access_count += 1
        node.review_count += 1
        node.last_access = time.time()
        # 复习增强记忆强度 S（遗忘曲线：R=e^(-t/S)，S 越大遗忘越慢）
        node.retention_strength = min(10.0, node.retention_strength + 0.5)
        with self.transaction():
            self.conn.execute(
                "UPDATE nodes SET last_access=?, access_count=?, review_count=?, "
                "retention_strength=? WHERE node_id=?",
                (node.last_access, node.access_count, node.review_count,
                 node.retention_strength, node_id),
            )

    # ---------- 置信度/强度 持久化（reconsolidation 用） ----------

    def update_confidence(self, node_id: str, confidence: float,
                          retention_strength: float) -> None:
        """持久化更新记忆的置信度和强度（reconsolidation 写回）。"""
        c = max(0.0, min(1.0, confidence))
        s = max(0.1, retention_strength)
        with self.transaction():
            self.conn.execute(
                "UPDATE nodes SET confidence=?, retention_strength=? WHERE node_id=?",
                (c, s, node_id),
            )

    # ---------- 作用域/分层 查询（Mem0 + Letta） ----------

    def by_scope(self, scope: str) -> List[MemoryNode]:
        return [n for n in self.all_nodes() if n.scope == scope]

    def by_tier(self, tier: str) -> List[MemoryNode]:
        return [n for n in self.all_nodes() if n.tier == tier]

    def valid_nodes(self, now: float = 0.0) -> List[MemoryNode]:
        """当前时序有效的记忆（Zep）。"""
        return [n for n in self.all_nodes() if n.is_valid(now)]

    def promote(self, node_id: str, tier: str) -> None:
        """分层晋升（Letta）：recall→core 等。"""
        self.conn.execute("UPDATE nodes SET tier=? WHERE node_id=?", (tier, node_id))
        self._commit_if_autonomous()

    def set_valid_until(self, node_id: str, valid_until: float) -> None:
        with self.transaction():
            self.conn.execute("UPDATE nodes SET valid_until=? WHERE node_id=?", (valid_until, node_id))

    # ---------- 自我编辑（Letta） ----------

    def edit(self, node_id: str, new_content: str) -> MemoryNode:
        """自我编辑记忆：内容冻结下，编辑=标记旧节点被取代 + 写新节点。

        返回新节点。原节点不改写（遵守内容冻结），用 supersede 边关联。
        """
        old = self.get(node_id)
        if old is None:
            raise KeyError(node_id)
        new = MemoryNode(
            content=new_content, embedding=old.embedding, tags=old.tags,
            scene=old.scene, confidence=old.confidence, kind=old.kind,
            scope=old.scope, tier=old.tier,
            valid_from=old.valid_from, valid_until=old.valid_until,
        )
        self.add(new)
        self.supersede(old.node_id, new.node_id, evidence="edit")
        return new

    # ---------- 边 ----------

    def add_edge(self, src: str, dst: str, weight: float, kind: str = "semantic", evidence: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
            (src, dst, weight, kind, evidence),
        )
        self._commit_if_autonomous()

    def neighbors(self, node_id: str) -> List[Tuple[MemoryNode, float]]:
        rows = self.conn.execute(
            "SELECT dst, weight FROM edges WHERE src=? UNION ALL "
            "SELECT src, weight FROM edges WHERE dst=?", (node_id, node_id)
        ).fetchall()
        out = []
        for other_id, w in rows:
            n = self.get(other_id)
            if n is not None:
                out.append((n, w))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def all_edges(self) -> List[Tuple[str, str, float, str, str]]:
        return [tuple(r) for r in self.conn.execute(
            "SELECT src,dst,weight,kind,evidence FROM edges").fetchall()]

    def supersede(self, old_id: str, new_id: str, evidence: str = "") -> None:
        self.add_edge(old_id, new_id, 1.0, kind="supersede", evidence=evidence)

    def superseded_by(self, node_id: str) -> Optional[MemoryNode]:
        row = self.conn.execute(
            "SELECT dst FROM edges WHERE src=? AND kind='supersede' ORDER BY weight DESC LIMIT 1",
            (node_id,),
        ).fetchone()
        return self.get(row[0]) if row else None

    # ---------- 同步水位线 ----------

    def set_meta(self, key: str, value: str) -> None:
        """Persist a small extension state value atomically."""
        self._set_meta(key, value)

    def get_meta(self, key: str) -> Optional[str]:
        """Read a persisted extension state value."""
        return self._get_meta(key)

    def set_watermark(self, device: str, seq: int) -> None:
        self._set_meta(f"sync_watermark_{device}", str(seq))

    def get_watermark(self, device: str) -> int:
        raw = self._get_meta(f"sync_watermark_{device}")
        try:
            return int(raw) if raw else 0
        except (ValueError, TypeError):
            return 0

    # ---------- 检索 ----------

    def search(self, query_vec: List[float], k: int = 5, rel_floor: float = 0.5,
               scope: Optional[str] = None,
               candidate_ids: Optional[List[str]] = None) -> List[Tuple[MemoryNode, float]]:
        """向量检索。candidate_ids 提供候选集时只在这些节点里算余弦（预筛）。

        混合策略：
        - 无候选集：全量余弦（小规模，零召回损失）
        - 有候选集：只在候选里算（大规模，FTS5 粗筛后精排）
        """
        if not query_vec:
            return []
        scored = []
        # 候选集：只算这些节点
        if candidate_ids is not None:
            cand_set = set(candidate_ids)
            for node_id in cand_set:
                vec = self._cached_vec(node_id)
                if vec is None:
                    continue
                s = self._sim(query_vec, vec)
                if s > 0.0:
                    scored.append((node_id, s))
        # 全量：内存缓存优先
        elif len(self._vec_cache) >= self.count_nodes():
            for node_id, (blob, vec) in self._vec_cache.items():
                s = self._sim(query_vec, vec)
                if s > 0.0:
                    scored.append((node_id, s))
        else:
            rows = self.conn.execute("SELECT node_id, embedding FROM nodes").fetchall()
            for node_id, blob in rows:
                if not blob:
                    continue
                vec = self._decode_cached(node_id, blob)
                s = self._sim(query_vec, vec)
                if s > 0.0:
                    scored.append((node_id, s))
        # 只需 top-k：避免数千条候选的 O(n log n) 全排序。
        # heapq.nlargest 保持与原排序相同的分数降序语义。
        scored = heapq.nlargest(k, scored, key=lambda t: t[1])
        if scored and rel_floor > 0.0:
            floor = scored[0][1] * rel_floor
            scored = [t for t in scored if t[1] >= floor]
        hits = []
        for node_id, s in scored[:k]:
            n = self.get(node_id)
            if n is None:
                continue
            if scope and n.scope != scope:
                continue
            if n.tier == "archival":
                # 遗忘-重激活：被重新提起的遗忘记忆恢复活跃（遗忘≠删除，提起来又活跃）
                self.promote(n.node_id, "recall")
                n.tier = "recall"
            hits.append((n, s))
        if hits:
            with self.transaction():
                for n, _ in hits:
                    self.conn.execute(
                        "UPDATE nodes SET last_access=?, access_count=access_count+1 WHERE node_id=?",
                        (time.time(), n.node_id),
                    )
        return hits

    def _sim(self, a: List[float], b: List[float]) -> float:
        """相似度：向量已归一化时用点积（省两次开方），否则余弦。"""
        if self.normalized:
            return dot_normalized(a, b)
        return cosine(a, b)

    def _cached_vec(self, node_id: str) -> Optional[List[float]]:
        """取节点向量（优先内存缓存，其次 DB）。"""
        cached = self._vec_cache.get(node_id)
        if cached is not None:
            return cached[1]
        row = self.conn.execute(
            "SELECT embedding FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        if row and row[0]:
            return self._decode_cached(node_id, row[0])
        return None

    def _decode_cached(self, node_id: str, blob: bytes) -> List[float]:
        cached = self._vec_cache.get(node_id)
        if cached is not None and cached[0] == blob:
            return cached[1]
        vec = _unpack_vec(blob)
        self._vec_cache[node_id] = (blob, vec)
        return vec

    # ---------- 缺口发现 ----------

    def record_gap(self, query: str) -> None:
        q = query.strip()
        if not q:
            return
        gaps = self.gap_queries()
        if any(g.get("q") == q for g in gaps):
            return
        gaps.insert(0, {"q": q[:80], "t": time.time()})
        with self.transaction():
            self._set_meta("gap_queries", json.dumps(gaps[:GAP_LIMIT], ensure_ascii=False))

    def gap_queries(self) -> List[dict]:
        raw = self._get_meta("gap_queries")
        if not raw:
            return []
        try:
            gaps = json.loads(raw)
            return gaps if isinstance(gaps, list) else []
        except (ValueError, TypeError):
            return []

    # ---------- 内部 ----------

    def _row_to_node(self, row: Tuple) -> MemoryNode:
        emb = row[2]
        if isinstance(emb, bytes) and emb:
            emb = _unpack_vec(emb)
        else:
            emb = []
        return MemoryNode(
            node_id=row[0], content=row[1], embedding=emb,
            tags=self._parse_tags(row[3]), scene=row[4], confidence=row[5],
            kind=row[6], scope=row[7], tier=row[8],
            valence=row[9],
            valid_from=row[10], valid_until=row[11],
            retention_strength=row[12], review_count=row[13],
            created_at=row[14], last_access=row[15], access_count=row[16],
        )

    @staticmethod
    def _parse_tags(raw) -> List[str]:
        """解析 tags JSON，旧库 NULL/脏值容错为 []（旧数据兼容）。"""
        if not raw:
            return []
        try:
            t = json.loads(raw)
            return t if isinstance(t, list) else []
        except Exception:
            return []

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
        self._commit_if_autonomous()

    def _get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _commit_if_autonomous(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()

    def stats(self) -> dict:
        return {"path": self.path, "nodes": self.count_nodes()}

    def close(self) -> None:
        try:
            self.conn.commit()
        except Exception:
            pass
        self.conn.close()
