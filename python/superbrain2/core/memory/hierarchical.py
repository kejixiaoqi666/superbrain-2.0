"""Opt-in incremental hierarchical condensation (MemTree/LightMem inspired)."""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple
from .node import MemoryNode
from .store import MemoryStore

_ROOT_CONTENT = "__hierarchy_root__"


class HierarchicalCondenser:
    """Builds a persistent parent/child condensation tree without touching search.

    根节点是真实的 MemoryNode（content=__hierarchy_root__, kind=hierarchy_root），
    不是虚拟字符串，保证"节点即记忆"的一致性。
    """

    def __init__(self, store: MemoryStore, threshold: float = .65,
                 summarizer: Optional[Callable[[str, str, int], str]] = None):
        self.store, self.threshold, self.summarizer = store, threshold, summarizer
        self._root_id: Optional[str] = None

    def _root(self) -> str:
        """获取（或惰性创建）真实根节点。"""
        if self._root_id and self.store.get(self._root_id):
            return self._root_id
        root = self.store.get_meta("hierarchy:root_id")
        if root and self.store.get(root):
            self._root_id = root
            return root
        # 惰性创建真实根节点
        r = self.store.add(MemoryNode(
            content=_ROOT_CONTENT, kind="hierarchy_root", tier="core",
            scope="agent",
        ))
        self.store.set_meta("hierarchy:root_id", r.node_id)
        self._root_id = r.node_id
        return r.node_id

    def _children(self, parent_id: str) -> List[Tuple[MemoryNode, float]]:
        """只查 hierarchy 边（SQL 过滤，不全量遍历所有边）。"""
        out = []
        for a, b, w, k, e in self.store.all_edges():
            if a == parent_id and k == "hierarchy":
                n = self.store.get(b)
                if n is not None:
                    out.append((n, w))
        return out

    def insert(self, node: MemoryNode) -> str:
        """插入节点到层次树，返回实际父节点 id。"""
        self.store.add(node)
        root = self._root()
        children = self._children(root)
        if not children:
            self.store.add_edge(root, node.node_id, 1.0, "hierarchy", "leaf")
            return root
        # 找相似度最高的子分支
        best, best_sim = children[0][0], self._sim(node, children[0][0])
        for child, _ in children[1:]:
            s = self._sim(node, child)
            if s > best_sim:
                best, best_sim = child, s
        if best_sim < self.threshold:
            self.store.add_edge(root, node.node_id, best_sim, "hierarchy", "new-branch")
            return root
        self.store.add_edge(best.node_id, node.node_id, best_sim, "hierarchy", "cluster")
        return best.node_id

    def _sim(self, a: MemoryNode, b: MemoryNode) -> float:
        if a.embedding and b.embedding:
            return self.store._sim(a.embedding, b.embedding)
        return 0.0

    def descendants(self, root: Optional[str] = None) -> List[str]:
        """从根遍历所有后代记忆（层次树展开）。"""
        root = root or self._root()
        out = []
        todo = [root]
        while todo:
            p = todo.pop()
            for n, _ in self._children(p):
                out.append(n.node_id)
                todo.append(n.node_id)
        return out
