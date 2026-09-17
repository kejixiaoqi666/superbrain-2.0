import math
import random
import tempfile
import unittest

from superbrain2.core.memory.node import MemoryNode
from superbrain2.core.memory.store import MemoryStore


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _exact_topk(store, query, k):
    """精确暴力 top-k 参考：逐对余弦 + 降序取 k。返回 [(node_id, score)]。
    （归一化时余弦即点积，故对 normalized/non-normalized 都成立。）"""
    out = []
    for node_id, (blob, vec) in store._vec_cache.items():
        s = _cos(query, vec)
        if s > 0.0:
            out.append((node_id, s))
    out.sort(key=lambda t: -t[1])
    return out[:k]


def _make_store(n, dim=32, seed=1, normalized=True):
    rnd = random.Random(seed)
    store = MemoryStore(path=tempfile.mktemp(suffix=".db"))
    store.normalized = normalized
    for i in range(n):
        raw = [rnd.gauss(0, 1) for _ in range(dim)]
        store.add(MemoryNode(content=f"节点{i}",
                             embedding=_norm(raw) if normalized else raw))
    return store


class MatrixSearchCorrectnessTest(unittest.TestCase):
    # search() 返回 [(MemoryNode, score)]；参考 _exact_topk 用 (node_id, score)。
    def _got(self, hits):
        return [h[0].node_id for h in hits]

    def test_matrix_matches_exact_topk_ordering(self):
        store = _make_store(300, dim=64, seed=7)
        q = _norm([random.Random(99).gauss(0, 1) for _ in range(64)])
        store.search(q, k=5)  # 预热,触发矩阵构建
        for k in (1, 3, 5, 10):
            got = store.search(q, k=k)
            exp = _exact_topk(store, q, k)
            self.assertEqual(self._got(got), [e[0] for e in exp],
                             f"矩阵路径 top-{k} 顺序与精确暴力不一致")
            for (_, gs), (_, es) in zip(got, exp):
                self.assertAlmostEqual(gs, es, places=5)
        store.close()

    def test_non_normalized_matrix_matches(self):
        store = _make_store(200, dim=48, seed=3, normalized=False)
        q = [random.Random(5).gauss(0, 1) for _ in range(48)]
        store.search(q, k=5)  # 预热
        got = store.search(q, k=5)
        exp = _exact_topk(store, q, 5)
        self.assertEqual(self._got(got), [e[0] for e in exp])
        store.close()

    def test_matrix_rebuilds_on_mutation(self):
        store = _make_store(50, dim=32, seed=2)
        q = _norm([random.Random(8).gauss(0, 1) for _ in range(32)])
        before_ids = self._got(store.search(q, k=5))
        before = [store.get(i).content for i in before_ids]
        # 新增一个与 query 高度相似的节点 → 位置应变化
        near = [x * 0.999 + y * 0.001 for x, y in zip(q, _norm(q))]
        store.add(MemoryNode(content="新增热节点", embedding=near))
        after_ids = self._got(store.search(q, k=5))
        after = [store.get(i).content for i in after_ids]
        self.assertIn("新增热节点", after)
        self.assertNotEqual(before, after, "新增节点后 top-k 应更新(矩阵需重建)")
        store.close()

    def test_ranking_returns_top_scoring_node(self):
        store = _make_store(100, dim=32, seed=11)
        # 造一个和查询几乎一致的节点
        q = _norm([random.Random(4).gauss(0, 1) for _ in range(32)])
        store.add(MemoryNode(content="目标命中", embedding=q))
        store.search(q, k=5)  # 预热
        hits = store.search(q, k=5)
        self.assertEqual(hits[0][0].content, "目标命中")
        store.close()


if __name__ == "__main__":
    unittest.main()