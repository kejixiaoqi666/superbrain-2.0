"""superbrain2 原生内核测试：SIMD 点积 + HNSW 向量索引（支持 rust/python 双实现）。"""
import os
import sys
import random
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from superbrain2 import HnswIndex, dot


class TestDot(unittest.TestCase):
    def test_dot_basic(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(dot(a, b), 32.0, places=5)

    def test_dot_orthogonal(self):
        self.assertAlmostEqual(dot([1, 0], [0, 1]), 0.0, places=5)

    def test_dot_negative(self):
        self.assertAlmostEqual(dot([1, 1], [-1, -1]), -2.0, places=5)


class TestHnsw(unittest.TestCase):
    def _rand_unit(self, dim, rng):
        v = [rng.uniform(-1, 1) for _ in range(dim)]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def test_add_len(self):
        idx = HnswIndex(8, seed=1)
        for _ in range(30):
            idx.add(self._rand_unit(8, random.Random(1)))
        self.assertEqual(len(idx), 30)

    def test_search_finds_self(self):
        idx = HnswIndex(8, m=8, ef=16, seed=1)
        rng = random.Random(7)
        vecs = [self._rand_unit(8, rng) for _ in range(80)]
        for v in vecs:
            idx.add(v)
        # 查询一个已插入向量 → top1 应是它自身
        r = idx.search(vecs[40], k=1)
        self.assertEqual(r[0][0], 40)

    def test_search_returns_sorted_by_score(self):
        idx = HnswIndex(8, seed=2)
        rng = random.Random(3)
        vecs = [self._rand_unit(8, rng) for _ in range(50)]
        for v in vecs:
            idx.add(v)
        r = idx.search(vecs[10], k=5)
        scores = [s for _, s in r]
        self.assertEqual(scores, sorted(scores, reverse=True))  # 点积降序


if __name__ == "__main__":
    unittest.main()
