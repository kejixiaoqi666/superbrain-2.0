"""基准：Rust HNSW vs 纯 Python 线性扫描的向量检索性能对比。"""
import random
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from superbrain2 import HnswIndex, _IMPL


def rand_unit(dim, rng):
    v = [rng.uniform(-1, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def main():
    dim, n, k, ef = 64, 10000, 5, 32
    print(f"实现: {_IMPL} | {n} 条向量, dim={dim}, top-{k}")
    rng = random.Random(42)
    vecs = [rand_unit(dim, rng) for _ in range(n)]
    q = vecs[0]

    # 纯 Python 线性扫描基准
    t0 = time.perf_counter()
    for _ in range(200):
        sc = [(i, sum(a*b for a, b in zip(q, v))) for i, v in enumerate(vecs)]
        sc.sort(key=lambda t: t[1], reverse=True)
        sc[:k]
    py_ms = (time.perf_counter() - t0) / 200 * 1000

    # Rust HNSW 基准
    idx = HnswIndex(dim=dim, m=16, ef=ef, seed=1)
    t0 = time.perf_counter()
    for v in vecs:
        idx.add(v)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(2000):
        idx.search(q, k, ef)
    rust_ms = (time.perf_counter() - t0) / 2000 * 1000

    # 正确性抽查：HNSW top1 是否 = 线性扫描 top1
    ref = sorted(range(n), key=lambda i: sum(a*b for a, b in zip(q, vecs[i])), reverse=True)[0]
    r = idx.search(q, k, ef)
    hnsw_top1 = r[0][0]
    print(f"  线性扫描: {py_ms:.2f} ms/查询 | HNSW: {rust_ms:.4f} ms/查询")
    print(f"  HNSW构建 {n} 条: {build_s:.2f}s | 加速比 ≈ {py_ms/rust_ms:.0f}x")
    print(f"  正确性: HNSW top1={hnsw_top1} vs 真最近={ref} {'✅' if hnsw_top1==ref else '⚠️近似'}")


if __name__ == "__main__":
    main()