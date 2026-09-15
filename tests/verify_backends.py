"""多后端验证：faiss / usearch / rust / python 端到端 + 召回对比。"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from superbrain2 import VectorIndex

DIM, N, K = 64, 2000, 10

def unit(dim, rng):
    v = [rng.uniform(-1, 1) for _ in range(dim)]
    n = sum(x*x for x in v) ** 0.5 or 1.0
    return [x/n for x in v]

def main():
    rng = random.Random(7)
    vecs = [unit(DIM, rng) for _ in range(N)]
    q = vecs[100]

    # 精确真值（线性）
    true = sorted(range(N), key=lambda i: sum(a*b for a,b in zip(q, vecs[i])), reverse=True)[:K]

    for backend in ["faiss", "usearch", "rust", "python"]:
        try:
            idx = VectorIndex(DIM, m=16, ef=32, backend=backend)
            for v in vecs:
                idx.add(v)
            r = dict(idx.search(q, K, 32))
            top = [i for i, _ in idx.search(q, K, 32)]
            hits = len(set(top) & set(true))
            ok_self = (top[0] == 100)
            print(f"  {backend:<8} len={len(idx):<5} top含自身={ok_self} recall@{K}={hits/K:.2f} top1={top[0]}")
        except Exception as e:
            print(f"  {backend:<8} 失败: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()