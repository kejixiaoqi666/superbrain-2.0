"""完整实测：线性 / 自研Rust HNSW / FAISS HNSW / FAISS IVF-PQ
测贴合"对话一个月"的量级 N=5000 和 50000，dim=128，给延迟/内存/召回选型。"""
import sys, os, time, random, statistics, resource
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

DIM, K, EF = 128, 10, 64
NQ = 200
SCALES = [5000, 50000]
print("参数: dim=128 k=10 ef=64 query=200  单机 %d 核\n" % os.cpu_count())


def gen(dim, n, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # 归一化 → 点积=余弦
    return X


def brute_topk(q, X, k):
    scores = X @ q
    return np.argsort(-scores)[:k].tolist()


def main():
    import numpy as np
    import faiss
    from superbrain2 import HnswIndex

    for N in SCALES:
        print("#" * 62)
        print(f"## 规模 N={N} (对话一个月量级)  | 向量原始 {N*DIM*4/1e6:.1f} MB")
        X = gen(DIM, N)
        qidx = random.Random(9).sample(range(N), NQ)
        Q = X[qidx]

        # 1 精确真值 (numpy)
        t0 = time.perf_counter()
        true = np.argsort(-(X @ Q.T), axis=0)[:K].T
        lin_ms = (time.perf_counter() - t0) / NQ * 1000

        rows = [("numpy线性(≈SQLite)", lin_ms, 0, 1.0, None)]

        # 2 自研 Rust HNSW
        idx = HnswIndex(dim=DIM, m=16, ef=EF, seed=1)
        t0 = time.perf_counter()
        for i in range(N):
            idx.add(X[i].tolist())
        rb = time.perf_counter() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        t0 = time.perf_counter()
        rres = [sorted(i for i, _ in idx.search(Q[i].tolist(), K, EF)) for i in range(NQ)]
        rms = (time.perf_counter() - t0) / NQ * 1000
        rrec = np.mean([len(set(true[i]) & set(rres[i])) / K for i in range(NQ)])
        rows.append(("自研Rust HNSW", rms, rb, rrec, rss))

        # 3 FAISS HNSW
        hnsw = faiss.IndexHNSWFlat(DIM, 16); hnsw.hnsw.efConstruction = EF
        t0 = time.perf_counter(); hnsw.add(X); hb = time.perf_counter() - t0
        hnsw.hnsw.efSearch = EF
        t0 = time.perf_counter(); _, I = hnsw.search(Q, K); hms = (time.perf_counter() - t0) / NQ * 1000
        hrec = np.mean([len(set(true[i]) & set(I[i])) / K for i in range(NQ)])
        rows.append(("FAISS HNSW", hms, hb, hrec, None))

        # 4 FAISS IVF-PQ
        pq = faiss.index_factory(DIM, "IVF4096,PQ32", faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter(); pq.train(X); pq.add(X); pb = time.perf_counter() - t0
        pq.nprobe = 16
        t0 = time.perf_counter(); _, Ip = pq.search(Q, K); pms = (time.perf_counter() - t0) / NQ * 1000
        prec = np.mean([len(set(true[i]) & set(Ip[i])) / K for i in range(NQ)])
        rows.append(("FAISS IVF-PQ", pms, pb, prec, None))

        print(f"{'引擎':<18}{'查询ms':>9}{'构建s':>9}{'recall@10':>10}{'RSS MB':>8}")
        for name, ms, b, rec, rss_ in rows:
            print(f"{name:<18}{ms:>9.3f}{b:>9.2f}{rec:>10.3f}{str(rss_ or '-'):>8}")
        base = rows[0][1]
        print(f"\n相对线性加速:")
        for name, ms, b, rec, rss_ in rows[1:]:
            print(f"  {name}: {base/ms:.0f}x | 吞吐提升 {base/ms*100-100:.0f}% | recall {rec:.3f}")
        print()


if __name__ == "__main__":
    main()