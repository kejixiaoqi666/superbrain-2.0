"""补测：各引擎的内存占用 + CPU（构建/查询）。N=50000, dim=128"""
import sys, os, time, resource
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import psutil, numpy as np

DIM, K, EF = 128, 10, 64
NQ = 500  # 减少查询次数，避免超时

def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def gen(n, dim, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

SCALES = [5000, 50000]
for N in SCALES:
    print(f"\n{'#'*62}\n## 规模 N={N}")
    X = gen(N, DIM)
    qidx = np.random.default_rng(9).choice(N, NQ, replace=False)
    Q = X[qidx]
    true = np.argsort(-(X @ Q.T), axis=0)[:K].T
    print(f"numpy 数据基线内存: {rss_mb():.1f} MB (含 X+Q)")
    base_rss = rss_mb()
    print(f"{'引擎':<16}{'索引内存MB':>9}{'查询ms':>8}{'recall':>8}{'构建CPU s':>9}{'查询CPU s':>9}{'CPU占用%':>8}")
    rows = []

    # numpy 线性
    t_cpu0=time.process_time(); t0=time.perf_counter()
    for _ in range(NQ): X @ Q.T
    lw=time.perf_counter()-t0; lc=time.process_time()-t_cpu0
    rows.append(("numpy线性(≈SQLite)", 0, lw*1000/NQ, 1.0, lc, lc, lc/lw*100))

    # 自研 Rust HNSW
    from superbrain2 import HnswIndex
    idx = HnswIndex(dim=DIM, m=16, ef=EF, seed=1)
    t_c=time.process_time(); t0=time.perf_counter()
    for i in range(N): idx.add(X[i].tolist())
    bc=time.process_time()-t_c; bw=time.perf_counter()-t0
    mem = rss_mb()-base_rss
    t_c=time.process_time(); t0=time.perf_counter()
    rr=[sorted(j for j,_ in idx.search(Q[i].tolist(),K,EF)) for i in range(NQ)]
    qc=time.process_time()-t_c; qw=time.perf_counter()-t0
    rec=np.mean([len(set(true[i])&set(rr[i]))/K for i in range(NQ)])
    rows.append(("自研Rust HNSW", mem, qw*1000/NQ, rec, bc, qc, qc/qw*100))

    # FAISS HNSW
    import faiss
    h=faiss.IndexHNSWFlat(DIM,16); h.hnsw.efConstruction=EF
    t_c=time.process_time(); t0=time.perf_counter(); h.add(X)
    fb=time.process_time()-t_c; fw=time.perf_counter()-t0
    mem=rss_mb()-base_rss
    h.hnsw.efSearch=EF
    t_c=time.process_time(); t0=time.perf_counter()
    for _ in range(NQ): h.search(Q,K)
    fcq=time.process_time()-t_c; fqw=time.perf_counter()-t0
    _,I=h.search(Q,K)
    rec=np.mean([len(set(true[i])&set(I[i]))/K for i in range(NQ)])
    rows.append(("FAISS HNSW", mem, fqw*1000/NQ, rec, fb, fcq, fcq/fqw*100))

    for name, mem, ms, rec, bc_, qc_, cpu in rows:
        print(f"{name:<16}{mem:>9.1f}{ms:>8.3f}{rec:>8.3f}{bc_:>9.2f}{qc_:>9.3f}{cpu:>8.0f}")
    base = rows[0][2]
    print(f"相对线性加速: " + " | ".join(f"{r[0]}={base/r[2]:.0f}x" for r in rows[1:]))
print("\n机器: %d 逻辑核, 内存 %.1fGB" % (psutil.cpu_count(), psutil.virtual_memory().total/1e9))