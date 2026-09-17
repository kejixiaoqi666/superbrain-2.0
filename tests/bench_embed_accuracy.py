"""嵌入器 A/B 准确率+延迟评测（基准，缺模型则自动跳过）。

对比 哈希默认 / 哈希增强(现优化版) / bge fp32，在同一检索管线上：
  纯向量层：隔离嵌入器语义能力（recall@k + MRR）
  完整混合层：四路 RRF 融合（真实系统端到端）
延迟：每查询（嵌入+检索）稳态耗时。

用法：
  python tests/bench_embed_accuracy.py        # 需已 install-model（bge）
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from superbrain2.core.memory.embeddings import build_embedder
from superbrain2.core.memory.store import MemoryStore
from superbrain2.core.memory.node import MemoryNode
from superbrain2.core.memory import retrieval

MODEL_DIR = os.path.expanduser("~/.superbrain/models/bge-base-zh-v1.5")

# ---------- 语料（16条领域记忆，含干扰项） ----------
CORPUS = [
    "服务节点延迟高不稳定经常掉线",
    "开通启用新的线路转发入口节点",
    "查询用户账号的订购套餐状态",
    "重启服务器让故障服务恢复",
    "退款退单流程三个工作日到账",
    "拨号连接超时连不上目标网络",
    "节点证书失效报错服务异常",
    "快充一号机器IP地址已更换",
    "奥地利机器的根分区默认只读",
    "今日午饭吃了番茄炒蛋",
    "机场订阅服务需要定期续费",
    "订阅网关热加载路由规则配置",
    "快充二号机器监控面板端口8787",
    "存储空间不足需清理过期日志",
    "新装客户端要覆盖安装探针",
    "日抛机IP漂移需要自动同步",
]
# ---------- 标注查询：key=query, value=目标语料索引 ----------
LITERAL = {   # 目标词字面出现在语料里 → 哈希也能部分捕获
    "节点很卡延迟高": 0, "新线路怎么开通": 1, "查一下订购套餐状态": 2,
    "重启机器恢复": 3, "退款退单流程": 4, "连不上网络了": 5, "节点报错异常": 6,
}
PARAPHRASE = {  # 零字面重叠改写 → 只语义模型（bge）能稳接
    "运行老是掉线卡得不行": 0, "再加一条新通道转发": 1,
    "我的套餐快到期待续上": 10, "服务异常没法用了": 6,
    "断连了网不好使": 5, "看看我订购的东西状态": 2,
}
EVAL = {"字面重叠": LITERAL, "零字面改写": PARAPHRASE}


def _skip() -> bool:
    return not all(os.path.exists(os.path.join(MODEL_DIR, f))
                   for f in ("model.onnx", "tokenizer.json"))


def evaluate(emb, full_system):
    s = MemoryStore(path=os.path.join(
        "/tmp", f"bench_{type(emb).__name__}_{id(emb)}.db"))
    for c in CORPUS:
        s.add(MemoryNode(content=c, embedding=emb.embed(c)))
    stats = {}
    for gname, pairs in EVAL.items():
        r1 = r5 = 0
        mrr = 0.0
        for q, target in pairs.items():
            qv = emb.embed(q)
            if full_system:
                res = [r[0].content for r in retrieval.search(s, qv, q, k=5)[:5]]
            else:
                res = [r[0].content for r in s.search(qv, k=5, rel_floor=0.0)]
            idx = next((i for i, c in enumerate(res) if c == CORPUS[target]), None)
            if idx is not None:
                r1 += idx < 1
                r5 += idx < 5
                mrr += 1.0 / (idx + 1)
        n = len(pairs)
        stats[gname] = (r1 / n, r5 / n, mrr / n)
    # 延迟
    qs = list(LITERAL) + list(PARAPHRASE)
    [emb.embed(q) for q in qs[:2]]           # warm
    t0 = time.time()
    for q in qs:
        qv = emb.embed(q)
        if full_system:
            retrieval.search(s, qv, q, k=5)
        else:
            s.search(qv, k=5, rel_floor=0.0)
    lat = (time.time() - t0) / len(qs)
    return stats, lat


def main():
    if _skip():
        print("未安装真实模型，跳过（先 `python -m superbrain2 install-model`）")
        return 0
    os.environ.setdefault("SUPERBRAIN_EMBEDDER_MODEL_DIR", MODEL_DIR)
    configs = [
        ("哈希默认", build_embedder("hashing")),
        ("哈希增强(现优化版)", build_embedder("hashing", enhance=True)),
        ("bge fp32(新)", build_embedder("bge")),
    ]
    for full in (False, True):
        mode = "完整混合(四路RRF)" if full else "纯向量"
        print(f"\n========== {mode} ==========")
        for name, emb in configs:
            st, lat = evaluate(emb, full)
            print(f"[{name}] 每查询延迟 {lat*1000:.1f}ms")
            for g, (r1, r5, mrr) in st.items():
                print(f"  {g}: R@1={r1:.0%} R@5={r5:.0%} MRR={mrr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())