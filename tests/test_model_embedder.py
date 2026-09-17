"""真实 embedding 模型（可插拔）质量/架构回归测试。

覆盖：
1. build_embedder 注册表默认仍为 hashing（零依赖、旧库向量兼容）
2. bge 懒加载：缺模型/缺依赖时给清晰安装渠道提示，绝不崩整个内核
3. 嵌入器指纹：hashing 与 bge 身份可区分（语义漂移守卫）
4. 存储语义漂移守卫 + reembed_all 迁移（换嵌入器修复向量不可比）
5. Agent 接线：默认仍然 hashing，不破坏现有检索
6. 真实模型 E2E（可选，机器上有模型文件才跑）
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from superbrain2.core import AgentConfig, SuperBrainAgent
from superbrain2.core.memory.embeddings import (
    EmbeddingInstallRequired, build_embedder,
    embedder_identity, registered_embedders,
)
from superbrain2.core.memory.store import MemoryStore


class RegistryDefaultTest(unittest.TestCase):
    """注册表默认必须仍为 hashing，保证零依赖 + 旧库向量兼容。"""

    def test_default_is_hashing(self):
        emb = build_embedder()
        self.assertEqual(type(emb).__name__, "HashingEmbedder")
        self.assertEqual(len(emb.embed("网络卡顿")), 256)  # 旧版本 dim

    def test_explicit_hashing_backcompat(self):
        emb = build_embedder("hashing", enhance=True)
        self.assertEqual(emb.revision, "v2-enhanced")

    def test_unknown_embedder_raises(self):
        with self.assertRaises(ValueError):
            build_embedder("nope")

    def test_registered_names(self):
        self.assertIn("hashing", registered_embedders())
        self.assertIn("bge", registered_embedders())


class BgeLazyInstallTest(unittest.TestCase):
    """bge 懒加载：缺模型/缺依赖时报清晰安装渠道，不影响默认路径。"""

    def test_bge_without_model_raises_actionable(self):
        emb = build_embedder("bge", model_dir="/nonexistent/no-model-here")
        # 构造不报错（懒加载），首次 embed 才抛"需安装"
        with self.assertRaises(EmbeddingInstallRequired) as ctx:
            emb.embed("测试")
        msg = str(ctx.exception)
        # 错误信息必须指向安装渠道（可行动的）
        self.assertIn("install-model", msg)
        self.assertIn("onnxruntime", msg)
        self.assertIn("缺少模型文件", msg)

    def test_help_text_points_at_install_channel(self):
        from superbrain2.core.memory import embeddings as em
        self.assertIn("install-model", em.HELP_TEXT)
        self.assertIn("pip install onnxruntime tokenizers", em.HELP_TEXT)


class EmbedderIdentityGuardTest(unittest.TestCase):
    """嵌入器指纹可区分：换嵌入器必须能嗅出语义漂移。"""

    def test_hashing_vs_bge_fp_differ(self):
        h = build_embedder()
        b = build_embedder("bge")
        self.assertNotEqual(embedder_identity(h)["fp"], embedder_identity(b)["fp"])

    def test_bge_identity_fields(self):
        b = build_embedder("bge", dim=768)
        info = embedder_identity(b)
        self.assertEqual(info["dim"], 768)
        self.assertEqual(info["type"], "OnnxEmbedder")
        self.assertTrue(b.normalized)


class StoreSemanticDriftTest(unittest.TestCase):
    """存储语义漂移守卫 + 重嵌入迁移：绝不静默混算不可比向量。"""

    def _store(self):
        return MemoryStore(path=tempfile.mktemp(suffix=".db"))

    def test_empty_store_is_ok(self):
        s = self._store()
        self.assertEqual(s.embedder_compat("any-fp"), "ok")

    def test_matching_fp_ok(self):
        s = self._store()
        from superbrain2.core.memory.node import MemoryNode
        s.add(MemoryNode(content="有一次记忆", embedding=[1.0, 0.0]))
        s.mark_embedder("fp-A")
        self.assertEqual(s.embedder_compat("fp-A"), "ok")

    def test_changed_fp_needs_reembed(self):
        s = self._store()
        from superbrain2.core.memory.node import MemoryNode
        s.add(MemoryNode(content="有一次记忆", embedding=[1.0, 0.0]))
        s.mark_embedder("fp-A")
        # 换嵌入器（新指纹）→ 必须重嵌入，不能混算
        self.assertEqual(s.embedder_compat("fp-B"), "needs_reembed")

    def test_legacy_unstamped_needs_reembed(self):
        # 旧库从未打标：保守要求重嵌入，防静默混算
        s = self._store()
        from superbrain2.core.memory.node import MemoryNode
        s.add(MemoryNode(content="旧记忆", embedding=[1.0, 0.0]))
        self.assertEqual(s.embedder_compat("fp-A"), "needs_reembed")

    def test_reembed_all_migrates_dim(self):
        s = self._store()
        from superbrain2.core.memory.node import MemoryNode
        s.add(MemoryNode(content="记忆一", embedding=[1.0, 0.0]))
        s.add(MemoryNode(content="记忆二", embedding=[0.0, 1.0]))
        s.mark_embedder("fp-2d")
        # 新嵌入器是 8 维 → 重嵌入后所有向量变 8 维
        n = s.reembed_all(lambda text: [0.1 * (i + 1) for i in range(8)])
        self.assertEqual(n, 2)
        for node in s.all_nodes():
            self.assertEqual(len(node.embedding), 8)

    def test_agent_ensure_ready_on_empty(self):
        # Agent 默认接线 hashing；空库 ensure_embedder_ready 直接 ok
        class DummyLLM:
            def chat(self, *a, **k): return "hi"
        cfg = AgentConfig()
        agent = SuperBrainAgent(DummyLLM(), config=cfg,
                                store=MemoryStore(path=tempfile.mktemp(suffix=".db")))
        self.assertEqual(type(agent._embedder).__name__, "HashingEmbedder")
        self.assertEqual(agent.embedder_info()["type"], "HashingEmbedder")
        r = agent.ensure_embedder_ready()
        self.assertEqual(r["compat"], "ok")


REAL_MODEL_DIR = os.environ.get(
    "SUPERBRAIN_EMBEDDER_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), ".superbrain", "models", "bge-base-zh-v1.5"))


@unittest.skipUnless(
    all(os.path.exists(os.path.join(REAL_MODEL_DIR, f))
        for f in ("model.onnx", "tokenizer.json")),
    "未安装真实模型文件，跳过端到端语义验证")
class RealModelE2ETest(unittest.TestCase):
    """真实 ONNX BGE 语义端到端：突破哈希的同义词天花板。"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import importlib.util
        for mod in ("onnxruntime", "tokenizers"):
            if importlib.util.find_spec(mod) is None:
                raise unittest.SkipTest(f"缺 {mod}")

    def test_semantic_synonym_high(self):
        emb = build_embedder("bge", model_dir=REAL_MODEL_DIR, dim=768)
        import superbrain2.core.memory.embeddings as em
        # 零字面重叠的同义词：哈希会失效，真实模型必须高相似
        s = em.cosine(emb.embed("节点线路卡顿不稳定"), emb.embed("服务延迟高很卡"))
        self.assertGreater(s, 0.6)
        s2 = em.cosine(emb.embed("开通启用节点路由"), emb.embed("启用新线路入口通道"))
        self.assertGreater(s2, 0.6)

    def test_related_beats_unrelated(self):
        emb = build_embedder("bge", model_dir=REAL_MODEL_DIR, dim=768)
        import superbrain2.core.memory.embeddings as em
        related = em.cosine(emb.embed("节点线路卡顿不稳定"), emb.embed("服务延迟高很卡"))
        unrelated = em.cosine(emb.embed("节点线路卡顿不稳定"), emb.embed("今天晚饭吃番茄炒蛋"))
        self.assertGreater(related, unrelated)

    def test_real_search_ranks_related_first(self):
        emb = build_embedder("bge", model_dir=REAL_MODEL_DIR, dim=768)
        s = MemoryStore(path=tempfile.mktemp(suffix=".db"))
        from superbrain2.core.memory.node import MemoryNode
        s.add(MemoryNode(content="服务节点延迟高很卡", embedding=emb.embed("服务节点延迟高很卡")))
        s.add(MemoryNode(content="今天晚饭吃了番茄炒蛋", embedding=emb.embed("今天晚饭吃了番茄炒蛋")))
        hits = s.search(emb.embed("节点线路卡顿不稳定怎么办"), k=2, rel_floor=0.3)
        self.assertTrue(hits)
        self.assertIn("延迟", hits[0][0].content)


if __name__ == "__main__":
    unittest.main()