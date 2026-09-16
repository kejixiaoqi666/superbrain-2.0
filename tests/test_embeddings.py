"""HashingEmbedder 质量回归测试：默认兼容 + 增强提升 + 不误伤。

防止未来改动把嵌入质量改退化、或破坏旧库向量一致性。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from superbrain2.core.memory.embeddings import HashingEmbedder, cosine


class DefaultCompatibilityTest(unittest.TestCase):
    """默认模式必须与旧版（dim=256,ngram=2）完全一致，保旧库向量。"""

    def test_default_matches_legacy(self):
        # 旧版构造（无 enhance）等价于默认
        legacy = HashingEmbedder(dim=256, ngram=2).embed("网络连接不稳定")
        default = HashingEmbedder().embed("网络连接不稳定")
        self.assertEqual(legacy, default)

    def test_default_is_low_dim_legacy(self):
        e = HashingEmbedder()
        self.assertFalse(e.enhance)
        self.assertEqual(e.revision, "v1")


class EnhancedQualityTest(unittest.TestCase):
    """增强版必须突破同义词天花板，且不误伤无关对。"""

    def setUp(self):
        self.enh = HashingEmbedder(enhance=True)

    def test_synonym_crosses_threshold(self):
        # 零字面重叠的同义词，旧版 0.236，增强版需 > 0.3
        s = cosine(self.enh.embed("网络连接不稳定"), self.enh.embed("网络卡顿"))
        self.assertGreater(s, 0.3)

    def test_expand_synonym(self):
        s = cosine(self.enh.embed("开通服务节点"), self.enh.embed("启用新的服务器"))
        self.assertGreater(s, 0.3)

    def test_unrelated_stays_low(self):
        s = cosine(self.enh.embed("无关对照"), self.enh.embed("天气怎么样"))
        self.assertLess(s, 0.3)

    def test_word_order_still_works(self):
        s = cosine(self.enh.embed("申请退款"), self.enh.embed("退款申请"))
        self.assertGreater(s, 0.5)

    def test_english_case_insensitive(self):
        s = cosine(self.enh.embed("login失败"), self.enh.embed("LOGIN连不上"))
        self.assertGreater(s, 0.5)


class EmbedderIdentityTest(unittest.TestCase):
    """增强版与默认版必须可区分指纹，防跨库比错向量。"""

    def test_revision_differs(self):
        e1 = HashingEmbedder()
        e2 = HashingEmbedder(enhance=True)
        self.assertNotEqual(e1.revision, e2.revision)
        self.assertEqual(e2.revision, "v2-enhanced")


if __name__ == "__main__":
    unittest.main()