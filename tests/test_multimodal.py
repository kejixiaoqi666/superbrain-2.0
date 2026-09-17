import base64
import tempfile
import unittest
from pathlib import Path

from superbrain2.core.llm import (_content_text, image_content_part,
                                  LLMProvider, LLMResponse, multimodal_message,
                                  to_data_url)
from superbrain2.core.memory.store import MemoryStore


class LlmHelpersTest(unittest.TestCase):
    def test_to_data_url_prefix_and_roundtrip(self):
        raw = b"\x89PNGfakebytes"
        url = to_data_url(raw, "image/png")
        self.assertTrue(url.startswith("data:image/png;base64,"))
        decoded = base64.b64decode(url.split(",", 1)[1])
        self.assertEqual(decoded, raw)

    def test_image_content_part(self):
        part = image_content_part("data:image/png;base64,xxx")
        self.assertEqual(part["type"], "image_url")
        self.assertEqual(part["image_url"]["url"], "data:image/png;base64,xxx")

    def test_multimodal_message_structure(self):
        msg = multimodal_message("看看这张图", ["u1", "u2"])
        self.assertEqual(msg["role"], "user")
        self.assertIsInstance(msg["content"], list)
        self.assertEqual(msg["content"][0],
                         {"type": "text", "text": "看看这张图"})
        self.assertEqual(msg["content"][1]["type"], "image_url")
        self.assertEqual(msg["content"][1]["image_url"]["url"], "u1")
        self.assertEqual(msg["content"][2]["image_url"]["url"], "u2")

    def test_multimodal_message_no_text(self):
        msg = multimodal_message("", ["u1"])
        self.assertEqual(len(msg["content"]), 1)
        self.assertEqual(msg["content"][0]["type"], "image_url")

    def test_content_text_extracts_text_from_list(self):
        content = [{"type": "text", "text": "甲"},
                   {"type": "image_url", "image_url": {"url": "u"}},
                   {"type": "text", "text": "乙"}]
        self.assertEqual(_content_text(content), "甲 乙")
        self.assertEqual(_content_text("纯文本"), "纯文本")


class _FakeLLM(LLMProvider):
    def __init__(self):
        super().__init__("http://fake/v1", "k", "m")
        self.calls = []

    def chat(self, messages, tools=None, max_tokens=2048):
        self.calls.append(list(messages))
        return LLMResponse(content="识别到：这是一张节点状态图", finish_reason="stop")


class AgentMultimodalTest(unittest.TestCase):
    def _make_agent(self):
        from superbrain2.core.agent import AgentConfig, SuperBrainAgent
        tmp = tempfile.mkdtemp()
        store = MemoryStore(path=Path(tmp) / "m.db")
        llm = _FakeLLM()
        agent = SuperBrainAgent(llm, store=store, config=AgentConfig())
        return agent, llm, store

    def test_agent_chat_passes_image_to_llm(self):
        agent, llm, store = self._make_agent()
        url = to_data_url(b"\x89PNGdata", "image/png")
        try:
            reply = agent.chat("看看这张图", images=[url])
            self.assertIn("识别到", reply)
            # 用户消息应为多模态 list，含 image_url
            last = llm.calls[-1]
            user_msgs = [m for m in last if m.get("role") == "user"]
            user_content = user_msgs[-1]["content"]
            self.assertIsInstance(user_content, list)
            self.assertTrue(any(p.get("type") == "image_url"
                                for p in user_content if isinstance(p, dict)))
            self.assertIn("data:image/png;base64,", str(user_content))
        finally:
            store.close()

    def test_agent_chat_without_images_stays_text(self):
        agent, llm, store = self._make_agent()
        try:
            agent.chat("你好")
            # 无图时会话里存的是纯文本 str（不是多模态 list）
            user_msgs = [m for m in agent._conversation
                         if m.get("role") == "user"]
            self.assertTrue(user_msgs)
            self.assertIsInstance(user_msgs[-1]["content"], str)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
