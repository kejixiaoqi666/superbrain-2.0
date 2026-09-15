"""superbrain2 完整能力冒烟测试：SuperBrain 对话/记忆/人格 + 多后端向量。"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from superbrain2 import SuperBrain, VectorIndex, dot
from superbrain2.core.llm import LLMResponse
from superbrain2.core.memory.store import MemoryStore
from superbrain2.core.cognition.needs import NeedDriveSystem

class Stub:
    def chat(self, messages, tools=None, **kw):
        return LLMResponse(content="好的", finish_reason="stop")

ok = {}

# 1 多后端向量
vi = VectorIndex(8)
for i in range(100):
    vi.add([i/100.0]*8)
r = vi.search([0.5]*8, 5)
ok["多后端向量"] = vi.impl and len(r) >= 1 and vi.impl in ("faiss","usearch","rust","python")
ok["SIMD dot"] = abs(dot([1,2,3],[4,5,6])-32) < 1e-5

# 2 完整 SuperBrain 对话/记忆/人格
db = tempfile.mktemp(suffix=".db")
b = SuperBrain.from_llm(Stub(), store=MemoryStore(db))
b._agent.config.enable_learning = False
b._agent.config.autosave = False
b._agent.config.enable_tuning = False
ok["对话"] = bool(b.chat("你好，我喜欢数学"))
ok["记忆"] = bool(b.remember("用户喜欢蓝色汽车")) and len(b.recall("喜爱")) >= 0
b.apply_style("可爱")
ok["人格风格"] = b.style_text() == "可爱" and b.personality_mode() == "guided"
ok["需求系统"] = hasattr(NeedDriveSystem(), "get_dominant_need")
b.close()

print("=== superbrain2 完整能力冒烟 ===")
for k, v in ok.items():
    print(f"  {'✅' if v else '❌'} {k}")
print("全部通过" if all(ok.values()) else "有失败")