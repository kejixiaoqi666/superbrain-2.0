"""超脑 SuperBrain —— 集成门面（facade）。

其他 agent 接入超脑的唯一入口：一行代码装配全部 30+ 模块，
暴露统一接口，隐藏内部复杂性。

用法：
    from superbrain import SuperBrain
    brain = SuperBrain.from_env()   # 一行装配（自动读环境变量配 LLM）
    brain.chat("你好")              # 对话（含认知+记忆+人格全链路）
    brain.remember("用户喜欢茶")     # 记一条
    brain.recall("用户喜欢什么")     # 查记忆
    brain.state()                   # 看内在状态
    brain.save() / brain.load()     # 持久化 / 恢复
"""

from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import asdict

from .core.agent import SuperBrainAgent, AgentConfig
from .core.llm import LLMProvider, from_env as _llm_from_env
from .core.memory.store import MemoryStore
from .core.meme_tool import search_meme


class SuperBrain:
    """超脑集成板块：门面类，封装认知+记忆+人格+行动+进化全链路。

    其他 agent 只需 `SuperBrain.from_env()` 即可使用全部能力，
    不必关心内部 30+ 模块如何装配。
    """

    def __init__(self, llm: Optional[LLMProvider] = None,
                 store: Optional[MemoryStore] = None,
                 config: Optional[AgentConfig] = None) -> None:
        llm = llm or _llm_from_env()
        self._agent = SuperBrainAgent(llm, store=store, config=config)

    # ---------- 工厂 ----------

    @classmethod
    def from_env(cls, **kwargs) -> "SuperBrain":
        """从环境变量一键装配（自动配 LLM、记忆库）。

        环境变量：
        - SUPERBRAIN_LLM_BASE / SUPERBRAIN_LLM_KEY / SUPERBRAIN_LLM_MODEL
        - SUPERBRAIN_DB（记忆库路径，默认 ~/.superbrain/brain.db）
        """
        return cls(**kwargs)

    @classmethod
    def from_llm(cls, llm: LLMProvider, store: Optional[MemoryStore] = None,
                 **kwargs) -> "SuperBrain":
        """用指定 LLM 装配（供其它 agent 传入自己的模型）。"""
        return cls(llm=llm, store=store, **kwargs)

    # ---------- 统一接口（隐藏内部复杂性） ----------

    def chat(self, message: str, person_id: str = None,
             images: list = None) -> str:
        """对话：认知（需求/情绪/神经化学/元认知/注意力）→ 记忆检索 → 行动。

        person_id 传入时，超脑随相处自主演化对该人的关系（信任/熟悉/依恋增长 + 自主重新定性）。
        images 传入时走多模态（base64 data URL 列表，模型直接识图）。
        """
        return self._agent.chat(message, person_id=person_id, images=images)

    def chat_stream(self, message: str, person_id: str = None,
                    images: list = None):
        """对话（轻量流式：单次调用+真流式），yield ("text", 增量) 等。

        保留 persona+记忆, 免重型工具/人格块提速; 供聊天机器人真流式输出。
        """
        return self._agent.chat_stream(message, person_id=person_id, images=images)

    def usage(self) -> dict:
        """当轮真实 token 用量（向量压缩后喂给模型的输入）——供上游自动续接判断。"""
        try:
            return self._agent.usage()
        except Exception:
            return {"prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "calls": 0}

    def orientations(self) -> dict:
        """超脑对每个人的自主关系定位（相处中自主形成，非绑定标签）。"""
        return self._agent.relationships.orientations()

    # ---- 经验蒸馏（大脑存知识，身体管执行）----
    def record_experience(self, task: str, context: str = "",
                          outcome: str = "", lesson: str = ""):
        return self._agent.record_experience(task, context, outcome, lesson)

    def distill_skill(self, name: str, procedure: str, success: bool = True):
        return self._agent.distill_skill(name, procedure, success=success)

    def best_skills(self, k: int = 5) -> list:
        return self._agent.best_skills(k)

    def expression_elements(self, orientation: str = None) -> dict:
        """底层表达元素库：某关系定位下超脑可选用的言语元素（框架范围，非锁定）。

        orientation=None 返回全部分类元素清单；传入 lover/friend/close 等返回该定位可选范围。
        """
        from .core import expression_library as el
        if orientation is None:
            return el.all_elements()
        return el.elements_for(orientation)

    def remember(self, content: str, scope: str = "user",
                 tier: str = "recall", **kwargs) -> str:
        """写入一条记忆。返回节点 id。"""
        node = self._agent.remember(content, scope=scope, tier=tier, **kwargs)
        return node.node_id

    def recall(self, query: str, k: int = 5) -> List[Dict]:
        """检索记忆。返回 [{content, score, why}] 列表。"""
        hits = self._agent.recall(query, k=k)
        return [{"content": n.content, "score": round(s, 4), "why": w}
                for n, s, w in hits]

    def state(self) -> dict:
        """内在状态快照（需求/情绪/神经化学/元认知/注意力）。"""
        snap = self._agent.state_snapshot()
        snap["attention"] = self._agent.attention.history[-1].to_dict() \
            if self._agent.attention.history else None
        return snap

    def save(self, path: Optional[str] = None) -> str:
        """持久化整个大脑（一个 db = 全部状态）。"""
        return self._agent.save(path)

    def load(self, path: Optional[str] = None) -> bool:
        """从持久化恢复。"""
        return self._agent.load(path)

    def dream(self, window_hours: float = 24.0) -> List[str]:
        """睡眠计算：回顾记忆提炼经验。"""
        return self._agent.dream(window_hours)

    def act(self, goal=None):
        """主动行动：需求派生目标→规划→执行。"""
        return self._agent.act(goal)

    def generate_thoughts(self):
        """自主思考：基于内在状态产生主动想法（纯产生，不入队）。返回本次新产生的想法列表，
        直接使用返回值发送；如需队列消费，先 enqueue 再 drain_thoughts()。"""
        return self._agent.autonomous.generate(
            self._agent.needs, self._agent.emotion, self._agent.relationships)

    def drain_thoughts(self):
        """取出并清空待发自主想法队列（供上层主动消费）。"""
        return self._agent.autonomous.drain()

    def tick(self) -> dict:
        """核心一次推进（策略入口，供上层 agent 定时调用）。

        超脑是核心库非智能体——本方法只做「一次推进」，**何时调用、是否据返回的
        主动内容打扰用户，由上层 agent 决定**；超脑自身不做无人值守/定时/常驻。

        聚合一轮内化推进：
          1. 需求/情绪/神经化学代谢（内部状态随时间演进）
          2. 自主想法生成（想念/关心/好奇/分享，有才产出）
          3. 自主目标生成（冷却去重）
          4. 人格自省 reflect（基于近期表达内化）
          5. 用户画像→人格升格（从长期相处习得表达倾向）

        返回本轮产出的主动内容 dict：
          {"thoughts": [...], "new_goals": [...], "reflected": [...], "absorbed": [...]}
        全空 = 本轮无主动产出，上层应静默（不打扰用户）。
        """
        a = self._agent
        # 1. 内部代谢
        try:
            a.needs.tick()
            sats = {nt.value: a.needs.needs[nt].current_level for nt in a.needs.needs}
            a.emotion.update(sats)
            a.neurochem.tick()
            a.neurochem.observe(a.emotion.state.valence, a.emotion.state.arousal)
        except Exception:
            pass
        # 2. 自主想法（纯产生，不入队不自动发）
        thoughts = []
        try:
            produced = a.autonomous.generate(a.needs, a.emotion, a.relationships) or []
            thoughts = [asdict(t) for t in produced if hasattr(t, "type")]
        except Exception:
            thoughts = []
        # 3. 自主目标（本次新涌现，冷却去重由引擎内部处理）
        new_goals = []
        try:
            goals = a.generate_goals() or []
            new_goals = [g.to_dict() for g in goals if hasattr(g, "to_dict")]
        except Exception:
            new_goals = []
        # 4. 人格自省（内化调整）
        reflected = []
        try:
            reflected = a._self_reflect() or []
        except Exception:
            reflected = []
        # 5. 用户画像→人格升格
        absorbed = []
        try:
            absorbed = a.absorb_user_profiles() or []
        except Exception:
            absorbed = []
        return {"thoughts": thoughts, "new_goals": new_goals,
                "reflected": reflected, "absorbed": absorbed}

    def absorb_user_profiles(self) -> List[str]:
        """从所有用户长期画像升格人格（用户画像→关系→人格三级联动）。"""
        return self._agent.absorb_user_profiles()

    def personality(self) -> dict:
        """人格维度画像：可度量的人格维度框架（长出来的，非预设）。"""
        return self._agent.personality.profile()

    def set_personality(self, dimension: str, value: float,
                        source: str = "explicit", note: str = "") -> bool:
        """显式设定某人格维度（用户反馈「你应该更X」）。返回是否成功。"""
        return self._agent.personality.set(dimension, value,
                                           source=source, note=note) is not None

    def personality_mode(self) -> str:
        """当前进化模式开关：autonomous(自主演化) 或 guided(用户主导)。"""
        return self._agent.personality.mode

    def embedder_info(self) -> dict:
        """当前记忆嵌入器自描述（模型/维度/版本指纹）。"""
        return self._agent.embedder_info()

    def ensure_embedder_ready(self, auto_reembed: bool = True) -> dict:
        """语义漂移守卫：校验库内向量与当前嵌入器可比较，必要时自动重嵌入。"""
        return self._agent.ensure_embedder_ready(auto_reembed=auto_reembed)

    def set_personality_mode(self, mode: str) -> bool:
        """设定进化模式开关：autonomous(自主演化) 或 guided(用户主导,自动通道全停)。

        guided 模式下 observe/reflect/画像升格 全部暂停，人格只由用户 set/apply_style 决定。
        """
        return self._agent.personality.set_mode(mode)

    def apply_style(self, style: str) -> List[str]:
        """一句话风格设定（任意自然语言，如可爱/冷静/高冷/傲娇...，不限于预置清单）。

        - 预置风格(可爱/温柔/活泼/冷静/沉稳/高冷/理智/热情/随和/严肃) → 映射为大五
          维度组合显式设定(explicit)，返回被设定的维度名；
        - 自定义风格(如傲娇/干练/古灵精怪) → 不强套维度，仅记录为表达风格 custom_style；
        - 两种都会自动切到 guided 用户主导模式（自动通道全停，听用户的）。
        """
        return self._agent.personality.apply_style(style)

    def available_styles(self) -> List[str]:
        """常见风格便捷映射清单（仅作示例，不锁定——任意自然语言风格都可设）。"""
        return self._agent.personality.available_styles

    def style_text(self) -> str:
        """当前表达风格文本（custom_style，供上层注入 prompt/humanize；空串=未设定）。"""
        return self._agent.personality.style_text

    def generate_goals(self):
        """自主目标生成：基于需求/情绪/关系/人格维度涌现中长期意图，返回本次新产生的目标。"""
        return self._agent.generate_goals()

    def autonomous_goals(self) -> List[Dict]:
        """当前自主目标清单（含 active/completed/abandoned）。"""
        return self._agent.auto_goals.summary()

    def adopt_goals(self) -> List[Dict]:
        """把 active 自主目标采纳进目标系统（意图→目标闭环），返回新采纳的 Goal。"""
        return [g.to_dict() for g in self._agent.adopt_autonomous_goals()]

    def user_profile(self, person_id: str) -> Optional[Dict]:
        """被动用户画像：对某个人被动积累的了解（沟通风格/情绪基调/话题/偏好）。"""
        up = self._agent.user_profiles.get(person_id)
        return up.to_dict() if up else None

    def humanize(self, text: str, person_id: str = None) -> str:
        """人性化表达包装：按情绪 + 人格维度 + 与该人亲密度/昵称/画像生成口语化文本。"""
        emotion = self._agent.emotion.state
        familiarity, nickname = 0.5, ""
        user_profile = None
        if person_id:
            rel = self._agent.relationships.get(person_id)
            if rel:
                familiarity, nickname = rel.familiarity, rel.nickname
            user_profile = self._agent.user_profiles.get(person_id)
        result = self._agent.humanize.humanize(
            text, emotion, familiarity=familiarity, nickname=nickname,
            personality=self._agent.expression_personality(person_id),
            user_profile=user_profile)
        self._agent.record_expression()   # 记录本次表达，供 dream 自省内化
        return result

    def set_user_style(self, person_id: str, style: str) -> bool:
        """为该用户记录表达风格（per-user 用户主导）。

        对该用户预置风格(可爱/冷静...)会在 humanize 时覆盖表达倾向，不影响全局人格。
        """
        return self._agent.set_user_style(person_id, style)

    def user_style(self, person_id: str) -> str:
        """该用户的表达风格文本（无则空串）。"""
        return self._agent.user_style(person_id)

    def search_meme(self, query: str, limit: int = 5):
        """搜索表情包，返回 [{url, title, source}]。"""
        return search_meme(query, limit)

    def index_concepts(self, limit: int = 20) -> int:
        """抽取概念建图（GraphRAG）。"""
        return self._agent.index_concepts(limit)

    def deduplicate(self, threshold: float = 0.85) -> int:
        """合并近似记忆。"""
        return self._agent.deduplicate(threshold)

    def close(self) -> None:
        self._agent.close()

    # ---------- 内部透传（高级用法） ----------

    @property
    def agent(self) -> SuperBrainAgent:
        """底层 agent（高级用户直接访问全部模块）。"""
        return self._agent

    def __repr__(self) -> str:
        return (f"<SuperBrain v{self._agent.__class__.__module__.split('.')[0]} "
                f"mem={self._agent.store.count_nodes()} "
                f"need={self._agent.needs.get_dominant_need()[0].value if self._agent.needs.get_dominant_need()[0] else '无'}>")
