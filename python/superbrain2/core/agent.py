"""核心 Agent：记忆层 + 认知层协同的主循环。

感知 → 需求评估 → 情绪更新 → 记忆检索（受情绪/需求影响）→ 行动 → 学习。

协同点：
- 情绪影响记忆写入权重（高唤醒/高支配时记忆更重要，更值得记）
- 主导需求决定行动倾向（如确定性需求高 → 主动检索/探索）
- 情绪/需求状态注入 system prompt，让 LLM 感知"内在状态"
"""

from __future__ import annotations

import json
import logging
import copy
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

from .cognition.needs import NeedDriveSystem, NeedType
from .cognition.emotion import EmotionGradient
from .cognition.self_tune import SelfTuner
from .cognition.neurochem import NeuroChemistry
from .cognition.metacognition import Metacognition
from .memory.store import MemoryStore
from .memory.node import MemoryNode, fingerprint
from .memory.embeddings import build_embedder, embedder_identity
from .memory import retrieval

from .memory.working import WorkingMemory
from .memory.eventlog import EventLog
from .memory.time_sense import fuzzy_time
from .memory.pisa import SchemaEngine
from .memory.expression_guide import analyze_memory_scale, expression_guide
from .attention import AttentionEngine, attention_block
from .context_cleaner import clean_tool_output, clean_history_message, is_noise
from .gwt import GlobalWorkspace
from .memory.distill import Distiller, Experience
from .memory.dream import DreamEngine
from .memory.concept import ConceptExtractor, ConceptGraph
from .memory import dedupe
from .llm import LLMProvider, _content_text, estimate_tokens, multimodal_message
from .tools import ToolRegistry, PermissionPolicy, register_builtin_tools
from .goals import GoalManager, Goal
from .state import (save_state, load_state, save_state_to_store,
                    load_state_from_store)
from .planner import LLMPlanner
from .scheduler import Scheduler
from .autonomous import AutonomousThoughtEngine
from .humanize import HumanizeEngine
from .meme_tool import register_meme_tool
from .personality.seed import Seed, SeedBank
from .personality.values import ValueSystem
from .personality.relationship import RelationshipGraph
from .personality.self_model import SelfModel
from .personality.identity import Identity
from .personality.dimensions import PersonalityDimensions
from .learning import LearningExtractor
from .autonomous_goals import AutonomousGoalEngine
from .user_profile import UserProfileGraph

logger = logging.getLogger("superbrain.agent")


@dataclass
class AgentConfig:
    name: str = "超脑"
    system_prompt: str = "你是「超脑」，一个带内在需求与情绪的自主智能体。用中文回答。"
    max_iterations: int = 10
    enable_memory: bool = True
    enable_tuning: bool = True
    enable_learning: bool = True   # 每轮对话后自动抽取种子/关系/事实
    max_context_tokens: int = 8000   # 上下文 token 预算（防无限增长）
    history_budget: int = 3000       # 对话历史 token 预算
    autosave: bool = True            # 自动保存（每 N 轮，防忘记 save 导致失忆）
    autosave_interval: int = 10      # 每 N 轮自动保存一次
    artifact_threshold: int = 2000   # 工具输出超过此字符数则 artifact 化（TokenPilot）
    artifact_upgrade_hits: int = 3   # artifact 被访问此次数后恢复完整内容（Ingestion Gate）
    compress_threshold: int = 4000   # 超过此 token 触发旧对话→摘要压缩
    humanize_output: bool = False    # 对外表达人性化（情绪口语化/卖萌）；默认关保持核心库纯粹，学习/记忆仍用原始答复
    # 嵌入器参数（增强版：同义词归一 + 去停用词，提升检索质量；默认关=兼容旧库向量）
    embed_enhance: bool = False
    embed_strip_stop: bool = False
    # 真实 embedding 模型（可插拔，脱离架构单独安装）：
    #   embedder='hashing'(默认,零依赖) | 'bge'(ONNX 中文语义模型, 需 install-model + 依赖)
    #   embedder_kwargs 透传，如 {'dim': 768} / {'model_dir': '...'} / {'enhance': True}
    embedder: str = "hashing"
    embedder_kwargs: dict = field(default_factory=dict)


DEFAULT_PROMPT = "你是「超脑」智能体，有内在需求、情绪和记忆，可自主行动。请用中文回答。"


class _StreamToolCall:
    """流式 tool_calls 增量聚合出的工具调用——兼容 _dispatch_tool(tc.name/tc.args_dict())。"""
    __slots__ = ("name", "_args")

    def __init__(self, name: str, args_json: str):
        self.name = name
        try:
            self._args = json.loads(args_json) if args_json else {}
        except Exception:
            self._args = {}

    def args_dict(self) -> dict:
        return self._args


class SuperBrainAgent:
    def __init__(self, llm: LLMProvider, store: Optional[MemoryStore] = None,
                 config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.llm = llm
        self.store = store or MemoryStore()
        self.needs = NeedDriveSystem()
        self.emotion = EmotionGradient()
        self.neurochem = NeuroChemistry()
        self.meta = Metacognition()
        self.tuner = SelfTuner() if self.config.enable_tuning else None
        _ekw = dict(self.config.embedder_kwargs)
        _ekw.setdefault("enhance", self.config.embed_enhance)
        _ekw.setdefault("strip_stop", self.config.embed_strip_stop)
        self._embedder = build_embedder(self.config.embedder, **_ekw)
        # 输出是否归一化（hashing 与 bge 均归一化）→ 决定检索用点积快路径
        self.store.normalized = bool(getattr(self._embedder, "normalized", False))
        self.distiller = Distiller()
        self.dreamer = DreamEngine(self.store, self.distiller)
        self.tools = ToolRegistry()
        self.permissions = PermissionPolicy()
        register_builtin_tools(self.tools)
        self.goals = GoalManager()
        self.planner = LLMPlanner(llm)
        self.scheduler = Scheduler()
        self.concept_extractor = ConceptExtractor(llm)
        self.concept_graph = ConceptGraph(self.store)
        self.seeds = SeedBank()
        self.values = ValueSystem()
        self.relationships = RelationshipGraph()
        self.self_model = SelfModel(identity="超脑", capability=["对话", "记忆", "工具执行", "规划"])
        self.identity = Identity(name="超脑")
        self.personality = PersonalityDimensions()      # 人格维度框架（长出来的，非预设）
        self.auto_goals = AutonomousGoalEngine()        # 自主目标生成（内在状态涌现意图）
        self.user_profiles = UserProfileGraph()         # 被动用户画像（随相处长出来）
        self._expressions = deque(maxlen=20)            # 近期 humanize 实际表现的人格维度（自省内化素材）
        self.learning = LearningExtractor(llm) if self.config.enable_learning else None
        self.working = WorkingMemory()
        self.event_log = EventLog()
        self.schema_engine = SchemaEngine()      # PISA 三模态演化
        self.gwt = GlobalWorkspace(self._gwt_controller)  # 全局工作空间
        self._register_gwt_modules()
        self.attention = AttentionEngine()  # 注意力/显著性引擎
        self.autonomous = AutonomousThoughtEngine()  # 自主思考引擎
        self.humanize = HumanizeEngine()      # 人性化表达引擎
        register_meme_tool(self.tools)        # 表情包搜索工具
        self._active_rel = None  # 当前对话对象的关系（person_id 时）
        self._conversation: List[Dict[str, str]] = []
        # 当轮真实 token 用量（向量压缩后喂给模型的输入，供上游自动续接判断）
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0, "calls": 0}
        self._compressed_summary: str = ""   # 旧对话的滚动摘要
        self._last_compress = 0.0
        self._last_message: str = ""   # 最近一条消息（GWT 种子模块用）
        self._artifacts: Dict[str, str] = {}  # 工具输出 artifact 存储（TokenPilot）
        self._artifact_access: Dict[str, int] = {}  # artifact 访问频率（Ingestion Gate）
        self._base_cache: str = ""  # 静态 base（身份+价值观）缓存，避免每轮重算
        self._stats = {"turns": 0}

    # ---------- 状态快照 ----------

    def state_snapshot(self) -> dict:
        dom, drive = self.needs.get_dominant_need()
        return {
            "needs": self.needs.get_profile(),
            "emotion": self.emotion.state.to_dict(),
            "neurochem": self.neurochem.to_dict(),
            "dominant_need": dom.value if dom else None,
            "dominant_drive": round(drive, 3),
            "memory_nodes": self.store.count_nodes(),
            "intrinsic_reward": round(self.emotion.mean_reward, 3),
            # 生命感板块可观察状态：自主消息积压 + 人性化可爱度
            "autonomous": {"pending_thoughts": self.autonomous.count},
            "humanize": {"cute_probability": self.humanize.cute_probability},
            # v1.22.0 框架能力可观察状态
            "personality": self.personality.profile(),
            "autonomous_goals": self.auto_goals.summary(),
            "user_profiles": self.user_profiles.summary(),
        }

    def _state_block(self) -> str:
        """精简内在状态（只注入关键值，省 token）。"""
        dom, drive = self.needs.get_dominant_need()
        e = self.emotion.state
        return (
            f"[状态] 主导需求={dom.value if dom else '无'} 驱动={round(drive,2)} "
            f"情绪={round(e.valence,2)}/{round(e.arousal,2)} "
            f"化学底色={round(self.neurochem.momentum(),2)} "
            f"记忆={self.store.count_nodes()}条"
        )

    def _base_prompt(self) -> str:
        """静态 base（system prompt + 身份 + 价值观），惰性缓存避免每轮重算。"""
        if self._base_cache:
            return self._base_cache
        parts = []
        anchor = self.identity.anchor_text()
        if anchor:
            parts.append(anchor)
        principle = self.values.principle_text()
        if principle:
            parts.append(f"核心原则：{principle}")
        base = (self.config.system_prompt or DEFAULT_PROMPT)
        if parts:
            base += "\n" + "\n".join(parts)
        base += (
            "\n诚实底线：绝不捏造事实。【不得假装执行过命令、不得谎称读过本机/磁盘/硬件、"
            "不得编造实时价格/数据/来源】。你有经门禁的工具(exec/web/write/git)："
            "只读操作直接真实执行，带副作用的写/执行会返回'需批准'——遇到'需批准'必须如实转达用户，"
            "不得擅自继续或假装完成。工具真实执行后只报真实结果；未执行的绝不能说'已执行'。"
            "无对应工具/数据源时如实说没有。"
        )
        self._base_cache = base
        return base

    # ---------- 记忆检索（受情绪影响） ----------

    def recall(self, query: str, k: int = 5) -> List[Tuple[MemoryNode, float, str]]:
        """检索记忆。情绪唤醒高时扩大召回 + 情绪偏好加权（Sentipolis 情绪-记忆耦合）。

        当前情绪(positive/negative)加权匹配的记忆：高兴时更容易想起高兴的经历。
        """
        if query is None:
            query = ""
        elif not isinstance(query, str):
            query = str(query)
        if not isinstance(k, int) or k <= 0:
            k = 5  # 脏 k（如字符串）防御，避免切片崩
        vec = self._embed(query)
        hits = retrieval.search(self.store, vec, query, k=k)
        # 情绪一致性记忆提取（认知-情绪回路）：以心境(mood)为一致性信号——
        # 心境低落时偏向想起负面记忆、高涨时想起正面；不一致的轻微抑制。
        mood = self.emotion.state.mood  # 慢变量心境 -1..1（比瞬时更稳定地偏置回忆基调）
        for i, (node, score, why) in enumerate(hits):
            sim = mood * node.valence
            boost = sim * 0.25          # 同向情绪记忆上扬（增强偏置）
            if sim < 0:
                boost *= 0.5             # 情绪不一致的记忆略抑制
            hits[i] = (node, score + boost, f"{why}+情绪一致")
        # 唤醒高时多召回一点（更愿意联想）
        eager_k = k + int(self.emotion.state.arousal * 2) if self.emotion.state.arousal > 0.6 else k
        return hits[:max(k, eager_k)]

    def remember(self, content: str, tags: Optional[List[str]] = None,
                 importance: float = 0.5, scope: str = "user",
                 tier: str = "recall") -> MemoryNode:
        """写入记忆。情绪唤醒/支配高时提高置信度（更重要，更值得记）。"""
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        conf = min(1.0, 0.5 + self.emotion.state.arousal * 0.3 + self.emotion.state.dominance * 0.2)
        node = MemoryNode(
            node_id=fingerprint(content)[:16],
            content=content,
            embedding=self._embed(content),
            tags=tags or [],
            confidence=conf,
            scope=scope,
            tier=tier,
            valence=self.emotion.state.valence,  # 情绪标签（Sentipolis 耦合）
        )
        self.store.add(node)
        # 满足确定性需求（获得了新信息）
        self.needs.satisfy(NeedType.CERTAINTY, 0.05)
        return node

    def _embed(self, text: str) -> List[float]:
        """用当前嵌入器生成向量（hashing 为 blake2b 特征哈希；bge 为真实语义模型）。"""
        return self._embedder.embed(text)

    # ---------- 嵌入器一致性 / 语义漂移守卫 ----------

    def embedder_info(self) -> dict:
        """当前嵌入器自描述（模型/维度/版本指纹），供状态面板与调试。"""
        try:
            return embedder_identity(self._embedder)
        except Exception:
            return {"type": type(self._embedder).__name__}

    def ensure_embedder_ready(self, auto_reembed: bool = True) -> dict:
        """语义漂移守卫：校验库内向量是否与当前嵌入器可比较。

        返回 {"compat": "ok"|"needs_reembed", "fingerprint": fp, "reembedded": n}。
        若库内已有向量但嵌入器换过（维度/模型/版本变化 → 向量不可比），当
        auto_reembed=True 时自动全量重嵌入并在完成后打上新的指纹，杜绝静默混算。
        """
        fp = embedder_identity(self._embedder)["fp"]
        status = self.store.embedder_compat(fp)
        if status == "ok":
            self.store.mark_embedder(fp)     # 空库/新库：打标即可
            return {"compat": "ok", "fingerprint": fp, "reembedded": 0}
        if auto_reembed:
            n = self.store.reembed_all(self._embedder.embed)
            self.store.mark_embedder(fp)
            logger.warning("嵌入器指纹变化，已重嵌入 %d 条记忆以修复语义漂移 (fp=%s)", n, fp)
            return {"compat": "reembedded", "fingerprint": fp, "reembedded": n}
        return {"compat": "needs_reembed", "fingerprint": fp, "reembedded": 0}

    # ---------- 主循环 ----------

    def chat(self, message: str, person_id: Optional[str] = None,
             images: Optional[List[str]] = None) -> str:
        # 入口防御：空/None/非 str 消息不崩、不污染会话、不调 LLM
        if message is None:
            message = ""
        elif not isinstance(message, str):
            message = str(message)
        if not message.strip() and not images:
            return "嗯？（没有收到消息内容）"
        # 自主关系演化：和某人相处，信任/熟悉/依恋随交互自然成长，超脑自主重新定性（非设定、非绑定）
        self._active_rel = None
        if person_id:
            try:
                rel = self.relationships.get_or_create(person_id)
                rel.grow(familiarity_delta=0.02, trust_delta=0.008, attachment_delta=0.004)
                self._active_rel = rel
                # 被动用户画像：零 LLM 启发式观测（沟通风格/情绪基调/话题兴趣），随相处长出来
                self.user_profiles.observe(person_id, message, name=rel.name)
            except Exception:
                self._active_rel = None
        # 人格维度被动成长：用户消息的风格/反馈塑造超脑人格（非预设，随相处长出来）
        try:
            self.personality.observe_interaction(message)
        except Exception:
            pass
        self._stats["turns"] += 1
        self._last_message = message
        # 多模态：有图则 user content 用 list（text + image_url 片段），否则纯文本
        user_content = multimodal_message(message, list(images))["content"] \
            if images else message
        self._conversation.append({"role": "user", "content": user_content})

        # 0. 记录事件日志（溯源基础）+ 工作记忆
        try:
            self.event_log.append("conversation", message, provenance="direct_observation")
            self.working.add(message, kind="event")
        except Exception:
            pass  # 记录失败不影响对话

        # 1. 感知：需求 + 情绪 tick + 神经化学
        try:
            self.needs.tick()
            sats = {nt.value: self.needs.needs[nt].current_level for nt in self.needs.needs}
            self.emotion.update(sats)
            self.neurochem.tick()  # 递质按半衰期清除（情绪动量）
            self.neurochem.observe(self.emotion.state.valence, self.emotion.state.arousal)
        except Exception:
            sats = {}  # 感知异常降级：不崩，继续

        # 2. 记忆检索（受情绪影响的联想）—— 极短消息跳过检索省性能
        if self.config.enable_memory and not self._is_trivial(message):
            try:
                recalled = self.recall(message)
                mem_block = self._format_memory(recalled)
                scale = analyze_memory_scale(recalled)
                guide = expression_guide(scale)
            except Exception:
                recalled, mem_block, guide = [], "", ""  # 检索异常降级
        else:
            recalled, mem_block, guide = [], "", ""

        # 注意力焦点（人脑选择性注意）：当前最该关注哪个维度
        try:
            dom, drive = self.needs.get_dominant_need()
            focus = self.attention.focus(
                valence=self.emotion.state.valence,
                arousal=self.emotion.state.arousal,
                dominant_need=dom.value if dom else "无",
                need_drive=drive,
                memory_hits=len(recalled),
                input_text=message,
            )
            att_block = attention_block(focus)
        except Exception:
            att_block = ""

        # 3. 组装 prompt（静态 base 用缓存，其余惰性注入）
        msgs = [{"role": "system", "content": self._base_prompt()}]
        msgs.append({"role": "system", "content": self._state_block()})
        # 主导种子：当前情境下影响行为的倾向
        try:
            dominant_seed = self.seeds.dominant(message)
            if dominant_seed:
                msgs.append({"role": "system",
                             "content": f"[行为倾向] {dominant_seed.content}（倾向：{dominant_seed.behavior_bias}）"})
        except Exception:
            pass
        # 自我叙事：过去/现在/将来的我
        narrative = self.self_model.narrative()
        if narrative:
            msgs.append({"role": "system", "content": f"[自我] {narrative}"})
        # 工作记忆：当前任务/未完成事项（短期，容量受限）
        wm = self.working.summary()
        if wm:
            msgs.append({"role": "system", "content": f"[工作记忆] {wm}"})
        if mem_block:
            msgs.append({"role": "system", "content": f"[相关记忆]\n{mem_block}"})
        if guide:
            msgs.append({"role": "system", "content": guide})
        if att_block:
            msgs.append({"role": "system", "content": att_block})
        # 自主关系定位：我对当前这个人的内在关系认知（超脑自主判断，非绑定标签）
        if self._active_rel is not None:
            rel = self._active_rel
            nick = f"，昵称「{rel.nickname}」" if rel.nickname else ""
            msgs.append({"role": "system", "content":
                f"[我对{rel.name or '对方'}的关系] 定位={rel.orientation} "
                f"(把握{rel.orientation_confidence:.2f})，依恋{rel.attachment:.2f}，"
                f"熟悉{rel.familiarity:.2f}{nick}。这是我相处中自主形成的判断。"})
        # 人格维度：只注入显著偏离中性且有把握的维度（框架长出来的，非预设）
        try:
            personality_text = self.personality.profile_text()
            if personality_text:
                msgs.append({"role": "system", "content": f"[{personality_text}]"})
        except Exception:
            pass
        # 被动用户画像：对当前这个人被动积累的了解（只列有实质观测的）
        if self._active_rel is not None:
            try:
                up = self.user_profiles.get(self._active_rel.person_id)
                up_text = up.profile_text() if up else ""
                if up_text:
                    msgs.append({"role": "system", "content": f"[{up_text}]"})
            except Exception:
                pass
        # 对话历史：按 token 预算截断（而非固定条数，避免无限增长）
        msgs.extend(self._recent_history())

        # 4. 行动：function-calling 闭环（LLM 决定调用工具 → 执行 → 回传）
        answer = self._run_tool_loop(msgs)

        # 5. 学习：情绪更新 + 记忆沉淀 + 自适应调参 + 自动抽取学习信号
        success = 1.0 if answer else 0.0
        # 情绪感染：与重要且长期低落的人相处 → 共情，自身情绪被牵引（社会情绪）
        contagion = 0.0
        if self._active_rel is not None:
            try:
                up = self.user_profiles.get(self._active_rel.person_id)
                if up and up.mood_count >= 3 and up.mood_mean < -0.3:
                    # 对方情绪低落(负) → 感染源为负，拖低自身 valence；熟悉度加权
                    contagion = up.mood_mean * min(1.0, self._active_rel.familiarity)
            except Exception:
                contagion = 0.0
        self.emotion.update(sats, task_success=success, contagion=contagion)
        # 社会情绪：任务产出自豪/愧疚（基于行为与"是否做成/帮到人"）
        try:
            if success >= 1.0:
                self.emotion.social_update(pride=0.3)      # 做成事 → 轻微自豪
            elif self._active_rel is not None:
                self.emotion.social_update(guilt=0.15)     # 对重要的人没做好 → 轻微愧疚
        except Exception:
            pass
        try:
            self.emotion.compute_intrinsic_reward()  # 记录本轮内在奖励，供 mean_reward/tuner 用
        except Exception:
            pass
        if self.config.enable_memory and answer:
            try:
                self._remember_interaction(message, answer)
            except Exception:
                pass  # 沉淀失败不影响返回
        # 自动抽取种子/关系/事实（事件→种子闭环，启发式预筛省 token）
        if self.learning and answer and self._should_learn(message):
            try:
                learned = self.learning.extract(message, answer)
                self._apply_learning(learned)
                # PISA schema 演化：新对话归类/修正/创建图式
                self.schema_engine.ingest(message)
            except Exception:
                pass  # 学习失败不影响返回

        # 元认知反思：本次回答质量如何，是否高估了
        try:
            self.meta.reflect(
                decision=message[:50],
                reasoning_quality=0.5 + success * 0.3,
                confidence=self.emotion.state.confidence,
                outcome=success,
            )
        except Exception:
            pass

        if self.tuner and self.tuner.should_tune():
            try:
                reward = self.emotion.mean_reward
                fitness = self.tuner.current_fitness(reward, success)
                proposal = self.tuner.propose()
                self.tuner.evaluate_and_apply(fitness, fitness + 0.01, proposal)
            except Exception:
                pass

        # 人性化对外表达（可选）：按当前情绪做口语化/卖萌包装，仅影响返回，不影响学习与记忆沉淀
        if self.config.humanize_output and answer:
            try:
                if self._active_rel is not None:
                    rel = self._active_rel
                    answer = self.humanize.humanize(
                        answer, self.emotion.state,
                        familiarity=rel.familiarity, nickname=rel.nickname)
                else:
                    answer = self.humanize.humanize(answer, self.emotion.state)
            except Exception:
                pass
        self._conversation.append({"role": "assistant", "content": answer})
        self._maybe_compress()  # 分层记忆：超预算时旧对话→摘要
        self._maybe_autosave()  # 自动保存（防忘记 save 导致失忆）
        return answer

    def _apply_learning(self, learned: dict) -> None:
        """把抽取的学习信号写入人格层（种子/关系/事实）。"""
        # 种子：经历 → 行为倾向（去重强化 + 置信度门槛）
        for s in learned.get("seeds", []) or []:
            self.seeds.add_or_reinforce(
                content=s.get("content", ""),
                triggers=[s["trigger"]] if s.get("trigger") else None,
                behavior_bias=s.get("bias", ""),
                valence=float(s.get("valence", 0.0)),
                confidence=0.6,
            )
        # 关系：对说话者的了解
        for rel in learned.get("relationships", []) or []:
            person = rel.get("person", "")
            note = rel.get("note", "")
            if person and note:
                self.relationships.note(person, note, source="inference", confidence=0.5)
        # 事实：值得记住的确凿事实
        for fact in learned.get("facts", []) or []:
            if isinstance(fact, str) and fact.strip():
                self.remember(fact, scope="user", tier="recall")

    def _run_tool_loop(self, msgs: List[Dict], max_steps: int = 5) -> str:
        """function-calling 闭环：循环直到模型给出最终回答或达到步数上限。"""
        tools = self.tools.openai_schemas() if self.tools.list() else None
        # 当轮真实输入重置：每个逻辑轮从零计量（跨轮取 max 单次输入）
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0, "calls": 0}
        for _ in range(max_steps):
            try:
                resp = self.llm.chat(msgs, tools=tools)
            except Exception as e:
                logger.warning("LLM 调用失败: %s", e)
                return self._fallback_reply("LLM 暂时不可用")
            # 累计真实用量：单次输入取最大（工具输出累积后末次最大 = 最坏当轮输入）
            pt = getattr(resp, "prompt_tokens", 0) or 0
            ct = getattr(resp, "completion_tokens", 0) or 0
            self.last_usage["calls"] += 1
            self.last_usage["prompt_tokens"] = max(self.last_usage["prompt_tokens"], pt)
            self.last_usage["completion_tokens"] += ct
            self.last_usage["total_tokens"] = max(self.last_usage["total_tokens"],
                                                  getattr(resp, "total_tokens", 0) or 0)
            if resp.tool_calls:
                # 记录 assistant 的 tool_calls 消息
                msgs.append({
                    "role": "assistant", "content": resp.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in resp.tool_calls
                    ],
                })
                for tc in resp.tool_calls:
                    result = self._dispatch_tool(tc)
                    msgs.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "name": tc.name, "content": str(result),
                    })
                continue
            return resp.content or ""
        return resp.content or ""

    def usage(self) -> dict:
        """当轮真实 token 用量（向量压缩后喂给模型的输入）——供上游自动续接判断。"""
        return dict(self.last_usage)

    def chat_stream(self, message: str, person_id: Optional[str] = None,
                    images: Optional[List[str]] = None):
        """工具化·真流式对话：保留 persona/记忆, 接门禁工具(exec/web/write/git), 边生成边 yield。

        产出 (kind, data)：(text, 增量) / (tool, 工具调用) / (usage, 用量) /
                        (done, 全文) / (error, 信息)。模型要调工具时经权限门禁真实执行。
        """
        from .llm import multimodal_message
        if message is None:
            message = ""
        elif not isinstance(message, str):
            message = str(message)
        user_content = multimodal_message(message, list(images))["content"] \
            if images else message
        self._conversation.append({"role": "user", "content": user_content})
        msgs = [{"role": "system", "content": self._base_prompt()},
                {"role": "system", "content": self._state_block()}]
        try:
            recalled = self.recall(message)
            mem_block = self._format_memory(recalled)
            if mem_block:
                msgs.append({"role": "system", "content": f"[相关记忆]\n{mem_block}"})
        except Exception:
            pass
        msgs.append({"role": "user", "content": user_content})
        tools = self.tools.openai_schemas() if self.tools.list() else None
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0, "calls": 0}
        acc = ""
        for _ in range(5):
            calls: Dict[int, dict] = {}
            for kind, data in self.llm.chat_stream(msgs, tools=tools):
                if kind == "text":
                    acc += data
                    yield ("text", data)
                elif kind == "usage":
                    self.last_usage["calls"] += 1
                    self.last_usage["prompt_tokens"] = max(
                        self.last_usage["prompt_tokens"], data.get("prompt_tokens", 0) or 0)
                    self.last_usage["completion_tokens"] += data.get("completion_tokens", 0) or 0
                    self.last_usage["total_tokens"] = max(
                        self.last_usage["total_tokens"], data.get("total_tokens", 0) or 0)
                elif kind == "error":
                    yield ("error", data)
                    return
                elif kind == "tool":
                    idx = data.get("index", 0)
                    c = calls.setdefault(idx, {"id": data.get("id", ""), "name": "", "args": ""})
                    fn = data.get("function") or {}
                    if fn.get("name"):
                        c["name"] = fn["name"]
                    if fn.get("arguments"):
                        c["args"] += fn["arguments"]
            if not calls:
                break
            tc_msgs = [{"id": c["id"] or f"call_{i}", "type": "function",
                        "function": {"name": c["name"], "arguments": c["args"]}}
                       for i, c in sorted(calls.items())]
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": tc_msgs})
            for i, c in sorted(calls.items()):
                try:
                    result = self._dispatch_tool(_StreamToolCall(c["name"], c["args"]))
                except Exception as e:
                    result = f"[工具 {c['name']} 异常] {e}"
                msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                             "content": str(result)[:4000]})
        self._conversation.append({"role": "assistant", "content": acc})
        yield ("done", acc)

    def _fallback_reply(self, reason: str) -> str:
        """LLM 失败时的降级回复（不崩会话）。"""
        return (f"[服务降级] {reason}。请稍后再试，或检查模型通道配置。"
                f"（当前需求：{self.needs.get_dominant_need()[0].value if self.needs.get_dominant_need()[0] else '无'}）")

    def _dispatch_tool(self, tc) -> str:
        """执行一次工具调用，带权限门禁 + 大输出 artifact 化（TokenPilot）。"""
        tool = self.tools.get(tc.name)
        if tool is None:
            return f"[错误] 未知工具 {tc.name}"
        args = tc.args_dict()
        if self.permissions.needs_approval(tool.name, tool.category, tool.side_effects):
            # 危险工具需批准：框架层面返回待批准标记，由上层决定
            return (f"[需批准] 工具 {tool.name} 是危险操作，"
                    f"需要用户批准后才能执行。参数：{json.dumps(args, ensure_ascii=False)[:200]}")
        try:
            result = tool.call(**args)
            result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"[错误] 工具 {tc.name} 执行失败: {e}"
        # 真实内容型工具(exec/read_file/web_extract)：模型需要本体, 不是 JSON 摘要
        if tool.name in ("exec", "read_file", "web_extract"):
            try:
                d = json.loads(result) if isinstance(result, str) else result
            except Exception:
                d = result
            if isinstance(d, dict) and "output" in d:
                content = str(d.get("output") or "")
                code = d.get("exit_code")
                prefix = f"[exit={code}] " if isinstance(code, int) else ""
                content = prefix + content
            else:
                content = str(result)
            cap = 12000
            if len(content) > cap:
                content = content[:cap] + f"\n...（已截断，共 {len(result)} 字符）"
            return content or "[已执行，无有效输出]"
        # 其它工具：Ingestion-Aware Compaction（TokenPilot）
        result = clean_tool_output(result)
        if not result:
            return "[已执行，无有效输出]"
        # TokenPilot：环境消息默认压缩，访问频率超阈值则恢复完整
        if len(result) > self.config.artifact_threshold:
            import hashlib
            h = hashlib.blake2b(result.encode("utf-8"), digest_size=8).hexdigest()
            self._artifacts[h] = result  # artifact 注册表 𝒜（content hash → 完整内容）
            self._artifact_access[h] = self._artifact_access.get(h, 0) + 1
            if self._artifact_access[h] >= self.config.artifact_upgrade_hits:
                return result  # 高频访问的 artifact 升级为完整内容
            summary = result[:200].replace("\n", " ")
            return (f"[artifact:{h}] 已存 artifact，摘要：{summary}..."
                    f"（需完整内容用 recall_artifact('{h}')）")
        return result

    def recall_artifact(self, h: str) -> str:
        """恢复 artifact 完整内容（工具大输出）。"""
        return self._artifacts.get(h, "[未找到该 artifact]")

    # ---- 经验蒸馏（Distiller 公开契约，供身体层调用；技能执行归身体，知识记忆归大脑）----
    def record_experience(self, task: str, context: str = "",
                          outcome: str = "", lesson: str = "") -> Experience:
        """记录一次任务经验（大脑存经验记忆）。"""
        return self.distiller.record_experience(task, context, outcome, lesson)

    def distill_skill(self, name: str, procedure: str, success: bool = True):
        """蒸馏/更新一个技能（大脑存知识；身体负责可执行形态）。"""
        return self.distiller.distill_skill(name, procedure, success=success)

    def best_skills(self, k: int = 5) -> list:
        """熟练度最高的技能（供身体同步成可执行 SKILL.md）。"""
        return self.distiller.best_skills(k)

    def dream(self, window_hours: float = 24.0) -> List[str]:
        """睡眠计算：回顾记忆 + 人格自省 + 记忆遗忘(降级非删) + 对话浓缩。"""
        if not isinstance(window_hours, (int, float)):
            window_hours = 24.0  # None/非数值防御
        self._self_reflect()
        self._forget_memory()                                  # 遗忘：长期不用的降级
        self._condense_session_memories()                      # 浓缩：30天窗口后提炼为要点
        # 情绪时间动力学：睡眠=时间流逝，瞬时情绪向心境基线回归（事件会回落，心境底色还在）
        try:
            self.emotion.elapse(dt_hours=max(window_hours, 1.0))
        except Exception:
            pass
        # 记忆库超物理上限(默认1GB)时，强制更激进浓缩（缩窗口/缩批次）
        if self.store.over_size_limit():
            c2 = self._condense_session_memories(window_hours=24.0, min_batch=2)
            logger.warning("记忆库超限，触发强制浓缩: %s", c2)
        return self.dreamer.dream(window_hours=window_hours)

    def _forget_memory(self) -> None:
        """主动遗忘：睡眠时把长期未访问、低价值的 recall 记忆降级到 archival。

        遗忘 ≠ 删除——只是脱离活跃召回(活跃集有界，性能不随总记忆无限退化)，
        用户重新提起(检索强命中)时经 store.search 自动重激活回 recall（提起来又活跃）。
        """
        try:
            from .memory.consolidation import Consolidator
            Consolidator(self.store).sleep(forget_below=0.05)
        except Exception:
            logger.debug("记忆遗忘降级失败", exc_info=True)

    def _summarize(self, system: str, text: str) -> str:
        """LLM 提炼要点；失败退化启发式（无 LLM 也能浓缩）。"""
        try:
            resp = self.llm.chat(
                [{"role": "system", "content": system},
                 {"role": "user", "content": text}], max_tokens=400)
            s = (resp.content or "").strip()
            if s:
                return s
        except Exception:
            pass
        # 退化：启发式保留含关键信号的句子（寒暄细节丢弃）
        keys = ("喜欢", "偏好", "需要", "希望", "习惯", "计划", "重要", "结论", "记住", "不")
        keep = [l for l in text.splitlines() if any(k in l for k in keys)]
        return "\n".join(keep)[:200] if keep else ""

    def _condense_session_memories(self, window_hours: float = 24 * 30.0,
                                   min_batch: int = 5) -> dict:
        """对话浓缩：把超过保留窗口的旧会话记忆，浓缩成一条要点常规记忆（优质记忆）。

        语义：保留窗口内(默认30天)全部记住；超过后开始整理——保留细节价值、
        删除多余(去重/降噪/寒暄)、把真实意义/重点提炼成一条 user 常青记忆（优质记忆），
        原始细节降级遗忘(可重激活)。返回 {"condensed","forgotten"}。
        """
        try:
            now = time.time()
            old = [n for n in self.store.by_scope("session")
                   if n.created_at and now - n.created_at > window_hours * 3600
                   and n.tier != "archival"]
            if len(old) < min_batch:
                return {"condensed": 0, "forgotten": 0}
            texts = [n.content for n in sorted(old, key=lambda n: n.created_at)]
            batch = "\n".join(texts[-60:])   # 最多浓缩最近 60 条
            summary = self._summarize(
                "你是对话记忆浓缩器。把下面的历史对话压缩成一条长期记忆，"
                "保留用户偏好、关键事实、重要结论，去掉寒暄和过程细节。中文150字内。",
                batch)
            if not summary:
                return {"condensed": 0, "forgotten": 0}
            # 存为常规记忆（保留重点关键，常青 non-session）
            self.remember(summary, scope="user", tags=["condensed"])
            # 原始细节降级遗忘（提起来可重激活，物理不删）
            for n in old:
                self.store.promote(n.node_id, "archival")
            return {"condensed": 1, "forgotten": len(old)}
        except Exception:
            logger.debug("对话浓缩失败", exc_info=True)
            return {"condensed": 0, "forgotten": 0}

    def record_expression(self) -> None:
        """记录一次 humanize 实际用人格维度驱动的表达强度（自省素材）。

        每次人性化表达时调用，把当前人格快照入队；dream 时回顾近期稳定表现。
        """
        try:
            self._expressions.append({n: t.value for n, t in self.personality.all()})
        except Exception:
            pass

    def absorb_user_profiles(self) -> List[str]:
        """从所有用户长期画像升格人格（用户画像→人格的自动内化）。

        三级联动：用户长期偏好(画像) → 相处(关系) → 自动升格人格。
        返回被升格的人格维度名列表。
        """
        try:
            return self.personality.set_from_user(
                self.user_profiles.aggregate_personality())
        except Exception:
            return []

    def set_user_style(self, person_id: str, style: str) -> bool:
        """为该用户记录表达风格（per-user 用户主导）。空 pid 或空/非字符串风格返回 False。"""
        if not person_id or not isinstance(style, str) or not (style or "").strip():
            return False
        self.user_profiles.get_or_create(person_id).set_style(style)
        return True

    def user_style(self, person_id: str) -> str:
        """该用户的表达风格文本（无则空串）。"""
        up = self.user_profiles.get(person_id)
        return up.style if up else ""

    def expression_personality(self, person_id: str = None):
        """返回用于该用户表达的人格。

        - 无 person_id → 全局人格；
        - 该用户设了预置风格(可爱/冷静...) → 以全局人格为基底，用该风格维度覆盖
          （对该用户呈现指定表达倾向，不污染全局人格）；
        - 自定义风格(傲娇...) → core 不预设映射，仍用全局人格（交给上层注入表达）。
        """
        pd = self.personality
        if not person_id:
            return pd
        up = self.user_profiles.get(person_id)
        style = up.style if up else ""
        if not style:
            return pd
        # 预置风格 → 覆盖维度；自定义风格 core 不硬套
        from .personality.dimensions import _STYLE_PRESETS
        preset = _STYLE_PRESETS.get(style)
        if not preset:
            return pd
        overlay = copy.deepcopy(pd)
        for dim, val in preset.items():
            t = overlay._traits.get(dim)
            if t is None:
                continue
            t.value = val
            t.confidence = max(t.confidence, 0.9)
        return overlay

    def _self_reflect(self) -> List[str]:
        """人格主动自省：汇总近期表达特征 → reflect() 内化（防脏数据/空队列）。

        返回本次被内化调整的人格维度名列表。
        """
        try:
            if not self._expressions:
                return []
            feats = {}
            for dim in ("openness", "conscientiousness", "extraversion",
                        "agreeableness", "neuroticism"):
                vals = [ex.get(dim) for ex in self._expressions
                        if isinstance(ex.get(dim), (int, float))]
                if vals:
                    feats[dim] = sum(vals) / len(vals)
            return self.personality.reflect(feats)
        except Exception:
            return []

    # ---------- 全局工作空间 GWT ----------

    def _register_gwt_modules(self) -> None:
        """注册认知子系统为竞争模块（情绪/记忆/目标/元认知/种子/神经化学）。"""
        self.gwt.register("情绪", self._gwt_emotion)
        self.gwt.register("记忆", self._gwt_memory)
        self.gwt.register("目标", self._gwt_goal)
        self.gwt.register("元认知", self._gwt_meta)
        self.gwt.register("种子", self._gwt_seed)

    def _gwt_emotion(self) -> dict:
        e = self.emotion.state
        # 情绪越强(偏离中性)紧迫度越高
        urgency = min(1.0, abs(e.valence) * 0.8 + e.arousal * 0.4)
        mood = "积极" if e.valence > 0.2 else "消极" if e.valence < -0.2 else "平静"
        return {"content": f"当前情绪{mood}（valence={round(e.valence,2)}, 唤醒={round(e.arousal,2)}）",
                "urgency": urgency, "salience": abs(e.valence)}

    def _gwt_memory(self) -> dict:
        # 记忆检索的显著度：主导种子或新记忆驱动
        n = self.store.count_nodes()
        return {"content": f"有 {n} 条长期记忆可用", "urgency": 0.3,
                "salience": 0.3 if n > 0 else 0.0}

    def _gwt_goal(self) -> dict:
        active = [g for g in self.goals._goals if g.status != "completed"]
        if not active:
            return {"content": "", "urgency": 0, "salience": 0}
        g = active[0]
        return {"content": f"当前目标：{g.description[:40]}",
                "urgency": 0.6, "salience": 0.8}

    def _gwt_meta(self) -> dict:
        m = self.meta.cautiousness if hasattr(self.meta, "cautiousness") else 0.5
        refls = len(getattr(self.meta, "_reflections", []))
        return {"content": f"元认知：谨慎度{m}, 反思{refls}次",
                "urgency": 0.3, "salience": 0.4}

    def _gwt_seed(self, context: str = "") -> dict:
        s = self.seeds.dominant(self._last_message or "")
        if not s:
            return {"content": "", "urgency": 0, "salience": 0}
        return {"content": f"倾向：{s.content[:40]}",
                "urgency": 0.5, "salience": 0.5 * s.effective_strength()}

    def _gwt_controller(self, workspace_text: str) -> str:
        """统一行动：工作区内容决定优先处理什么。"""
        if self.learning:
            return f"[GWT决策] 基于工作区：{workspace_text[:80]}"
        return workspace_text[:80]

    def gwt_act(self) -> Optional[str]:
        """执行一次全局竞争→广播→行动。"""
        self._last_message = self._last_message or ""
        return self.gwt.act()

    # ---------- 状态持久化 ----------

    def _session_snapshot(self) -> dict:
        """会话态快照：对话历史 + 滚动摘要 + 工作记忆 + 事件日志。"""
        return {
            "conversation": list(self._conversation),
            "summary": self._compressed_summary,
            "working": self.working.to_dict(),
            "event_log": self.event_log.to_dict(),
            "schemas": self.schema_engine.to_dict(),
            "autonomous": self.autonomous.to_dict(),
            "humanize": self.humanize.to_dict(),
            "gwt_winner": [b.module for b in self.gwt.history[-10:]],
        }

    def _session_restore(self, d: dict) -> None:
        """恢复会话态（对话/摘要/工作记忆/事件日志）。"""
        self._conversation = list(d.get("conversation", []))
        self._compressed_summary = d.get("summary", "")
        self.working = self.working.__class__.from_dict(d.get("working", {}))
        self.event_log = self.event_log.__class__.from_dict(d.get("event_log", {}))
        from .memory.pisa import SchemaEngine
        self.schema_engine = SchemaEngine.from_dict(d.get("schemas", {}))
        from .autonomous import AutonomousThoughtEngine
        try:
            self.autonomous = AutonomousThoughtEngine.from_dict(d.get("autonomous", {}))
        except Exception:
            self.autonomous = AutonomousThoughtEngine()
        from .humanize import HumanizeEngine
        try:
            self.humanize = HumanizeEngine.from_dict(d.get("humanize", {}))
        except Exception:
            self.humanize = HumanizeEngine()

    def save(self, path: Optional[str] = None) -> str:
        """持久化认知+人格+会话状态到 store（默认，一个 .db = 整个大脑）或 JSON 文件。"""
        sess = self._session_snapshot()
        if path:
            save_state(path, self.needs, self.emotion, self.distiller, self.goals,
                       seeds=self.seeds, values=self.values,
                       relationships=self.relationships, self_model=self.self_model,
                       identity=self.identity, neurochem=self.neurochem, metacog=self.meta,
                       personality=self.personality, auto_goals=self.auto_goals,
                       user_profiles=self.user_profiles, session=sess)
            return path
        save_state_to_store(self.store, self.needs, self.emotion,
                            self.distiller, self.goals, seeds=self.seeds,
                            values=self.values, relationships=self.relationships,
                            self_model=self.self_model, identity=self.identity,
                            neurochem=self.neurochem, metacog=self.meta,
                            personality=self.personality, auto_goals=self.auto_goals,
                            user_profiles=self.user_profiles, session=sess)
        return self.store.path

    def load(self, path: Optional[str] = None) -> bool:
        """恢复认知+人格+会话状态（优先 store，其次 JSON 文件）。"""
        sess = {}
        if path:
            ok = load_state(path, self.needs, self.emotion, self.distiller, self.goals,
                            seeds=self.seeds, values=self.values,
                            relationships=self.relationships, self_model=self.self_model,
                            identity=self.identity, neurochem=self.neurochem, metacog=self.meta,
                            personality=self.personality, auto_goals=self.auto_goals,
                            user_profiles=self.user_profiles, session=sess)
        else:
            ok = load_state_from_store(self.store, self.needs, self.emotion,
                                       self.distiller, self.goals, seeds=self.seeds,
                                       values=self.values, relationships=self.relationships,
                                       self_model=self.self_model, identity=self.identity,
                                       neurochem=self.neurochem, metacog=self.meta,
                                       personality=self.personality, auto_goals=self.auto_goals,
                                       user_profiles=self.user_profiles, session=sess)
        if ok and sess:
            self._session_restore(sess)
        return ok

    def enable_autopilot(self, dream_interval: float = 3600.0,
                         act_interval: float = 1800.0,
                         thought_interval: float = 900.0) -> None:
        """开启主动运行：定时睡眠计算（dream）+ 定时行动（act）。

        让 agent 从被动应答变为主动感知——定时回顾沉淀经验、定时按需求行动。
        """
        self.scheduler.add("dream", dream_interval, lambda: self.dream())
        self.scheduler.add("act", act_interval, lambda: self.act())
        self.scheduler.add("thought", thought_interval, self._tick_autonomous)
        self.scheduler.start()

    def _tick_autonomous(self) -> Optional[str]:
        """空闲触发自主思考 + 自主目标生成（想法入队、意图入库存，均幂等冷却）。"""
        try:
            for thought in self.autonomous.generate(self.needs, self.emotion,
                                                    self.relationships):
                self.autonomous.enqueue(thought)
        except Exception:
            pass
        try:
            self.generate_goals()   # 从内在状态涌现中长期自主目标（内部有冷却/去重）
        except Exception:
            pass
        return None

    def generate_goals(self):
        """自主目标生成：基于需求/情绪/关系/人格维度涌现意图，返回本次新产生的目标。"""
        return self.auto_goals.generate(
            self.needs, self.emotion, self.relationships,
            personality=self.personality)

    # ---------- 概念图 + 去重 ----------

    def index_concepts(self, limit: int = 20) -> int:
        """增量抽取概念建图（只处理未索引的新节点，省 token）。"""
        total = 0
        processed = 0
        for node in self.store.all_nodes():
            if node.kind == "entity":
                continue
            if self.store.is_concept_indexed(node.node_id):
                continue  # 已索引，跳过（增量）
            entities, relations = self.concept_extractor.extract(node.content)
            total += self.concept_graph.index_memory(node, entities, relations)
            processed += 1
            if processed >= limit:
                break
        return total

    def deduplicate(self, threshold: float = 0.85) -> int:
        """合并近似记忆，返回合并对数。"""
        return dedupe.deduplicate(self.store, threshold)

    def expand_context(self, node_id: str, hops: int = 1) -> List:
        """沿概念图扩展，找跨记忆关联。"""
        return self.concept_graph.expand(node_id, hops)

    # ---------- 人格：种子/价值观/关系 ----------

    def learn_seed(self, content: str, triggers: List[str] = None,
                   behavior_bias: str = "", valence: float = 0.0,
                   confidence: float = 0.5, source_event: str = "") -> Seed:
        """从经历沉淀一条倾向种子（事件→种子的闭环入口）。"""
        return self.seeds.add(content, triggers=triggers, behavior_bias=behavior_bias,
                              valence=valence, confidence=confidence,
                              source_event=source_event)

    def note_relationship(self, person_id: str, content: str,
                          source: str = "observation", confidence: float = 0.5) -> None:
        """记录关系记忆（对某人的长期了解）。"""
        self.relationships.note(person_id, content, source, confidence)

    def adopt_autonomous_goals(self) -> List[Goal]:
        """把 active 自主目标采纳进目标系统（意图→目标闭环），返回新采纳的 Goal。"""
        adopted = []
        for ag in self.auto_goals.active():
            goal = self.goals.adopt_autonomous(ag)
            if goal is not None:
                adopted.append(goal)
        return adopted

    def act(self, goal: Optional[Goal] = None) -> Optional[str]:
        """行动：优先采纳自主目标（内在状态涌现的意图），否则需求即时派生 → 拆解 → 执行。

        返回本次行动的结果描述，或 None（无目标/无行动）。
        """
        dom, drive = self.needs.get_dominant_need()
        if goal is None:
            # 优先推进自主目标（意图→目标闭环）；无自主目标再即时派生
            adopted = self.adopt_autonomous_goals()
            if adopted:
                goal = adopted[0]
            else:
                goal = self.goals.derive_goal(dom, drive)
                if goal is None:
                    return None
            # LLM 规划拆解（Plan-and-Execute），失败回退规则
            steps = self.planner.plan(goal, [t.name for t in self.tools.list()])
            self.goals.decompose(goal, steps)
        task = self.goals.next_task(goal)
        if task is None:
            return f"目标「{goal.description[:30]}」已完成"
        # 用工具执行（这里用 read_file 做演示性只读行动）
        # 真正的工具选择应由 LLM 决定，这里提供框架
        result = f"已推进任务：{task.description}"
        self.goals.complete_task(goal, task.id, result)
        self.needs.satisfy(dom, 0.1)  # 行动满足需求
        return result

    def _remember_interaction(self, message: str, answer: str) -> None:
        """智能沉淀对话：只记有实质价值的，避免废话/寒暄膨胀记忆库。

        规则：
        - 太短的寒暄（如"你好""在吗"）不记
        - 只记答案里含实质信息（长度够、非纯寒暄）的交互
        """
        msg_len = len(message.strip())
        ans_len = len(answer.strip())
        # 寒暄过滤：双方都太短，或纯打招呼
        greetings = ("你好", "在吗", "hi", "hello", "谢谢", "好的", "ok")
        if msg_len < 4 or ans_len < 10:
            return
        if message.strip().lower() in greetings:
            return
        self.remember(f"用户问：{message[:80]} → 我答：{answer[:150]}",
                      scope="session")

    def _is_trivial(self, message: str) -> bool:
        """极短/无实质消息（"好""嗯"）跳过记忆检索，省性能。"""
        m = message.strip()
        if len(m) <= 2:  # 1-2 字视为寒暄/确认
            return True
        if len(m) <= 6 and m in ("好的", "嗯嗯", "哦", "是的", "对", "OK", "ok", "谢谢"):
            return True
        return False

    def _should_learn(self, message: str) -> bool:
        """启发式预筛：消息是否含可学习信号（避免每轮都调 LLM 抽取，省 token）。

        当消息涉及偏好/需求/习惯/自述经历/重要事实时才触发学习。
        """
        if len(message.strip()) < 6:  # 太短不值得学
            return False
        signals = ("我", "我们", "喜欢", "讨厌", "偏好", "希望", "习惯",
                   "需要", "不要", "想要", "别", "最", "通常", "一直",
                   "总是", "觉得", "认为", "记得", "计划", "目标",
                   "家人", "朋友", "工作", "学习")
        return any(s in message for s in signals)

    def _recent_history(self) -> List[Dict[str, str]]:
        """注入历史：滚动摘要 + 最近原始消息（清洗结构噪声，防上下文污染）。"""
        out: List[Dict[str, str]] = []
        budget = self.config.history_budget
        used = 0
        # 摘要也必须计入同一预算；此前摘要未计入，可能使历史块超预算。
        summary_msg = None
        if self._compressed_summary:
            summary_msg = {"role": "system",
                           "content": f"[过往摘要] {self._compressed_summary}"}
            used = estimate_tokens(summary_msg["content"])
            if used > budget:
                summary_msg = None
                used = 0
        for m in reversed(self._conversation):
            role = m.get("role", "user")
            raw = m["content"]
            # 多模态消息：文本部分走清洗，图片片段原样保留（图仍喂给模型）
            if isinstance(raw, list):
                txt = _content_text(raw)
                imgs = [p for p in raw if isinstance(p, dict)
                        and p.get("type") == "image_url"]
                clean = clean_history_message(txt, role) if txt else ""
                if clean is None or (txt and is_noise(clean)):
                    clean = ""
                parts = ([{"type": "text", "text": clean}] if clean else []) + imgs
                if not parts:
                    continue
                t = estimate_tokens(clean) if clean else 0
                if used + t > budget:
                    break
                out.append({"role": role, "content": parts})
                used += t
                continue
            content = clean_history_message(raw, role)
            if content is None or is_noise(content):
                continue  # 纯噪声丢弃，不占预算
            t = estimate_tokens(content)
            if used + t > budget:
                break
            out.append({"role": role, "content": content})
            used += t
        if summary_msg:
            out.append(summary_msg)
        return list(reversed(out))

    def _maybe_autosave(self) -> None:
        """自动保存：每 N 轮持久化一次，防止忘记手动 save 导致失忆。"""
        if not self.config.autosave:
            return
        if self._stats["turns"] % self.config.autosave_interval == 0:
            try:
                self.save()
            except Exception as e:  # 自动保存失败不应中断对话
                logger.warning("自动保存失败: %s", e)

    def _maybe_compress(self) -> None:
        """分层记忆：对话超预算时，把旧对话压成滚动摘要，只保留最近 N 条原始。

        参照 Hermes 的滚动摘要压缩：旧对话→LLM归纳成摘要，新对话→原始。
        避免所有对话全量上传，同时保留关键语义。
        """
        total = sum(estimate_tokens(_content_text(m["content"]))
                    for m in self._conversation
                    if isinstance(m.get("content"), str) or isinstance(
                        m.get("content"), list))
        if total <= self.config.compress_threshold:
            return
        import time as _time
        if _time.time() - self._last_compress < 60:  # 1分钟节流
            return
        self._last_compress = _time.time()
        # 保留最近 N 条原始（默认 6 条），其余拿去压摘要
        keep = self._conversation[-6:]
        to_summarize = self._conversation[:-6]
        if not to_summarize:
            return
        convo_text = "\n".join(
            f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:150]}"
            for m in to_summarize[-30:])
        try:
            resp = self.llm.chat([
                {"role": "system", "content":
                 "你是对话压缩器。把下面的历史对话压缩成一段简洁摘要，保留："
                 "用户偏好、关键事实、未完成任务、重要结论。用中文，150字内。"},
                {"role": "user", "content": convo_text},
            ], max_tokens=400)
            summary = (resp.content or "").strip()
            if summary:
                self._compressed_summary = summary
        except Exception:
            logger.warning("对话压缩失败，降级截断（保最近，防无界增长）")
        finally:
            # 无论压缩成功/失败/LLM空返回，都截断保留最近 N 条原始，保证 _conversation 有界
            # （否则 LLM 不可用时 _conversation 无限增长，恰在最需要降级时雪上加霜）
            self._conversation = keep

    def _format_memory(self, hits: List[Tuple[MemoryNode, float, str]],
                       max_chars: int = 800) -> str:
        """格式化记忆注入，带 token 预算（限制注入字符数，省 token）。"""
        if not hits:
            return ""
        lines = []
        budget = 0
        for node, score, why in hits[:5]:
            content = node.content[:120]
            when = fuzzy_time(node.created_at)
            line = f"- [{why}] ({when}) {content}"
            if budget + len(line) > max_chars:
                break
            lines.append(line)
            budget += len(line)
        return "\n".join(lines)

    def stats(self) -> dict:
        return dict(self._stats, **self.state_snapshot())

    def close(self) -> None:
        # 先停后台调度线程再关闭 store，避免线程访问已释放的 SQLite 连接（段错误）
        try:
            self.scheduler.stop()
            self.scheduler.join(timeout=1.0)
        except Exception:
            pass
        self.store.close()
