"""被动用户画像：从对话中被动积累对用户的理解（只做框架，不猜不编）。

哲学：画像不是预设的、也不是一次性 LLM 推断出来的「刻板印象」，
而是随相处被动「长出来」的观测值。每条观测都可溯源、有把握度，避免把推测当事实。

被动积累的维度：
- 沟通风格：话多/话少（消息长度的启发式滚动观测，非语义推断）
- 情绪基调：近期消息的情绪倾向（关键词启发式，滚动均值）
- 话题兴趣：出现频率最高的关键词（纯计数、有界，非预设兴趣表）
- 明确偏好/兴趣：来自学习信号的显式陈述（带来源与置信度）
- 关系深度：随互动次数累积（与 Relationship 解耦，但可对照）

纯启发式、零 LLM 调用，每轮对话后被动 observe 一次，不阻塞主流程。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 常见中文停用词（话题计数用，过滤无意义词）
_STOPWORDS = frozenset({
    "的", "了", "是", "我", "你", "他", "她", "它", "这", "那", "吗", "呢",
    "啊", "吧", "么", "就", "都", "很", "也", "在", "有", "和", "与", "及",
    "不", "要", "会", "能", "想", "说", "做", "个", "一", "上", "下", "中",
    "好", "还", "又", "再", "被", "把", "让", "给", "对", "从", "到", "向",
    "跟", "等", "或", "但", "而", "因", "所", "以", "为", "着", "过", "之",
    "其", "可", "无", "没", "别", "我们", "你们", "他们", "这个", "那个",
    "什么", "怎么", "为什么", "这样", "那样", "现在", "已经", "知道", "觉得",
})

# 情绪倾向关键词（与 autonomous.py 的启发式风格一致）
_MOOD_POS = ("开心", "高兴", "喜欢", "满意", "顺利", "成功", "期待", "哈哈", "赞", "棒")
_MOOD_NEG = ("难过", "生气", "烦", "焦虑", "担心", "失望", "崩溃", "讨厌", "累", "沮丧", "低落")

_TOPIC_WORD_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}")

_MAX_TOPICS = 20      # 话题兴趣条目上限
_MAX_PREFS = 50       # 偏好/兴趣条目上限

# 话题关键词 → 人格维度（用户长期偏好 → 超脑习得表达倾向的映射）
_TOPIC_DIM_MAP = (
    (("探索", "学习", "新技术", "研究", "了解", "好奇", "未知", "尝试"), "openness"),
    (("计划", "坚持", "完成", "认真", "负责", "目标", "执行", "整理", "专注"), "conscientiousness"),
    (("陪伴", "关心", "照顾", "体贴", "温柔", "家人", "朋友", "帮助", "支持"), "agreeableness"),
    (("聚会", "认识", "分享", "热闹", "出去玩", "社交", "一起"), "extraversion"),
)
_MIN_INTERACTIONS_FOR_DERIVE = 10    # 相处至少 10 次才允许升格人格（防早期瞎猜）
_DIM_TOPIC_MIN_COUNT = 3             # 某维度话题累计至少 3 次才算有把握
_MOOD_LOW_THRESHOLD = -0.3           # 情绪基调长期低落阈值
_MOOD_MIN_COUNT = 5                  # 情绪样本至少 5 条


@dataclass
class UserProfile:
    """对单个用户的被动画像。"""
    person_id: str = ""
    name: str = ""
    # 沟通风格：-1(极度精简)..1(话多详细)，累积观测
    style_verbose: float = 0.0
    style_observations: int = 0
    # 情绪基调：滚动均值 -1..1
    mood_mean: float = 0.0
    mood_count: int = 0
    # 话题兴趣：内部计数 + 提取出的高频词
    _topic_counter: Dict[str, int] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    # 明确偏好/兴趣（来自学习信号）
    preferences: List[dict] = field(default_factory=list)
    interests: List[dict] = field(default_factory=list)
    # 关系深度：0-1，随互动累积
    relationship_depth: float = 0.0
    interactions: int = 0
    # 用户为该用户设定的表达风格（任意自然语言，如可爱/冷静/高冷/傲娇...，per-user）
    style: str = ""
    updated_at: float = field(default_factory=time.time)

    def set_style(self, style: str) -> None:
        """为该用户记录表达风格（用户主导，per-user）。"""
        if not isinstance(style, str):
            self.style = ""
            return
        self.style = (style or "").strip()
        self.updated_at = time.time()

    def observe_message(self, text: str) -> None:
        """被动观测一条用户消息，更新启发式画像（纯规则，零 LLM）。"""
        if not isinstance(text, str) or not text.strip():
            return
        t = text.strip()
        # 1. 沟通风格：长度启发式（长消息→话多，短消息→精简）
        n = len(t)
        # 中位长度约 15 字视为中性；用 sigmoid 映射到 -1..1
        verbose_signal = max(-1.0, min(1.0, (n - 15) / 40.0))
        if self.style_observations == 0:
            self.style_verbose = verbose_signal
        else:
            # 指数滑动平均：旧观测缓慢衰减，新观测权重更高
            alpha = 0.3
            self.style_verbose += alpha * (verbose_signal - self.style_verbose)
        self.style_observations += 1
        # 2. 情绪基调：关键词计数 → 本轮倾向 → 滚动均值
        pos = sum(1 for w in _MOOD_POS if w in t)
        neg = sum(1 for w in _MOOD_NEG if w in t)
        if pos or neg:
            mood_signal = max(-1.0, min(1.0, (pos - neg) / (pos + neg)))
            if self.mood_count == 0:
                self.mood_mean = mood_signal
            else:
                self.mood_mean += 0.2 * (mood_signal - self.mood_mean)
            self.mood_count += 1
        # 3. 话题兴趣：切词计数（有界）
        for w in _TOPIC_WORD_RE.findall(t):
            wl = w.lower()
            if wl in _STOPWORDS or len(w) < 2:
                continue
            self._topic_counter[wl] = self._topic_counter.get(wl, 0) + 1
        self._refresh_topics()
        # 4. 关系深度 + 互动次数
        self.interactions += 1
        self.relationship_depth = min(1.0, self.relationship_depth + 0.02)
        self.updated_at = time.time()

    def _refresh_topics(self) -> None:
        """从计数提取高频话题（有界 top-N，纯频率、非预设）。"""
        top = sorted(self._topic_counter.items(), key=lambda kv: kv[1], reverse=True)
        self.topics = [w for w, _ in top[:_MAX_TOPICS]]

    def add_preference(self, key: str, value: str, confidence: float = 0.5,
                       source: str = "inference") -> None:
        """记录一条明确偏好（来自学习信号）。"""
        self.preferences.append({
            "key": key, "value": value, "confidence": confidence,
            "source": source, "time": time.time(),
        })
        if len(self.preferences) > _MAX_PREFS:
            del self.preferences[:-_MAX_PREFS]

    def add_interest(self, value: str, confidence: float = 0.5,
                     source: str = "inference") -> None:
        """记录一条明确兴趣。"""
        self.interests.append({
            "value": value, "confidence": confidence,
            "source": source, "time": time.time(),
        })
        if len(self.interests) > _MAX_PREFS:
            del self.interests[:-_MAX_PREFS]

    def derive_personality(self) -> Dict[str, float]:
        """从长期画像推断超脑应「习得」的人格维度强度（人格升格素材）。

        语义：用户长期稳定表达某类偏好 → 超脑在与该用户相处中习得对应表达倾向。
        只在相处足够(interactions≥阈值)且有把握(话题累计够/情绪样本够)时输出；
        证据不足则返回空 dict（不瞎猜）。返回 {维度: 强度}，强度以 0.5 为中位基线。
        """
        if self.interactions < _MIN_INTERACTIONS_FOR_DERIVE:
            return {}
        derived = {}
        tc = self._topic_counter
        for words, dim in _TOPIC_DIM_MAP:
            # 话题切词产出的是连续汉字/字母串（如"我想探索新技术"），用关键词子串匹配计数
            cnt = sum(c for topic, c in tc.items()
                      if any(w in topic for w in words))
            if cnt >= _DIM_TOPIC_MIN_COUNT:
                derived[dim] = min(0.9, 0.5 + cnt / 200.0)
        # 情绪长期低落 → 习得共情/更情绪化（神经质微升）
        if self.mood_count >= _MOOD_MIN_COUNT and self.mood_mean < _MOOD_LOW_THRESHOLD:
            derived["neuroticism"] = max(derived.get("neuroticism", 0.5),
                                         0.5 + abs(self.mood_mean) * 0.8)
        return derived

    def profile_text(self, max_topics: int = 5) -> str:
        """生成可注入 prompt 的画像描述（无实质内容则空串）。"""
        parts = []
        if self.style_observations >= 2:
            style = ("偏好简洁" if self.style_verbose < -0.3
                     else "表达较详细" if self.style_verbose > 0.3 else "")
            if style:
                parts.append(f"沟通{style}")
        if self.topics:
            parts.append("常聊：" + "、".join(self.topics[:max_topics]))
        if self.preferences:
            prefs = [f"{p['key']}={p['value']}" for p in self.preferences[:3]]
            parts.append("偏好：" + "、".join(prefs))
        if self.interests:
            parts.append("兴趣：" + "、".join(
                [i["value"] for i in self.interests[:3]]))
        return "对用户：" + "；".join(parts) if parts else ""

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id, "name": self.name,
            "style_verbose": self.style_verbose,
            "style_observations": self.style_observations,
            "mood_mean": self.mood_mean, "mood_count": self.mood_count,
            "topic_counter": dict(self._topic_counter),
            "topics": self.topics,
            "preferences": self.preferences, "interests": self.interests,
            "relationship_depth": self.relationship_depth,
            "interactions": self.interactions,
            "style": self.style,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        if not isinstance(d, dict):
            d = {}
        p = cls(
            person_id=d.get("person_id", ""), name=d.get("name", ""),
            style_verbose=d.get("style_verbose", 0.0),
            style_observations=d.get("style_observations", 0),
            mood_mean=d.get("mood_mean", 0.0),
            mood_count=d.get("mood_count", 0),
            preferences=d.get("preferences", []) or [],
            interests=d.get("interests", []) or [],
            relationship_depth=d.get("relationship_depth", 0.0),
            interactions=d.get("interactions", 0),
            style=d.get("style", "") or "",
            updated_at=d.get("updated_at", 0.0),
        )
        tc = d.get("topic_counter", {})
        p._topic_counter = ({k: int(v) for k, v in tc.items()}
                            if isinstance(tc, dict) else {})
        p._refresh_topics()
        return p


class UserProfileGraph:
    """管理多个用户的被动画像。"""

    def __init__(self) -> None:
        self._profiles: Dict[str, UserProfile] = {}

    def get_or_create(self, person_id: str, name: str = "") -> UserProfile:
        if person_id not in self._profiles:
            self._profiles[person_id] = UserProfile(
                person_id=person_id, name=name)
        elif name and not self._profiles[person_id].name:
            self._profiles[person_id].name = name
        return self._profiles[person_id]

    def observe(self, person_id: str, text: str, name: str = "") -> UserProfile:
        """被动观测一条消息（零 LLM，纯启发式）。"""
        p = self.get_or_create(person_id, name)
        p.observe_message(text)
        return p

    def get(self, person_id: str) -> Optional[UserProfile]:
        return self._profiles.get(person_id)

    def all(self) -> List[UserProfile]:
        return list(self._profiles.values())

    def aggregate_personality(self) -> Dict[str, float]:
        """汇总所有用户画像的习得人格：取各维度最高强度。

        超脑从所有关系中学习：只要任一用户长期表现出某偏好，即习得对应维度。
        """
        agg: Dict[str, float] = {}
        for p in self._profiles.values():
            for dim, s in p.derive_personality().items():
                agg[dim] = max(agg.get(dim, 0.5), s)
        return agg

    def summary(self) -> List[dict]:
        """可观察快照：所有画像的紧凑摘要。"""
        return [{"person_id": p.person_id, "name": p.name,
                 "interactions": p.interactions,
                 "relationship_depth": round(p.relationship_depth, 3),
                 "topics": p.topics[:5]} for p in self._profiles.values()]

    def to_dict(self) -> dict:
        return {"profiles": {k: v.to_dict() for k, v in self._profiles.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfileGraph":
        g = cls()
        if isinstance(d, dict):
            g._profiles = {k: UserProfile.from_dict(v)
                           for k, v in d.get("profiles", {}).items()}
        return g
