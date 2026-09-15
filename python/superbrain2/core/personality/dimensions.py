"""人格维度显式化：可度量的人格维度框架（只提供骨架，不做人格预设）。

哲学：内核只做框架不做设定。这里只定义「有哪些维度可观测、如何被推动」，
维度初值全部中性(0.5)、把握度 0——人格是相处/经历/用户反馈中「长出来」的，
不是写死的预设。采用大五人格(Big Five)作为中性的维度骨架（名字只是观测轴，
不代表任何预设倾向）。

每个维度是一条 Trait：
- value：0-1，0.5=中性（无预设）
- confidence：对当前值的把握度（证据越多越确定）
- source：default | experience | explicit | inference（来源，区分「长出来」还是「被设定」）
- observations：有界证据日志（防无限增长）

两条推进路径：
- observe()：被动观测，一次经历轻微推动某维度（渐近成长、有饱和、有界移动）
- set()：显式设定（用户反馈「你应该更X」），来源标记为 explicit
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 大五人格维度（中性骨架）
BIG_FIVE = ("openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism")

BIG_FIVE_CN = {
    "openness": "开放性",
    "conscientiousness": "尽责性",
    "extraversion": "外向性",
    "agreeableness": "宜人性",
    "neuroticism": "神经质",
}

_MAX_OBSERVATIONS = 50  # 每条维度证据日志上限（防无限增长）

# 从相处经历被动观察人格的启发式规则：(触发词, 维度, 幅度)
# 语义：超脑与用户相处，用户的风格/反馈塑造超脑的人格（人格是相处出来的，非预设）。
# 幅度小 + observe() 有界移动与饱和，长期收敛到稳定人格。
_INTERACTION_RULES = (
    (("探索", "新", "试试", "好奇", "了解", "学习"), "openness", 0.02),
    (("谢谢", "关心", "体贴", "温柔", "你真好", "抱抱"), "agreeableness", 0.02),
    (("聊", "分享", "互动", "热闹", "说说"), "extraversion", 0.02),
    (("坚持", "认真", "负责", "完成", "专注"), "conscientiousness", 0.02),
)
# 用户情绪低落的信号词（共情吸收 → 神经质轻微上升）
_LOW_MOOD_SIGNAL = ("难过", "低落", "焦虑", "担心", "崩溃", "累", "烦")

# 人格成长通道的权威优先级（低→高）。冲突消解核心：explicit 是唯一"硬锁定"，
# 四个通道(observe/reflect/set_from_user/set)中，凡 source==explicit 的维度，
# 被动通道一律不修改其 value/confidence，只有用户显式 set() 才能改。
_CHANNEL_PRIORITY = ("experience", "reflection", "derived", "explicit")

# 一句话风格预设：自然语言人格风格 → 大五维度组合（apply_style 用，explicit 显式设定）
_STYLE_PRESETS = {
    "可爱": {"extraversion": 0.75, "agreeableness": 0.8, "openness": 0.7},
    "温柔": {"agreeableness": 0.85, "neuroticism": 0.35},
    "活泼": {"extraversion": 0.8, "openness": 0.7},
    "冷静": {"conscientiousness": 0.75, "extraversion": 0.45, "neuroticism": 0.3},
    "沉稳": {"conscientiousness": 0.8, "neuroticism": 0.3},
    "高冷": {"agreeableness": 0.3, "extraversion": 0.3, "neuroticism": 0.35},
    "理智": {"neuroticism": 0.25, "openness": 0.6, "conscientiousness": 0.7},
    "热情": {"extraversion": 0.85, "agreeableness": 0.7, "openness": 0.75},
    "随和": {"agreeableness": 0.8, "neuroticism": 0.4, "extraversion": 0.5},
    "严肃": {"conscientiousness": 0.8, "extraversion": 0.35, "agreeableness": 0.45},
}


@dataclass
class Trait:
    """单条人格维度。value 0-1（0.5=中性），confidence 对当前值的把握度。
    source 通道：default | experience(被动观察) | reflection(主动内化)
              | derived(画像升格) | explicit(用户显式设定)
    冲突消解优先级(高→低)：explicit > derived > reflection > experience；
    其中 explicit 是唯一硬锁定——被动通道一律不改其 value/confidence。"""
    name: str = ""
    value: float = 0.5
    confidence: float = 0.0
    source: str = "default"  # default | experience | explicit | inference
    observations: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "value": self.value,
            "confidence": self.confidence, "source": self.source,
            "observations": self.observations, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trait":
        if not isinstance(d, dict):
            return cls()
        return cls(
            name=d.get("name", ""),
            value=max(0.0, min(1.0, d.get("value", 0.5))),
            confidence=max(0.0, min(1.0, d.get("confidence", 0.0))),
            source=d.get("source", "default"),
            observations=d.get("observations", []) or [],
            updated_at=d.get("updated_at", 0.0),
        )


class PersonalityDimensions:
    """人格维度框架：只定义骨架，人格从经历中长出来，不做预设。"""

    def __init__(self, dimensions=None, autonomy_mode: str = "autonomous") -> None:
        names = dimensions or BIG_FIVE
        self._traits: Dict[str, Trait] = {
            n: Trait(name=n, value=0.5, confidence=0.0, source="default")
            for n in names
        }
        # 进化模式开关：autonomous(默认,四通道自主演化) / guided(用户主导,自动通道全停)
        self.autonomy_mode = autonomy_mode if autonomy_mode in ("autonomous", "guided") else "autonomous"
        # 用户最近设定的表达风格（任意自然语言，如可爱/冷静/高冷/傲娇...，供上层注入表达）
        self.custom_style: str = ""

    def set_mode(self, mode: str) -> bool:
        """设定进化模式：autonomous(自主演化) 或 guided(用户主导,自动通道全停)。"""
        if mode in ("autonomous", "guided"):
            self.autonomy_mode = mode
            return True
        return False

    @property
    def mode(self) -> str:
        return self.autonomy_mode

    def get(self, name: str) -> Optional[Trait]:
        return self._traits.get(name)

    def set(self, name: str, value: float, source: str = "explicit",
            confidence: float = 0.8, note: str = "") -> Optional[Trait]:
        """显式设定某维度（用户反馈「你应该更X」），来源标记 explicit。"""
        t = self._traits.get(name)
        if t is None:
            return None
        if not isinstance(value, (int, float)):
            return None  # 防御脏输入：维度值必须是数值
        t.value = max(0.0, min(1.0, value))
        t.confidence = max(0.0, min(1.0, confidence))
        t.source = source
        t.updated_at = time.time()
        if note:
            self._push_obs(t, note)
        return t

    def apply_style(self, style: str) -> List[str]:
        """一句话风格设定（任意自然语言，如可爱/冷静/高冷/傲娇...，不限于预置清单）。

        - 预置风格 → 映射为大五维度组合显式设定(explicit) + 自动切到 guided；
        - 自定义风格（如傲娇/干练/古灵精怪...）→ 不强套维度，仅记录为表达风格
          `custom_style` + 切到 guided（完全听用户，核心库不做设定）；
        - 返回被设定维度的列表（预置返回维度名；自定义返回 [] 但 custom_style 已记录）。
        """
        if not isinstance(style, str):
            return []  # 防御脏输入：风格必须是字符串
        style = (style or "").strip()
        if not style:
            return []
        self.autonomy_mode = "guided"      # 一句风格指令 = 用户主导，自动通道全停
        self.custom_style = style          # 始终记录用户的风格指令（含自定义）
        preset = _STYLE_PRESETS.get(style)
        if not preset:
            return []                      # 自定义风格：不硬套维度，仅记录指令
        applied = []
        for dim, val in preset.items():
            t = self.set(dim, val, source="explicit", confidence=0.9,
                         note=f"用户设置风格:{style}")
            if t is not None:
                applied.append(dim)
        return applied

    @property
    def available_styles(self) -> List[str]:
        """常见风格便捷映射清单（仅作示例，不锁定——任意自然语言风格都可设）。"""
        return list(_STYLE_PRESETS.keys())

    @property
    def style_text(self) -> str:
        """当前表达风格文本（custom_style，供上层注入 prompt/humanize）。"""
        return self.custom_style

    def observe(self, name: str, delta: float, note: str = "",
                source: str = "experience",
                confidence: float = 0.3) -> Optional[Trait]:
        """被动观测：一次经历轻微推动某维度（渐近成长、有饱和、有界移动）。"""
        t = self._traits.get(name)
        if t is None:
            return None
        if self.autonomy_mode != "autonomous":
            return None  # 用户主导(guided)：自动通道全停，不被动改变人格
        if t.source == "explicit":
            return None  # 用户显式锁定(explicit>experience)：被动观察不修改 value
        # 有界移动：幅度受 clamp，且越接近极值越难再推（饱和）
        room = (1.0 - t.value) if delta >= 0 else t.value
        t.value = max(0.0, min(1.0, t.value + delta * max(0.05, room)))
        # 把握度：更多证据 → 更确定，但有上限
        t.confidence = min(1.0, t.confidence + confidence * (1.0 - t.confidence))
        if t.source == "default" and source != "default":
            t.source = source
        t.updated_at = time.time()
        if note:
            self._push_obs(t, note)
        return t

    def _push_obs(self, t: Trait, note: str) -> None:
        t.observations.append(note)
        if len(t.observations) > _MAX_OBSERVATIONS:
            del t.observations[:-_MAX_OBSERVATIONS]

    def profile(self) -> Dict[str, dict]:
        """全部维度的当前画像（含把握度与来源）。"""
        return {n: {"value": round(t.value, 3),
                    "confidence": round(t.confidence, 3),
                    "source": t.source}
                for n, t in self._traits.items()}

    def salient(self, threshold: float = 0.15) -> List[Trait]:
        """偏离中性且有一定把握的显著维度（用于 prompt 注入）。"""
        return [t for t in self._traits.values()
                if abs(t.value - 0.5) >= threshold and t.confidence > 0.0]

    def profile_text(self, threshold: float = 0.15) -> str:
        """生成可注入 prompt 的人格描述（只列显著维度，无则空串）。"""
        parts = []
        for t in self.salient(threshold):
            direction = "高" if t.value > 0.5 else "低"
            parts.append(f"{BIG_FIVE_CN.get(t.name, t.name)}{direction}({t.value:.2f})")
        return "人格：" + "、".join(parts) if parts else ""

    def all(self) -> List[Trait]:
        return list(self._traits.values())

    def observe_interaction(self, text) -> List[str]:
        """从一次相处经历被动观察人格（用户的风格/反馈塑造超脑，非预设）。

        返回本次被微调的维度名列表。幅度小 + observe() 有界移动与饱和，
        长期收敛到稳定人格（人格自主演化闭环的被动输入侧）。
        """
        if not isinstance(text, str) or not text.strip():
            return []
        touched = []
        for words, dim, delta in _INTERACTION_RULES:
            if any(w in text for w in words):
                if self.observe(dim, delta, note=f"互动触发:{dim}",
                                source="experience") is not None:
                    touched.append(dim)
        # 共情吸收：用户情绪低落 → 神经质轻微上升（被负面情绪感染）
        if any(w in text for w in _LOW_MOOD_SIGNAL):
            if self.observe("neuroticism", 0.02, note="共情吸收负面情绪",
                            source="experience") is not None:
                touched.append("neuroticism")
        return touched

    def reflect(self, features: Optional[Dict[str, float]] = None) -> List[str]:
        """人格主动自省（主动内化，区别于被动观察）。

        回顾近期稳定表现出的行为强度（features），与当前人格对照：
        - 言行差距大(>0.25) → 朝近期行为微移 + 把握度微增（内化新自我认知）
        - 显著但近期无表现的维度 → 把握度轻微下调（自我怀疑，人格尚未落地）

        与 observe(被动吸收) + set(显式反馈) 并列为第三条成长通道。返回被调整维度名。
        """
        if not features:
            return []
        if self.autonomy_mode != "autonomous":
            return []  # 用户主导(guided)：不主动内化人格
        touched = []
        recent = set()
        for dim, t in features.items():
            if dim not in self._traits:
                continue
            recent.add(dim)   # 无论值是否有效，该维度视为"近期被观察"（防误触发自我怀疑）
            if not isinstance(t, (int, float)):
                continue
            trait = self._traits[dim]
            if trait.source == "explicit":
                continue  # 用户显式锁定(explicit>reflection)：不被短期行为内化覆盖
            if abs(trait.value - t) > 0.25:
                # 言行不一 → 朝近期稳定行为内化（小步，不一步到位）
                trait.value = max(0.0, min(1.0, trait.value + (t - trait.value) * 0.1))
                trait.confidence = min(1.0, trait.confidence + 0.05)
                trait.source = "reflection"
                trait.updated_at = time.time()
                touched.append(dim)
        # 显著但近期无稳定表现 → 把握度轻微下调（自我怀疑）
        for dim, tr in self._traits.items():
            if tr.source == "explicit":
                continue  # 用户显式维度的把握度不因近期无表现而下调
            if dim not in recent and abs(tr.value - 0.5) > 0.2 and tr.confidence > 0.5:
                tr.confidence = max(0.0, tr.confidence - 0.05)
                tr.updated_at = time.time()
                touched.append(dim)
        return touched

    def set_from_user(self, derived: Dict[str, float]) -> List[str]:
        """从用户长期画像升格人格（被动观察 → 显式内化的自动闭环）。

        - 只提升超过当前值的维度，不做回退；
        - 不覆盖用户显式 set() 的维度（source=explicit 优先）；
        - 升格后 source 标记为 derived，把握度提升。
        返回被升格的维度名列表。
        """
        if not isinstance(derived, dict) or not derived:
            return []
        if self.autonomy_mode != "autonomous":
            return []  # 用户主导(guided)：不自动从画像升格人格
        lifted = []
        for dim, strength in derived.items():
            if dim not in self._traits or not isinstance(strength, (int, float)):
                continue
            t = self._traits[dim]
            if t.source == "explicit":
                continue  # 用户显式设定优先，不覆盖
            if strength > t.value:
                t.value = max(0.0, min(1.0, strength))
                t.confidence = min(1.0, t.confidence + 0.4)
                t.source = "derived"
                t.updated_at = time.time()
                self._push_obs(t, "从用户长期画像习得")
                lifted.append(dim)
        return lifted

    def to_dict(self) -> dict:
        return {"traits": {n: t.to_dict() for n, t in self._traits.items()},
                "autonomy_mode": self.autonomy_mode,
                "custom_style": self.custom_style}

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalityDimensions":
        pd = cls()
        if not isinstance(d, dict):
            return pd
        # 恢复进化模式开关（脏值回退 autonomous）
        mode = d.get("autonomy_mode")
        if mode in ("autonomous", "guided"):
            pd.autonomy_mode = mode
        style = d.get("custom_style")
        if isinstance(style, str):
            pd.custom_style = style
        traits = d.get("traits", {})
        if isinstance(traits, dict):
            # 合并而非替换：只覆盖已知维度的值，缺失/脏维度保留中性默认，不丢维度
            for name, td in traits.items():
                if name in pd._traits:
                    loaded = Trait.from_dict(td)
                    loaded.name = name
                    pd._traits[name] = loaded
        return pd
