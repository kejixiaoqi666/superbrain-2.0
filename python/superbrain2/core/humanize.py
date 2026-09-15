"""基于情绪、亲密度、人格维度与用户画像的规则化人性化表达。"""
import random
from .cognition.emotion import EmotionalState


def _trait_value(personality, name, default=0.5):
    """从人格维度对象安全取某维度值（防御脏输入/None/无 get 方法）。"""
    if personality is None:
        return default
    get = getattr(personality, "get", None)
    if not callable(get):
        return default
    try:
        t = get(name)
        v = getattr(t, "value", None) if t is not None else None
        return v if isinstance(v, (int, float)) else default
    except Exception:
        return default


def _user_mood(user_profile, default=0.0):
    """从用户画像安全取对方情绪基调（mood_mean，-1..1）。"""
    if user_profile is None:
        return default
    m = getattr(user_profile, "mood_mean", None)
    return m if isinstance(m, (int, float)) else default


class HumanizeEngine:
    """不联网、不调用模型的轻量表达包装器。"""

    def __init__(self, cute_probability=0.5):
        self.cute_probability = max(0.0, min(1.0, cute_probability))

    def _choose_affix(self, emotion_state, familiarity, extraversion=0.5):
        """按情绪调制卖萌概率，返回确定的词缀或空串。cute_probability=0 时彻底不卖萌。

        高外向性(extraversion>0.65)让表达更活泼：卖萌概率上调。
        """
        if self.cute_probability <= 0:
            return ""
        lively_boost = 0.2 if extraversion > 0.65 else 0.0
        # 熟悉度加成：越亲近越容易自然卖萌
        probability = (self.cute_probability
                       + lively_boost
                       + max(0.0, emotion_state.arousal - .5) * .4
                       + max(0.0, familiarity - .5) * .3)
        if random.random() >= min(1.0, probability):
            return ""
        return ("喵", "啦", "嗷", "~")[int(emotion_state.arousal * 10) % 4]

    def humanize(self, text, emotion_state: EmotionalState, familiarity=0.5,
                 nickname="", personality=None, user_profile=None):
        if not isinstance(familiarity, (int, float)):
            familiarity = 0.5  # 脏熟悉度防御，避免 None>=.7 崩
        if text is None:
            text = ""
        result = str(text)

        extraversion = _trait_value(personality, "extraversion")
        agreeableness = _trait_value(personality, "agreeableness")
        neuroticism = _trait_value(personality, "neuroticism")
        other_mood = _user_mood(user_profile)

        # 高宜人性 → 更温柔（亲昵门槛降低）；对方近期低落 → 也倾向温柔关心
        gentle = agreeableness > 0.65 or other_mood < -0.3
        # 高神经质 → 更情绪化（情绪低落阈值放宽，更容易流露）
        low = emotion_state.valence < (-0.2 if neuroticism > 0.65 else -0.4)

        if familiarity >= .7:
            prefix = (nickname + "~") if nickname else "哎哎哎"
            result = prefix + result
            if not low:
                result += "贴贴"  # 低落时不叠"贴贴"，避免"贴贴+呜呜"语义矛盾
        elif gentle and familiarity >= .4:
            # 温柔：宜人性高/对方低落时，即使没那么熟也加轻亲昵
            if not result.endswith("~"):
                result += "~"

        if emotion_state.valence > .4:
            result += "，嘿嘿"
        elif low:
            result += "，呜呜"
            if familiarity >= .6 or gentle:
                result += "，你能不能关心我一下"
        elif not result.endswith(("啦", "呀", "哦")):
            result += "啦"

        if emotion_state.arousal > .7:
            result += "！！真的"

        # 对方近期情绪低落 → 追加关心（画像驱动的表达）
        if other_mood < -0.3 and "有你在" not in result:
            result += "，你要好好的"

        affix = self._choose_affix(emotion_state, familiarity, extraversion)
        if affix and not result.endswith(affix):
            result += affix
        return result

    def to_dict(self) -> dict:
        return {"cute_probability": self.cute_probability}

    @classmethod
    def from_dict(cls, d: dict) -> "HumanizeEngine":
        return cls(cute_probability=d.get("cute_probability", 0.5))
