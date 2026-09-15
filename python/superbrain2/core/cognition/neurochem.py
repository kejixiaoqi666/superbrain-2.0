"""神经化学层：多通道情绪动量（多巴胺/血清素/皮质醇）。

真实情绪有化学惯性——多巴胺不会瞬间归零，情绪会残留、发酵、滞后。
现有 EmotionGradient 是"即时计算+单值平滑"，这一层补上"化学动量"：

- 多巴胺 dopamine：奖赏/期待，来得快去得也快，驱动"想要"
- 血清素 serotonin：满足/安宁，慢变，长期基调，驱动"满足"
- 皮质醇 cortisol：压力/威胁，急性响应，残留久，驱动"焦虑"

关键：各递质有独立的 产生速率 / 清除速率（半衰期），
所以情绪不会随输入瞬间跳变，而是有滞后的"发酵感"。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NeuroChemState:
    dopamine: float = 0.5      # 0..1 奖赏/期待
    serotonin: float = 0.5     # 0..1 满足/安宁（长期基调）
    cortisol: float = 0.2      # 0..1 压力/威胁

    def to_dict(self) -> dict:
        return {
            "dopamine": round(self.dopamine, 3),
            "serotonin": round(self.serotonin, 3),
            "cortisol": round(self.cortisol, 3),
        }

    @property
    def mood_valence(self) -> float:
        """由递质合成的情感价：多巴胺+血清素 升愉悦，皮质醇降愉悦。"""
        return max(-1.0, min(1.0, (self.dopamine + self.serotonin) / 2.0 - self.cortisol))


class NeuroChemistry:
    """神经化学引擎：递质有产生/清除速率，产生情绪动量。"""

    def __init__(self,
                 da_half_life: float = 0.5,   # 多巴胺半衰期（小时，短）
                 se_half_life: float = 6.0,   # 血清素半衰期（小时，长）
                 co_half_life: float = 2.0,   # 皮质醇半衰期（小时）
                 ) -> None:
        self.state = NeuroChemState()
        # 半衰期 → 清除速率 k = ln2 / half_life
        import math
        self._k_da = math.log(2) / max(0.1, da_half_life)
        self._k_se = math.log(2) / max(0.1, se_half_life)
        self._k_co = math.log(2) / max(0.1, co_half_life)

    def release(self, dopamine: float = 0.0, serotonin: float = 0.0,
                cortisol: float = 0.0) -> None:
        """事件触发递质释放（正/负事件）。"""
        self.state.dopamine = min(1.0, self.state.dopamine + dopamine)
        self.state.serotonin = min(1.0, self.state.serotonin + serotonin)
        self.state.cortisol = min(1.0, self.state.cortisol + cortisol)

    def tick(self, dt_hours: float = 0.05) -> None:
        """递质按半衰期清除（产生情绪残留/滞后）。"""
        import math
        self.state.dopamine *= math.exp(-self._k_da * dt_hours)
        self.state.serotonin *= math.exp(-self._k_se * dt_hours)
        self.state.cortisol *= math.exp(-self._k_co * dt_hours)

    def observe(self, valence: float, arousal: float) -> None:
        """从情绪状态更新递质（情绪→化学的映射）。

        valence>0 → 多巴胺/血清素释放；valence<0 或 arousal高 → 皮质醇。
        """
        if valence > 0:
            self.release(dopamine=valence * 0.3, serotonin=valence * 0.15)
        else:
            self.release(cortisol=-valence * 0.4)
        if arousal > 0.7:  # 高唤醒 → 压力
            self.release(cortisol=(arousal - 0.7) * 0.5)

    def momentum(self) -> float:
        """情绪动量：当前递质合成的"情绪底色"，滞后于即时情绪。"""
        return self.state.mood_valence

    def to_dict(self) -> dict:
        return {"state": self.state.to_dict(),
                "half_lives": {"da": round(0.693/self._k_da, 2),
                               "se": round(0.693/self._k_se, 2),
                               "co": round(0.693/self._k_co, 2)}}

    @classmethod
    def from_dict(cls, d: dict) -> "NeuroChemistry":
        nc = cls()
        s = d.get("state", {})
        nc.state.dopamine = s.get("dopamine", 0.5)
        nc.state.serotonin = s.get("serotonin", 0.5)
        nc.state.cortisol = s.get("cortisol", 0.2)
        return nc
