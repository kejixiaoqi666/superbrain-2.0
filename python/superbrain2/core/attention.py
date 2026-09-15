"""注意力/显著性机制（人脑选择性注意的工程化）。

人脑不是对所有刺激平等处理，而是按显著性(salience)选择焦点。
超脑已有 GWT 竞争广播，但缺"注意力焦点"：哪个维度当前最该优先处理。

显著性来源：
- 情绪显著性：当前情绪 valence/arousal 偏离中性 → 高显著
- 需求显著性：主导需求的驱动力 → 高显著
- 记忆显著性：检索到的记忆数量/强度 → 高显著
- 新颖性显著性：当前输入的新奇程度 → 高显著

输出一个"注意力焦点"描述 + 各维度的显著性权重，
供上层决定优先关注什么。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AttentionFocus:
    dominant: str = ""            # 当前注意力焦点（哪个维度）
    weights: Dict[str, float] = field(default_factory=dict)  # 各维度显著性权重
    description: str = ""          # 焦点的自然语言描述

    def to_dict(self) -> dict:
        return {"dominant": self.dominant, "weights": dict(self.weights),
                "description": self.description}


class AttentionEngine:
    """计算当前输入的注意力焦点（各维度显著性加权竞争）。"""

    def __init__(self) -> None:
        # 维度权重（可调）
        self.dim_weights = {
            "emotion": 0.30,   # 情绪显著性
            "need": 0.30,      # 需求显著性
            "memory": 0.20,    # 记忆显著性
            "novelty": 0.20,   # 新颖性
        }
        self._novelty_seen: set = set()  # 记录见过的内容指纹，计算新颖性
        self._novelty_order: deque = deque()  # 指纹插入顺序，手动裁剪配合 seen 同步移除
        self.history = deque(maxlen=50)   # 有上限，防无限增长

    def focus(self, valence: float, arousal: float,
              dominant_need: str, need_drive: float,
              memory_hits: int, input_text: str) -> AttentionFocus:
        """计算注意力焦点。

        参数来自 agent 当前状态：
        - valence/arousal: 情绪
        - dominant_need/need_drive: 主导需求
        - memory_hits: 检索到记忆条数
        - input_text: 当前输入
        """
        # 情绪显著性：偏离中性
        emo_sig = min(1.0, abs(valence) + arousal)

        # 需求显著性：驱动力
        need_sig = min(1.0, need_drive)

        # 记忆显著性：有记忆关联
        mem_sig = min(1.0, memory_hits / 5.0)

        # 新颖性：没见过 → 高（有界：只记最近 2000 个指纹，防 set 无限增长）
        import hashlib
        fp = hashlib.blake2b(input_text.encode("utf-8"), digest_size=8).hexdigest()
        novel = 1.0 if fp not in self._novelty_seen else 0.2
        self._novelty_seen.add(fp)
        self._novelty_order.append(fp)
        if len(self._novelty_order) > 2000:
            self._novelty_seen.discard(self._novelty_order.popleft())

        # 加权
        scores = {
            "emotion": self.dim_weights["emotion"] * emo_sig,
            "need": self.dim_weights["need"] * need_sig,
            "memory": self.dim_weights["memory"] * mem_sig,
            "novelty": self.dim_weights["novelty"] * novel,
        }
        dominant = max(scores, key=scores.get)

        # 描述
        desc_map = {
            "emotion": f"情绪显著（valence={valence:.2f}, arousal={arousal:.2f}）",
            "need": f"需求驱动强（{dominant_need}, drive={need_drive:.2f}）",
            "memory": f"记忆关联多（{memory_hits}条）",
            "novelty": "内容新颖（首次出现）",
        }
        focus = AttentionFocus(dominant=dominant, weights=scores,
                               description=desc_map[dominant])
        self.history.append(focus)
        return focus


def attention_block(focus: AttentionFocus) -> str:
    """把注意力焦点格式化为注入 prompt 的文本（省 token 的精简版）。"""
    return f"[注意力] 焦点={focus.dominant}（{focus.description}）"
