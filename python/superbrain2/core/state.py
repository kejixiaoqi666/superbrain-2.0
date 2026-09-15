"""状态持久化：认知状态（需求/情绪/技能/目标）跨会话保存。

记忆持久了，但「自我」也要有连续性——需求值、情绪值、技能熟练度、目标
这些内在状态重启后必须恢复，否则有记忆但没自我连续性。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .cognition.needs import NeedDriveSystem, NeedType
from .cognition.emotion import EmotionGradient
from .memory.distill import Distiller
from .goals import GoalManager, Goal


def default_state_path() -> str:
    return os.environ.get("SUPERBRAIN_STATE") or str(
        Path.home() / ".superbrain" / "state.json")


def save_state(path: str, needs: NeedDriveSystem, emotion: EmotionGradient,
               distiller: Distiller, goals: GoalManager,
               seeds=None, values=None, relationships=None, self_model=None,
               identity=None, neurochem=None, metacog=None,
               personality=None, auto_goals=None, user_profiles=None,
               session=None) -> None:
    """序列化认知+人格状态到 JSON 文件。"""
    state = _snapshot(needs, emotion, distiller, goals, seeds, values, relationships, self_model, identity, neurochem, metacog, personality, auto_goals, user_profiles, session)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    # 原子写：先写临时文件再 rename，避免写一半崩溃损坏原文件
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _snapshot(needs, emotion, distiller, goals,
              seeds=None, values=None, relationships=None, self_model=None,
              identity=None, neurochem=None, metacog=None,
              personality=None, auto_goals=None, user_profiles=None,
              session=None) -> dict:
    """导出认知+人格状态快照 dict（人格层可选）。session 为对话历史等会话态。"""
    state = {
        "needs": {nt.value: {
            "current": needs.needs[nt].current_level,
            "target": needs.needs[nt].target_level,
        } for nt in needs.needs},
        "emotion": emotion.state.to_dict(),
        "distiller": distiller.to_dict(),
        "goals": [g.to_dict() for g in goals._goals],
    }
    if seeds is not None:
        state["seeds"] = seeds.to_dict()
    if values is not None:
        state["values"] = values.to_dict()
    if relationships is not None:
        state["relationships"] = relationships.to_dict()
    if self_model is not None:
        state["self_model"] = self_model.to_dict()
    if identity is not None:
        state["identity"] = identity.to_dict()
    if neurochem is not None:
        state["neurochem"] = neurochem.to_dict()
    if metacog is not None:
        state["metacognition"] = metacog.to_dict()
    if personality is not None:
        state["personality"] = personality.to_dict()
    if auto_goals is not None:
        state["auto_goals"] = auto_goals.to_dict()
    if user_profiles is not None:
        state["user_profiles"] = user_profiles.to_dict()
    if session is not None:
        state["session"] = session
    return state


def save_state_to_store(store, needs, emotion, distiller, goals,
                        seeds=None, values=None, relationships=None, self_model=None,
                        identity=None, neurochem=None, metacog=None,
                        personality=None, auto_goals=None, user_profiles=None,
                        session=None) -> None:
    """把认知+人格状态存入 MemoryStore 的 meta 表（统一持久化：一个 .db = 整个大脑）。"""
    store._set_meta("cognition_state", json.dumps(
        _snapshot(needs, emotion, distiller, goals, seeds, values, relationships, self_model, identity, neurochem, metacog, personality, auto_goals, user_profiles, session),
        ensure_ascii=False))


def load_state_from_store(store, needs, emotion, distiller, goals,
                          seeds=None, values=None, relationships=None, self_model=None,
                          identity=None, neurochem=None, metacog=None,
                          personality=None, auto_goals=None, user_profiles=None,
                          session=None) -> bool:
    """从 MemoryStore 的 meta 表恢复认知+人格状态。"""
    raw = store._get_meta("cognition_state")
    if not raw:
        return False
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(state, dict):
        return False   # meta 中非 dict → 容错为加载失败
    return _restore(state, needs, emotion, distiller, goals, seeds, values, relationships, self_model, identity, neurochem, metacog, personality, auto_goals, user_profiles, session)


def _restore(state: dict, needs, emotion, distiller, goals,
             seeds=None, values=None, relationships=None, self_model=None,
             identity=None, neurochem=None, metacog=None,
             personality=None, auto_goals=None, user_profiles=None,
             session=None) -> bool:
    """从状态 dict 恢复认知+人格状态。"""
    # 恢复需求
    for nt_str, val in state.get("needs", {}).items():
        try:
            nt = NeedType(nt_str)
        except ValueError:
            continue
        if nt in needs.needs:
            needs.needs[nt].current_level = val.get("current", 0.5)
            needs.needs[nt].target_level = val.get("target", 0.8)

    # 恢复情绪（全部 8 个维度）
    emo = state.get("emotion", {})
    if emo:
        emotion.state.valence = emo.get("valence", 0.0)
        emotion.state.arousal = emo.get("arousal", 0.5)
        emotion.state.dominance = emo.get("dominance", 0.5)
        emotion.state.confidence = emo.get("confidence", 0.5)
        emotion.state.certainty = emo.get("certainty", 0.5)
        emotion.state.safety = emo.get("safety", 0.5)
        emotion.state.fatigue = emo.get("fatigue", 0.0)
        emotion.state.attachment = emo.get("attachment", 0.5)

    # 恢复技能/经验
    dist_state = state.get("distiller", {})
    if dist_state:
        loaded = Distiller.from_dict(dist_state)
        distiller._experiences = loaded._experiences
        distiller._skills = loaded._skills

    # 恢复目标
    for gd in state.get("goals", []):
        g = Goal(
            id=gd.get("id", ""), description=gd.get("description", ""),
            driven_by=gd.get("driven_by", ""), priority=gd.get("priority", 0.0),
            status=gd.get("status", "pending"), created_at=gd.get("created_at", 0.0),
            source=gd.get("source", "need"), source_id=gd.get("source_id", ""),
        )
        for td in gd.get("tasks", []):
            from .goals import Task
            g.tasks.append(Task(
                id=td.get("id", ""), description=td.get("description", ""),
                status=td.get("status", "pending"), result=td.get("result", ""),
                dependencies=td.get("dependencies", []),
            ))
        goals._goals.append(g)

    # 恢复人格层（可选）
    if seeds is not None and "seeds" in state:
        from .personality.seed import SeedBank
        loaded_seeds = SeedBank.from_dict(state["seeds"])
        seeds._seeds = loaded_seeds._seeds
        seeds._by_trigger = loaded_seeds._by_trigger
    if values is not None and "values" in state:
        from .personality.values import ValueSystem
        loaded_values = ValueSystem.from_dict(state["values"])
        values.principles = loaded_values.principles
        values.preferences = loaded_values.preferences
    if relationships is not None and "relationships" in state:
        from .personality.relationship import RelationshipGraph
        loaded_rels = RelationshipGraph.from_dict(state["relationships"])
        relationships._rels = loaded_rels._rels
    if self_model is not None and "self_model" in state:
        from .personality.self_model import SelfModel
        sm = SelfModel.from_dict(state["self_model"])
        self_model.identity = sm.identity
        self_model.current_goal = sm.current_goal
        self_model.capability = sm.capability
        self_model.limitation = sm.limitation
        self_model.relationships = sm.relationships
        self_model.continuity = sm.continuity
        self_model.uncertainty = sm.uncertainty
        self_model.next_intention = sm.next_intention
    if identity is not None and "identity" in state:
        from .personality.identity import Identity
        id_ = Identity.from_dict(state["identity"])
        identity.name = id_.name
        identity.core_identity = id_.core_identity
        identity.expression_style = id_.expression_style
        identity.boundaries = id_.boundaries
        identity.past_self = id_.past_self
        identity.present_self = id_.present_self
        identity.future_self = id_.future_self
    if neurochem is not None and "neurochem" in state:
        from .cognition.neurochem import NeuroChemistry
        nc = NeuroChemistry.from_dict(state["neurochem"])
        neurochem.state = nc.state
    if metacog is not None and "metacognition" in state:
        from .cognition.metacognition import Metacognition
        mc = Metacognition.from_dict(state["metacognition"])
        metacog.cautiousness = mc.cautiousness
        metacog.strategy_bias = mc.strategy_bias
        metacog._reflections = mc._reflections
    # 恢复 v1.22.0 框架能力（人格维度 / 自主目标 / 被动用户画像）
    if personality is not None and "personality" in state:
        from .personality.dimensions import PersonalityDimensions
        loaded = PersonalityDimensions.from_dict(state["personality"])
        personality._traits = loaded._traits
        # 恢复进化模式开关与自定义风格（否则往返后回到默认 autonomous/空）
        personality.autonomy_mode = loaded.autonomy_mode
        personality.custom_style = loaded.custom_style
    if auto_goals is not None and "auto_goals" in state:
        from .autonomous_goals import AutonomousGoalEngine
        loaded = AutonomousGoalEngine.from_dict(state["auto_goals"])
        auto_goals._goals = loaded._goals
        auto_goals._last_generate = loaded._last_generate
        auto_goals.cooldown_seconds = loaded.cooldown_seconds
        auto_goals.max_goals = loaded.max_goals
    if user_profiles is not None and "user_profiles" in state:
        from .user_profile import UserProfileGraph
        loaded = UserProfileGraph.from_dict(state["user_profiles"])
        user_profiles._profiles = loaded._profiles
    if session is not None and "session" in state:
        for k, v in state["session"].items():
            session[k] = v
    return True


def load_state(path: str, needs: NeedDriveSystem, emotion: EmotionGradient,
               distiller: Distiller, goals: GoalManager,
               seeds=None, values=None, relationships=None, self_model=None,
               identity=None, neurochem=None, metacog=None,
               personality=None, auto_goals=None, user_profiles=None,
               session=None) -> bool:
    """从文件恢复认知+人格状态。返回是否成功加载。"""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(state, dict):
        return False   # 合法 JSON 但非 dict（如字符串/列表）→ 容错为加载失败
    return _restore(state, needs, emotion, distiller, goals, seeds, values, relationships, self_model, identity, neurochem, metacog, personality, auto_goals, user_profiles, session)
