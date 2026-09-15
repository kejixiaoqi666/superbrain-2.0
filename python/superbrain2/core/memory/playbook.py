"""Opt-in ACE-style evolving context playbook (delta updates, no LLM required).
Entries are immutable observations; curation deterministically merges/refines by key.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable

@dataclass
class PlaybookEntry:
    key: str
    lesson: str
    success: float = 0.0
    uses: int = 0
    version: int = 1

class ContextPlaybook:
    def __init__(self, entries: Iterable[PlaybookEntry] = ()):
        self.entries = {e.key: e for e in entries}

    def delta(self, key: str, lesson: str, success: bool) -> dict:
        """Reflect a trajectory into a localized delta."""
        return {"key": key.strip(), "lesson": lesson.strip(), "success": 1.0 if success else 0.0}

    def curate(self, deltas: Iterable[dict]) -> PlaybookEntry | None:
        """Grow-and-refine: update only the addressed entry, preserving detail."""
        chosen = None
        for d in deltas:
            key, lesson = d.get("key", "").strip(), d.get("lesson", "").strip()
            if not key or not lesson: continue
            old = self.entries.get(key)
            if old:
                old.lesson = old.lesson if lesson in old.lesson else old.lesson + " | " + lesson
                old.success = (old.success * old.uses + float(d.get("success", 0))) / (old.uses + 1)
                old.uses += 1; old.version += 1; chosen = old
            else:
                chosen = PlaybookEntry(key, lesson, float(d.get("success", 0)), 1)
                self.entries[key] = chosen
        return chosen

    def render(self, limit: int | None = None) -> str:
        vals = list(self.entries.values())
        vals.sort(key=lambda e: (-e.success, -e.uses, e.key))
        return "\n".join(f"[{e.key}] {e.lesson}" for e in vals[:limit])

    def to_dict(self): return [asdict(e) for e in self.entries.values()]
    @classmethod
    def from_dict(cls, data): return cls(PlaybookEntry(**e) for e in data)
