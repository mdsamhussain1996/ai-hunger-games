from __future__ import annotations

import os
from typing import List, Optional
import yaml

from src.models import Contestant


def load_seed_personalities(
    filepath: str = "personalities/seed.yaml",
    default_model: Optional[str] = None,
) -> List[Contestant]:
    """Load the initial 8 seed contestants from YAML."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Seed personality file not found at {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    contestants = []
    for item in data.get("personalities", []):
        base_model = default_model if default_model else item.get("base_model", "qwen2.5:0.5b")
        contestants.append(
            Contestant(
                id=item["id"],
                name=item["name"],
                generation=0,
                parent_ids=[],
                base_model=base_model,
                temperature=float(item.get("temperature", 0.7)),
                system_prompt=item["system_prompt"].strip(),
                status="active",
                replacement_strategy="seed",
            )
        )
    return contestants


class ContestantManager:
    """Manages active, eliminated, and archived contestants."""

    def __init__(self, contestants: Optional[List[Contestant]] = None):
        self._contestants: List[Contestant] = contestants or []

    @classmethod
    def from_seed(cls, filepath: str = "personalities/seed.yaml") -> "ContestantManager":
        return cls(load_seed_personalities(filepath))

    @property
    def all_contestants(self) -> List[Contestant]:
        return list(self._contestants)

    @property
    def active_contestants(self) -> List[Contestant]:
        return [c for c in self._contestants if c.status == "active"]

    def get_by_id(self, contestant_id: str) -> Optional[Contestant]:
        for c in self._contestants:
            if c.id == contestant_id:
                return c
        return None

    def add_contestant(self, contestant: Contestant) -> None:
        self._contestants.append(contestant)

    def eliminate(self, contestant_id: str, eliminated_at: Optional[any] = None) -> Optional[Contestant]:
        c = self.get_by_id(contestant_id)
        if c:
            c.status = "eliminated"
            c.eliminated_at = eliminated_at
            return c
        return None
