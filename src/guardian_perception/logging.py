"""Dependency-free structured event logging."""

import json
from pathlib import Path

from .types import RiskDecision


class JsonlEventWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, decision: RiskDecision) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(decision.as_dict(), sort_keys=True) + "\n")
