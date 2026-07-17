"""Guardian's detector-neutral, vision-only forward-collision warning core."""

from .config import RiskConfig
from .risk import RiskEngine
from .tracker import IouTracker
from .types import BoundingBox, Detection, RiskDecision, RiskLevel, TrackObservation

__all__ = [
    "BoundingBox",
    "Detection",
    "RiskConfig",
    "RiskDecision",
    "RiskEngine",
    "RiskLevel",
    "TrackObservation",
    "IouTracker",
]
