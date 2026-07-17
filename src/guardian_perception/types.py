"""Stable contracts shared by detector, tracker, risk, and logging layers."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("bounding box must have positive width and height")

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, self.bottom)


@dataclass(frozen=True)
class Detection:
    frame_id: int
    timestamp_ms: int
    bbox: BoundingBox
    class_name: str
    detector_confidence: float


@dataclass(frozen=True)
class TrackObservation:
    """A tracker-normalized detection consumed by the risk engine."""

    track_id: str
    detection: Detection
    tracker_confidence: float
    frame_width: int
    frame_height: int


class RiskLevel(StrEnum):
    NONE = "none"
    CAUTION = "caution"
    WARNING = "warning"


@dataclass(frozen=True)
class RiskDecision:
    frame_id: int
    timestamp_ms: int
    risk: RiskLevel
    selected_track_id: str | None
    ttc_estimate_s: float | None
    ttc_confidence: float
    reason: str
    candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["risk"] = self.risk.value
        return result
