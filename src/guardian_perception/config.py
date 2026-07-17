"""Configuration for the warning-only vision baseline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    """Safety-oriented defaults for a recorded-video baseline.

    All coordinates in ``ego_corridor`` are normalized image coordinates. Thresholds are
    deliberately configuration, not model, so every experiment can be reproduced.
    """

    ego_corridor: tuple[tuple[float, float], ...] = (
        (0.34, 1.00),
        (0.45, 0.48),
        (0.55, 0.48),
        (0.66, 1.00),
    )
    relevant_classes: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
        )
    )
    min_track_age_frames: int = 10
    max_observation_gap_ms: int = 250
    min_detector_confidence: float = 0.50
    min_tracker_confidence: float = 0.40
    min_approach_rate_px_s: float = 2.0
    caution_ttc_s: float = 4.0
    warning_ttc_s: float = 2.5
    warning_persistence_frames: int = 3
    exit_hysteresis_multiplier: float = 1.25

    def __post_init__(self) -> None:
        if len(self.ego_corridor) < 3:
            raise ValueError("ego_corridor requires at least three points")
        if not 0 < self.warning_ttc_s < self.caution_ttc_s:
            raise ValueError("warning TTC must be positive and below caution TTC")
        if self.min_track_age_frames < 2:
            raise ValueError("min_track_age_frames must be at least two")
