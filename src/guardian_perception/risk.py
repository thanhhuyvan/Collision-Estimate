"""Temporal, confidence-gated risk logic for the vision-only baseline."""

from collections import deque
from dataclasses import dataclass, field

from .config import RiskConfig
from .geometry import bottom_center_is_in_corridor
from .types import RiskDecision, RiskLevel, TrackObservation


@dataclass
class _TrackState:
    observations: deque[TrackObservation] = field(default_factory=lambda: deque(maxlen=10))
    warning_streak: int = 0
    active_risk: RiskLevel = RiskLevel.NONE


class RiskEngine:
    """Convert tracked 2D detections into conservative FCW risk decisions.

    The calculation is a visual TTC proxy: box height divided by its positive temporal
    growth rate. It must never be interpreted as physical range or as permission to brake.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._states: dict[str, _TrackState] = {}

    def evaluate_frame(
        self, observations: list[TrackObservation], *, frame_id: int = 0, timestamp_ms: int = 0
    ) -> RiskDecision:
        """Evaluate one frame of tracker observations and choose one relevant target."""

        if not observations:
            return RiskDecision(
                frame_id, timestamp_ms, RiskLevel.NONE, None, None, 0.0, "no observations", 0
            )

        candidates: list[tuple[RiskLevel, float, float, str, TrackObservation]] = []
        for observation in observations:
            evaluation = self._evaluate_observation(observation)
            if evaluation is not None:
                candidates.append((*evaluation, observation))

        latest = max(observations, key=lambda item: item.detection.timestamp_ms).detection
        if not candidates:
            return RiskDecision(
                latest.frame_id,
                latest.timestamp_ms,
                RiskLevel.NONE,
                None,
                None,
                0.0,
                "no warning-eligible target",
                0,
            )

        # Severity dominates selection; lower TTC breaks ties within the same severity.
        priority = {RiskLevel.NONE: 0, RiskLevel.CAUTION: 1, RiskLevel.WARNING: 2}
        risk, ttc, confidence, reason, selected = max(
            candidates, key=lambda item: (priority[item[0]], -item[1], item[2])
        )
        return RiskDecision(
            latest.frame_id,
            latest.timestamp_ms,
            risk,
            selected.track_id,
            ttc,
            confidence,
            reason,
            len(candidates),
        )

    def _evaluate_observation(
        self, observation: TrackObservation
    ) -> tuple[RiskLevel, float, float, str] | None:
        detection = observation.detection
        if detection.class_name not in self.config.relevant_classes:
            return None
        if not bottom_center_is_in_corridor(
            detection.bbox,
            observation.frame_width,
            observation.frame_height,
            self.config.ego_corridor,
        ):
            return None
        if detection.detector_confidence < self.config.min_detector_confidence:
            return None
        if observation.tracker_confidence < self.config.min_tracker_confidence:
            return None

        state = self._states.setdefault(observation.track_id, _TrackState())
        if state.observations:
            elapsed = detection.timestamp_ms - state.observations[-1].detection.timestamp_ms
            if elapsed <= 0 or elapsed > self.config.max_observation_gap_ms:
                state.observations.clear()
                state.warning_streak = 0
                state.active_risk = RiskLevel.NONE
        state.observations.append(observation)

        if len(state.observations) < self.config.min_track_age_frames:
            return None

        first = state.observations[0].detection
        last = state.observations[-1].detection
        seconds = (last.timestamp_ms - first.timestamp_ms) / 1000
        if seconds <= 0:
            return None
        approach_rate = (last.bbox.height - first.bbox.height) / seconds
        if approach_rate < self.config.min_approach_rate_px_s:
            state.warning_streak = 0
            state.active_risk = RiskLevel.NONE
            return None

        ttc = last.bbox.height / approach_rate
        confidence = min(detection.detector_confidence, observation.tracker_confidence)
        proposed = self._risk_from_ttc(ttc)
        if proposed is RiskLevel.WARNING:
            state.warning_streak += 1
            if state.warning_streak < self.config.warning_persistence_frames:
                proposed = RiskLevel.CAUTION
        else:
            state.warning_streak = 0

        # Do not oscillate down from warning until TTC has moved safely beyond the
        # caution threshold by a deliberate exit margin.
        if (
            state.active_risk is RiskLevel.WARNING
            and ttc < self.config.caution_ttc_s * self.config.exit_hysteresis_multiplier
            and proposed is not RiskLevel.WARNING
        ):
            proposed = RiskLevel.WARNING

        state.active_risk = proposed
        return proposed, ttc, confidence, "stable in-corridor visual TTC"

    def _risk_from_ttc(self, ttc: float) -> RiskLevel:
        if ttc <= self.config.warning_ttc_s:
            return RiskLevel.WARNING
        if ttc <= self.config.caution_ttc_s:
            return RiskLevel.CAUTION
        return RiskLevel.NONE
