"""Warning-only lead-object collision estimator for the hackathon MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from guardian_perception import CameraMeasurement, FusedTarget, RadarMeasurement, ReliabilityAwareFuser


class LeadRisk(StrEnum):
    NONE = "none"
    CAUTION = "caution"
    WARNING = "warning"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class LeadObjectInput:
    frame_id: int
    timestamp_ms: int
    camera: CameraMeasurement
    radar: RadarMeasurement | None
    in_ego_corridor: bool
    track_age_frames: int


@dataclass(frozen=True)
class LeadObjectDecision:
    frame_id: int
    timestamp_ms: int
    risk: LeadRisk
    fused_range_m: float
    fused_ttc_s: float | None
    reliability: float
    radar_used: bool
    reason: str
    evidence_status: str = "no_lead"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["risk"] = self.risk.value
        return result


class LeadObjectCollisionSystem:
    """Apply conservative physical-TTC policy after reliability-aware late fusion."""

    def __init__(self, *, minimum_track_age_frames: int = 3, caution_ttc_s: float = 4.0, warning_ttc_s: float = 2.5, warning_persistence_frames: int = 2, minimum_detector_confidence: float = 0.50, minimum_radar_association_confidence: float = 0.65, maximum_range_jump_m: float = 8.0, maximum_identity_gap_ms: int = 600) -> None:
        self.minimum_track_age_frames = minimum_track_age_frames
        self.caution_ttc_s = caution_ttc_s
        self.warning_ttc_s = warning_ttc_s
        self.warning_persistence_frames = warning_persistence_frames
        self.minimum_detector_confidence = minimum_detector_confidence
        self.minimum_radar_association_confidence = minimum_radar_association_confidence
        self.maximum_range_jump_m = maximum_range_jump_m
        self.maximum_identity_gap_ms = maximum_identity_gap_ms
        self._fuser = ReliabilityAwareFuser()
        self._active_track_id: str | None = None
        self._active_timestamp_ms: int | None = None
        self._radar_range_by_track: dict[str, tuple[int, float]] = {}
        self._warning_track_id: str | None = None
        self._warning_timestamp_ms: int | None = None
        self._warning_streak = 0

    def evaluate(self, input: LeadObjectInput) -> LeadObjectDecision:
        # A box-height range is deliberately only a camera fallback. Once an associated
        # radar target is trusted, radar owns range and relative velocity; treating the
        # proxy as a competing physical range caused false sensor-disagreement states.
        fallback = self._fuser.fuse(input.camera, None)
        if not input.in_ego_corridor:
            return self._decision(input, LeadRisk.NONE, fallback, "object outside ego corridor", "no_lead")
        if input.track_age_frames < self.minimum_track_age_frames:
            return self._decision(input, LeadRisk.NONE, fallback, "lead track is not stable yet", "provisional")
        if input.camera.detector_confidence < self.minimum_detector_confidence:
            return self._decision(input, LeadRisk.UNCERTAIN, fallback, "lead detector confidence below condition gate", "uncertain")
        if self._identity_changed_recently(input):
            return self._decision(input, LeadRisk.UNCERTAIN, fallback, "lead identity changed inside continuity window", "uncertain")
        if input.radar is None:
            return self._decision(input, LeadRisk.UNCERTAIN, fallback, "stable camera lead; radar confirmation unavailable", "camera_only")
        if input.radar is not None and input.radar.association_confidence < self.minimum_radar_association_confidence:
            return self._decision(input, LeadRisk.UNCERTAIN, fallback, "radar association confidence below warning gate", "uncertain")
        fused = self._radar_primary(input.camera, input.radar)
        if self._radar_range_jumped(input):
            return self._decision(input, LeadRisk.UNCERTAIN, fused, "radar range jump violates temporal condition gate", "uncertain")
        if fused.reliability < 0.60:
            return self._decision(input, LeadRisk.UNCERTAIN, fused, "radar confirmation quality below condition gate", "uncertain")
        self._remember(input)
        if fused.ttc_s is None:
            self._reset_warning_persistence()
            return self._decision(input, LeadRisk.NONE, fused, "lead object is not closing", "radar_confirmed")
        if fused.ttc_s <= self.warning_ttc_s:
            if self._advance_warning_persistence(input) < self.warning_persistence_frames:
                return self._decision(input, LeadRisk.CAUTION, fused, "radar TTC below warning threshold; waiting for persistence", "radar_confirmed")
            return self._decision(input, LeadRisk.WARNING, fused, "stable radar TTC below warning threshold", "radar_confirmed")
        self._reset_warning_persistence()
        if fused.ttc_s <= self.caution_ttc_s:
            return self._decision(input, LeadRisk.CAUTION, fused, "stable radar TTC below caution threshold", "radar_confirmed")
        return self._decision(input, LeadRisk.NONE, fused, "radar TTC is safe", "radar_confirmed")

    @staticmethod
    def _radar_primary(camera: CameraMeasurement, radar: RadarMeasurement) -> FusedTarget:
        """Create a lead-target state where camera proves identity/path and radar measures TTC."""

        reliability = min(1.0, radar.reliability * (0.75 + 0.25 * camera.reliability))
        ttc_s = radar.range_m / radar.closing_speed_mps if radar.closing_speed_mps > 0.10 else None
        return FusedTarget(
            track_id=camera.track_id,
            range_m=radar.range_m,
            closing_speed_mps=radar.closing_speed_mps,
            ttc_s=ttc_s,
            reliability=reliability,
            camera_weight=0.0,
            radar_weight=1.0,
            radar_used=True,
            disagreement=False,
            explanation="radar-primary TTC; camera confirms lead identity/path",
        )

    def _identity_changed_recently(self, input: LeadObjectInput) -> bool:
        previous_track = self._active_track_id
        previous_time = self._active_timestamp_ms
        return (
            previous_track is not None
            and previous_track != input.camera.track_id
            and previous_time is not None
            and input.timestamp_ms - previous_time <= self.maximum_identity_gap_ms
        )

    def _radar_range_jumped(self, input: LeadObjectInput) -> bool:
        if input.radar is None:
            return False
        previous = self._radar_range_by_track.get(input.camera.track_id)
        if previous is None:
            return False
        previous_time, previous_range = previous
        if input.timestamp_ms - previous_time > self.maximum_identity_gap_ms:
            return False
        return abs(input.radar.range_m - previous_range) > self.maximum_range_jump_m

    def _remember(self, input: LeadObjectInput) -> None:
        self._active_track_id = input.camera.track_id
        self._active_timestamp_ms = input.timestamp_ms
        if input.radar is not None:
            self._radar_range_by_track[input.camera.track_id] = (input.timestamp_ms, input.radar.range_m)

    def _advance_warning_persistence(self, input: LeadObjectInput) -> int:
        contiguous = (
            self._warning_track_id == input.camera.track_id
            and self._warning_timestamp_ms is not None
            and input.timestamp_ms - self._warning_timestamp_ms <= self.maximum_identity_gap_ms
        )
        self._warning_streak = self._warning_streak + 1 if contiguous else 1
        self._warning_track_id = input.camera.track_id
        self._warning_timestamp_ms = input.timestamp_ms
        return self._warning_streak

    def _reset_warning_persistence(self) -> None:
        self._warning_track_id = None
        self._warning_timestamp_ms = None
        self._warning_streak = 0

    @staticmethod
    def _decision(input: LeadObjectInput, risk: LeadRisk, fused, reason: str, evidence_status: str) -> LeadObjectDecision:
        return LeadObjectDecision(input.frame_id, input.timestamp_ms, risk, fused.range_m, fused.ttc_s, fused.reliability, fused.radar_used, reason, evidence_status)
