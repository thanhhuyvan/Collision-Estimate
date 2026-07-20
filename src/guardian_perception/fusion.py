"""Lightweight reliability-aware camera-radar late fusion.

This module intentionally fuses *object-level measurements*, not raw neural-network features.
It is the learning baseline before BEV/attention fusion: every output keeps the source weights,
association confidence, disagreement, and fallback reason visible.
"""

from dataclasses import dataclass

from .config import ReliabilityFusionConfig


def _unit_interval(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class CameraMeasurement:
    """A camera track after a monocular range/velocity estimator is available."""

    track_id: str
    range_m: float
    closing_speed_mps: float
    detector_confidence: float
    image_quality: float

    def __post_init__(self) -> None:
        if self.range_m <= 0:
            raise ValueError("camera range must be positive")
        _unit_interval(self.detector_confidence, "detector_confidence")
        _unit_interval(self.image_quality, "image_quality")

    @property
    def reliability(self) -> float:
        return self.detector_confidence * self.image_quality


@dataclass(frozen=True)
class RadarMeasurement:
    """A radar return already associated to an image track by geometry/time gating."""

    return_id: str
    range_m: float
    closing_speed_mps: float
    signal_quality: float
    association_confidence: float

    def __post_init__(self) -> None:
        if self.range_m <= 0:
            raise ValueError("radar range must be positive")
        _unit_interval(self.signal_quality, "signal_quality")
        _unit_interval(self.association_confidence, "association_confidence")

    @property
    def reliability(self) -> float:
        return self.signal_quality * self.association_confidence


@dataclass(frozen=True)
class FusedTarget:
    track_id: str
    range_m: float
    closing_speed_mps: float
    ttc_s: float | None
    reliability: float
    camera_weight: float
    radar_weight: float
    radar_used: bool
    disagreement: bool
    explanation: str


class ReliabilityAwareFuser:
    """Fuse optional camera/radar measurements using reliability-scaled inverse variance.

    Radar is ignored when its camera association is below the configured gate. When sensor
    estimates conflict, the state remains available but its reliability is deliberately reduced;
    downstream safety policy can therefore choose warning-only behavior.
    """

    def __init__(self, config: ReliabilityFusionConfig | None = None) -> None:
        self.config = config or ReliabilityFusionConfig()

    def fuse(
        self, camera: CameraMeasurement, radar: RadarMeasurement | None = None
    ) -> FusedTarget:
        use_radar = radar is not None and (
            radar.association_confidence >= self.config.minimum_association_confidence
        )
        camera_range_weight = self._weight(camera.reliability, self.config.camera_range_std_m)
        camera_speed_weight = self._weight(
            camera.reliability, self.config.camera_closing_speed_std_mps
        )
        if use_radar:
            assert radar is not None
            radar_range_weight = self._weight(radar.reliability, self.config.radar_range_std_m)
            radar_speed_weight = self._weight(
                radar.reliability, self.config.radar_closing_speed_std_mps
            )
            range_m = self._mean(
                (camera.range_m, camera_range_weight), (radar.range_m, radar_range_weight)
            )
            closing_speed_mps = self._mean(
                (camera.closing_speed_mps, camera_speed_weight),
                (radar.closing_speed_mps, radar_speed_weight),
            )
            disagreement = (
                abs(camera.range_m - radar.range_m) > self.config.range_disagreement_m
                or abs(camera.closing_speed_mps - radar.closing_speed_mps)
                > self.config.closing_speed_disagreement_mps
            )
            reliability = self._combined_reliability(camera.reliability, radar.reliability)
            if disagreement:
                reliability *= 0.5
            total_range_weight = camera_range_weight + radar_range_weight
            explanation = (
                "camera+radar late fusion"
                + ("; sensor disagreement reduces reliability" if disagreement else "")
            )
            radar_weight = radar_range_weight / total_range_weight
            camera_weight = camera_range_weight / total_range_weight
        else:
            range_m = camera.range_m
            closing_speed_mps = camera.closing_speed_mps
            reliability = camera.reliability
            disagreement = False
            camera_weight = 1.0
            radar_weight = 0.0
            explanation = (
                "camera fallback; no radar return"
                if radar is None
                else "camera fallback; radar association below confidence gate"
            )

        ttc_s = (
            range_m / closing_speed_mps
            if closing_speed_mps > self.config.minimum_closing_speed_mps
            else None
        )
        return FusedTarget(
            track_id=camera.track_id,
            range_m=range_m,
            closing_speed_mps=closing_speed_mps,
            ttc_s=ttc_s,
            reliability=reliability,
            camera_weight=camera_weight,
            radar_weight=radar_weight,
            radar_used=use_radar,
            disagreement=disagreement,
            explanation=explanation,
        )

    @staticmethod
    def _weight(reliability: float, standard_deviation: float) -> float:
        return max(reliability, 0.01) / standard_deviation**2

    @staticmethod
    def _mean(*measurements: tuple[float, float]) -> float:
        total_weight = sum(weight for _, weight in measurements)
        return sum(value * weight for value, weight in measurements) / total_weight

    @staticmethod
    def _combined_reliability(camera: float, radar: float) -> float:
        # Adds only a bounded reinforcement bonus instead of assuming sensor independence.
        return min(1.0, max(camera, radar) + 0.25 * min(camera, radar))
