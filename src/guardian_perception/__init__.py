"""Guardian's detector-neutral, vision-only forward-collision warning core."""

from .config import ReliabilityFusionConfig, RiskConfig
from .association import AssociationCandidate, greedy_one_to_one_assignment
from .fusion import CameraMeasurement, FusedTarget, RadarMeasurement, ReliabilityAwareFuser
from .radar import RadarCluster, RadarClusterTracker, RadarPoint, cluster_radar_points
from .risk import RiskEngine
from .tracker import IouTracker
from .types import BoundingBox, Detection, RiskDecision, RiskLevel, TrackObservation

__all__ = [
    "BoundingBox",
    "AssociationCandidate",
    "CameraMeasurement",
    "Detection",
    "FusedTarget",
    "RadarMeasurement",
    "ReliabilityAwareFuser",
    "RadarCluster",
    "RadarClusterTracker",
    "RadarPoint",
    "ReliabilityFusionConfig",
    "RiskConfig",
    "RiskDecision",
    "RiskEngine",
    "RiskLevel",
    "TrackObservation",
    "IouTracker",
    "cluster_radar_points",
    "greedy_one_to_one_assignment",
]
