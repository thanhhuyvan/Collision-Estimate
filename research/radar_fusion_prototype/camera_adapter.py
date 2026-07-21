"""Turn detector/tracker observations into one lead-object camera contract per frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from guardian_perception.geometry import bottom_center_is_in_corridor
from guardian_perception.types import TrackObservation


@dataclass(frozen=True)
class CameraLeadRecord:
    frame_id: int
    timestamp_ms: int
    track_id: str | None
    class_name: str | None
    detector_confidence: float | None
    tracker_confidence: float | None
    track_age_frames: int
    in_ego_corridor: bool
    bbox_xyxy: tuple[float, float, float, float] | None
    range_proxy_m: float | None
    selection_reason: str

    def as_dict(self) -> dict:
        return asdict(self)


class CameraLeadSelector:
    """Select the visually closest stable object inside the configured ego corridor."""

    def __init__(self, ego_corridor: tuple[tuple[float, float], ...]) -> None:
        self.ego_corridor = ego_corridor
        self._age_by_track: dict[str, int] = {}

    def select(self, observations: list[TrackObservation]) -> CameraLeadRecord:
        for observation in observations:
            self._age_by_track[observation.track_id] = self._age_by_track.get(observation.track_id, 0) + 1
        if not observations:
            return CameraLeadRecord(0, 0, None, None, None, None, 0, False, None, None, "no tracked detection")
        in_corridor = [
            observation for observation in observations
            if bottom_center_is_in_corridor(
                observation.detection.bbox,
                observation.frame_width,
                observation.frame_height,
                self.ego_corridor,
            )
        ]
        newest = max(observations, key=lambda item: item.detection.timestamp_ms).detection
        if not in_corridor:
            return CameraLeadRecord(newest.frame_id, newest.timestamp_ms, None, None, None, None, 0, False, None, None, "no object inside ego corridor")
        # Larger image height is a deliberately transparent proximity proxy. Radar later
        # provides physical range and velocity when association is reliable.
        selected = max(in_corridor, key=lambda item: item.detection.bbox.height)
        box = selected.detection.bbox
        assumed_focal_px = selected.frame_width * 0.9
        assumed_vehicle_height_m = 1.6
        range_proxy_m = assumed_focal_px * assumed_vehicle_height_m / box.height
        return CameraLeadRecord(
            selected.detection.frame_id,
            selected.detection.timestamp_ms,
            selected.track_id,
            selected.detection.class_name,
            selected.detection.detector_confidence,
            selected.tracker_confidence,
            self._age_by_track[selected.track_id],
            True,
            (box.left, box.top, box.right, box.bottom),
            range_proxy_m,
            "largest tracked object inside ego corridor",
        )
