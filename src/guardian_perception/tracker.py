"""Small detector-neutral trackers for the laptop baseline.

``IouTracker`` is retained as a transparent reference. ``KalmanTracker`` adds a
constant-velocity Kalman motion model without adding a neural-network dependency.
"""

from dataclasses import dataclass
from math import hypot

from .types import BoundingBox, Detection, TrackObservation


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Return 2D box IoU, or zero when boxes do not overlap."""

    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = (first.right - first.left) * (first.bottom - first.top)
    second_area = (second.right - second.left) * (second.bottom - second.top)
    return intersection / (first_area + second_area - intersection)


@dataclass
class _Track:
    track_id: str
    class_name: str
    bbox: BoundingBox
    last_timestamp_ms: int
    tracker_confidence: float


class IouTracker:
    """Greedy same-class IoU tracker intended for low-rate offline evaluation.

    It deliberately has no appearance model or learned state. The GPU benchmark may replace
    it with ByteTrack/NvDCF, while preserving the same ``TrackObservation`` output contract.
    """

    def __init__(self, minimum_iou: float = 0.30, maximum_gap_ms: int = 400) -> None:
        if not 0 < minimum_iou <= 1:
            raise ValueError("minimum_iou must be in (0, 1]")
        if maximum_gap_ms <= 0:
            raise ValueError("maximum_gap_ms must be positive")
        self.minimum_iou = minimum_iou
        self.maximum_gap_ms = maximum_gap_ms
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1
        self.created_track_count = 0

    def update(
        self, detections: list[Detection], *, frame_width: int, frame_height: int
    ) -> list[TrackObservation]:
        """Assign persistent IDs and return observations for this frame's detections."""

        if detections:
            timestamp_ms = max(detection.timestamp_ms for detection in detections)
            self._expire(timestamp_ms)
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self._tracks)
        matches: list[tuple[str, int, float]] = []

        candidates: list[tuple[float, str, int]] = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                iou = intersection_over_union(track.bbox, detection.bbox)
                if iou >= self.minimum_iou:
                    candidates.append((iou, track_id, index))
        for iou, track_id, index in sorted(candidates, reverse=True):
            if track_id in unmatched_tracks and index in unmatched_detections:
                matches.append((track_id, index, iou))
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(index)

        observation_by_index: dict[int, TrackObservation] = {}
        for track_id, index, iou in matches:
            detection = detections[index]
            track = self._tracks[track_id]
            track.bbox = detection.bbox
            track.last_timestamp_ms = detection.timestamp_ms
            track.tracker_confidence = min(detection.detector_confidence, 0.60 + 0.40 * iou)
            observation_by_index[index] = self._observation(track, detection, frame_width, frame_height)

        for index in sorted(unmatched_detections):
            detection = detections[index]
            track_id = f"track-{self._next_id:04d}"
            self._next_id += 1
            self.created_track_count += 1
            track = _Track(
                track_id=track_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                last_timestamp_ms=detection.timestamp_ms,
                tracker_confidence=detection.detector_confidence,
            )
            self._tracks[track_id] = track
            observation_by_index[index] = self._observation(track, detection, frame_width, frame_height)

        return [observation_by_index[index] for index in range(len(detections))]

    def _expire(self, timestamp_ms: int) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp_ms - track.last_timestamp_ms > self.maximum_gap_ms
        ]
        for track_id in expired:
            del self._tracks[track_id]

    @staticmethod
    def _observation(
        track: _Track, detection: Detection, frame_width: int, frame_height: int
    ) -> TrackObservation:
        return TrackObservation(
            track_id=track.track_id,
            detection=detection,
            tracker_confidence=track.tracker_confidence,
            frame_width=frame_width,
            frame_height=frame_height,
        )


@dataclass
class _Kalman1D:
    """A 1D constant-velocity Kalman filter with a position-only measurement."""

    position: float
    velocity: float = 0.0
    covariance_pp: float = 25.0
    covariance_pv: float = 0.0
    covariance_vp: float = 0.0
    covariance_vv: float = 100.0

    def predict(self, dt_s: float) -> None:
        # F = [[1, dt], [0, 1]]; modest process noise allows box motion to adapt.
        previous_pp, previous_pv, previous_vp, previous_vv = (
            self.covariance_pp,
            self.covariance_pv,
            self.covariance_vp,
            self.covariance_vv,
        )
        self.position += dt_s * self.velocity
        self.covariance_pp = previous_pp + dt_s * (previous_pv + previous_vp) + dt_s * dt_s * previous_vv + 9.0
        self.covariance_pv = previous_pv + dt_s * previous_vv
        self.covariance_vp = previous_vp + dt_s * previous_vv
        self.covariance_vv = previous_vv + 16.0

    def update(self, measurement: float, measurement_variance: float = 16.0) -> None:
        # H = [1, 0].  The Joseph form is unnecessary for this tiny scalar filter,
        # but this covariance update keeps the position/velocity cross terms intact.
        residual = measurement - self.position
        innovation = self.covariance_pp + measurement_variance
        gain_position = self.covariance_pp / innovation
        gain_velocity = self.covariance_vp / innovation
        previous_pp, previous_pv = self.covariance_pp, self.covariance_pv
        self.position += gain_position * residual
        self.velocity += gain_velocity * residual
        self.covariance_pp = (1.0 - gain_position) * previous_pp
        self.covariance_pv = (1.0 - gain_position) * previous_pv
        self.covariance_vp -= gain_velocity * previous_pp
        self.covariance_vv -= gain_velocity * previous_pv


@dataclass
class _KalmanTrack(_Track):
    center_x: _Kalman1D
    center_y: _Kalman1D
    width: _Kalman1D
    height: _Kalman1D

    @classmethod
    def from_detection(cls, track_id: str, detection: Detection) -> "_KalmanTrack":
        box = detection.bbox
        return cls(
            track_id=track_id,
            class_name=detection.class_name,
            bbox=box,
            last_timestamp_ms=detection.timestamp_ms,
            tracker_confidence=detection.detector_confidence,
            center_x=_Kalman1D((box.left + box.right) / 2),
            center_y=_Kalman1D((box.top + box.bottom) / 2),
            width=_Kalman1D(box.right - box.left),
            height=_Kalman1D(box.bottom - box.top),
        )

    def predict_to(self, timestamp_ms: int) -> None:
        dt_s = max(0.0, timestamp_ms - self.last_timestamp_ms) / 1000.0
        for filter_ in (self.center_x, self.center_y, self.width, self.height):
            filter_.predict(dt_s)
        self.bbox = self._box_from_state()

    def update_from_detection(self, detection: Detection) -> None:
        box = detection.bbox
        for filter_, measurement in (
            (self.center_x, (box.left + box.right) / 2),
            (self.center_y, (box.top + box.bottom) / 2),
            (self.width, box.right - box.left),
            (self.height, box.bottom - box.top),
        ):
            filter_.update(measurement)
        self.bbox = self._box_from_state()
        self.last_timestamp_ms = detection.timestamp_ms

    def _box_from_state(self) -> BoundingBox:
        width, height = max(2.0, self.width.position), max(2.0, self.height.position)
        return BoundingBox(
            self.center_x.position - width / 2,
            self.center_y.position - height / 2,
            self.center_x.position + width / 2,
            self.center_y.position + height / 2,
        )


class KalmanTracker:
    """Same-class tracker using predicted boxes plus a conservative centre-distance gate.

    This is intentionally a lightweight bridge between IoU-only matching and a full
    appearance tracker such as ByteTrack.  It does not use object appearance, so dense
    traffic still requires the downstream reliability gates.
    """

    def __init__(self, minimum_iou: float = 0.10, maximum_gap_ms: int = 750, maximum_normalized_center_distance: float = 0.65) -> None:
        self.minimum_iou = minimum_iou
        self.maximum_gap_ms = maximum_gap_ms
        self.maximum_normalized_center_distance = maximum_normalized_center_distance
        self._tracks: dict[str, _KalmanTrack] = {}
        self._next_id = 1
        self.created_track_count = 0

    def update(self, detections: list[Detection], *, frame_width: int, frame_height: int) -> list[TrackObservation]:
        if not detections:
            return []
        timestamp_ms = max(detection.timestamp_ms for detection in detections)
        self._expire(timestamp_ms)
        for track in self._tracks.values():
            track.predict_to(timestamp_ms)

        unmatched_detections, unmatched_tracks = set(range(len(detections))), set(self._tracks)
        candidates: list[tuple[float, str, int, float]] = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                iou = intersection_over_union(track.bbox, detection.bbox)
                distance = self._normalized_center_distance(track.bbox, detection.bbox)
                if iou < self.minimum_iou and distance > self.maximum_normalized_center_distance:
                    continue
                # Predicted IoU remains primary; distance only rescues a fast but
                # geometrically plausible target after a sparse sensor interval.
                score = iou + 0.20 * max(0.0, 1.0 - distance)
                candidates.append((score, track_id, index, iou))

        observations_by_index: dict[int, TrackObservation] = {}
        for score, track_id, index, iou in sorted(candidates, reverse=True):
            if track_id not in unmatched_tracks or index not in unmatched_detections:
                continue
            track, detection = self._tracks[track_id], detections[index]
            track.update_from_detection(detection)
            track.tracker_confidence = min(detection.detector_confidence, 0.60 + 0.40 * max(iou, 1.0 - self._normalized_center_distance(track.bbox, detection.bbox)))
            observations_by_index[index] = IouTracker._observation(track, detection, frame_width, frame_height)
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(index)

        for index in sorted(unmatched_detections):
            detection = detections[index]
            track_id = f"track-{self._next_id:04d}"
            self._next_id += 1
            self.created_track_count += 1
            track = _KalmanTrack.from_detection(track_id, detection)
            self._tracks[track_id] = track
            observations_by_index[index] = IouTracker._observation(track, detection, frame_width, frame_height)
        return [observations_by_index[index] for index in range(len(detections))]

    def _expire(self, timestamp_ms: int) -> None:
        for track_id in [key for key, track in self._tracks.items() if timestamp_ms - track.last_timestamp_ms > self.maximum_gap_ms]:
            del self._tracks[track_id]

    @staticmethod
    def _normalized_center_distance(first: BoundingBox, second: BoundingBox) -> float:
        first_center = ((first.left + first.right) / 2, (first.top + first.bottom) / 2)
        second_center = ((second.left + second.right) / 2, (second.top + second.bottom) / 2)
        scale = max(2.0, hypot(first.right - first.left, first.bottom - first.top))
        return hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]) / scale
