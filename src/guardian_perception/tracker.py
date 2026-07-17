"""A small dependency-free same-class IoU tracker for the laptop baseline."""

from dataclasses import dataclass

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
