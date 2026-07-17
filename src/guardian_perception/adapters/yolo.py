"""Ultralytics YOLO adapter; it owns detection only, never risk decisions."""

from collections.abc import Iterable, Mapping
from typing import Any

from ..types import BoundingBox, Detection


def yolo_rows_to_detections(
    rows: Iterable[tuple[float, float, float, float, float, int]],
    names: Mapping[int, str],
    *,
    frame_id: int,
    timestamp_ms: int,
) -> list[Detection]:
    """Normalize YOLO ``xyxy, confidence, class_id`` rows to ``Detection`` values."""

    return [
        Detection(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            bbox=BoundingBox(left, top, right, bottom),
            class_name=names[class_id],
            detector_confidence=confidence,
        )
        for left, top, right, bottom, confidence, class_id in rows
    ]


class YoloDetector:
    """Lazy Ultralytics wrapper for CPU smoke tests and future GPU execution."""

    def __init__(self, weights: str = "yolo11n.pt", confidence: float = 0.50) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:  # Keeps the core importable without ML dependencies.
            raise RuntimeError(
                "YOLO is optional. Install it with `.venv\\Scripts\\python -m pip install ultralytics`."
            ) from error
        self._model = YOLO(weights)
        self._confidence = confidence

    def detect_frame(self, frame: Any, *, frame_id: int, timestamp_ms: int) -> list[Detection]:
        """Run one frame and return detector-neutral objects.

        This method intentionally returns no track ID. A tracker adapter must create
        ``TrackObservation`` values before the risk engine is called.
        """

        result = self._model.predict(frame, conf=self._confidence, verbose=False)[0]
        rows = [
            (*box.xyxy[0].tolist(), float(box.conf[0]), int(box.cls[0]))
            for box in result.boxes
        ]
        return yolo_rows_to_detections(
            rows,
            result.names,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )
