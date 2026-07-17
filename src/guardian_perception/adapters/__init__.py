"""Runtime-specific adapters that normalize outputs into Guardian contracts."""

from .yolo import YoloDetector, yolo_rows_to_detections

__all__ = ["YoloDetector", "yolo_rows_to_detections"]
