"""Render a small CPU-friendly YOLO contact sheet from a recorded dashcam clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from guardian_perception.adapters.yolo import YoloDetector


def draw_detections(frame, detections) -> None:
    for detection in detections:
        box = detection.bbox
        left, top, right, bottom = map(round, (box.left, box.top, box.right, box.bottom))
        cv2.rectangle(frame, (left, top), (right, bottom), (38, 202, 114), 2)
        label = f"{detection.class_name} {detection.detector_confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (left, max(24, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (38, 202, 114),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/nexar_samples/positive/00015.mp4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_preview_00015.jpg"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--samples", type=int, default=4)
    arguments = parser.parse_args()

    capture = cv2.VideoCapture(str(arguments.input))
    if not capture.isOpened():
        raise RuntimeError(f"cannot read {arguments.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = [round(index * max(frame_count - 1, 0) / (arguments.samples - 1)) for index in range(arguments.samples)]
    detector = YoloDetector(arguments.weights)
    panels = []
    for frame_id in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        readable, frame = capture.read()
        if not readable:
            raise RuntimeError(f"cannot read frame {frame_id}")
        detections = detector.detect_frame(
            frame,
            frame_id=frame_id,
            timestamp_ms=round(frame_id * 1000 / fps),
        )
        draw_detections(frame, detections)
        title = f"DETECTION-ONLY | t={frame_id / fps:.1f}s | objects={len(detections)}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(frame, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(frame)
    capture.release()

    top = cv2.hconcat(panels[:2])
    bottom = cv2.hconcat(panels[2:])
    contact_sheet = cv2.vconcat([top, bottom])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(arguments.output), contact_sheet):
        raise RuntimeError(f"cannot write {arguments.output}")
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
