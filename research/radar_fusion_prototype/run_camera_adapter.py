"""Run YOLO + tracker on a small image sequence and emit camera lead-object records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from guardian_perception.adapters.yolo import YoloDetector
from guardian_perception.config import RiskConfig
from guardian_perception.tracker import IouTracker

from camera_adapter import CameraLeadSelector


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path)
    source.add_argument("--manifest", type=Path, help="nuScenes-style manifest; preserves intended frame order.")
    parser.add_argument("--data-root", type=Path, help="Root used to resolve manifest camera file paths.")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "camera_lead_records.jsonl")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--confidence", type=float, default=0.50)
    args = parser.parse_args()
    if args.manifest:
        if args.data_root is None:
            raise ValueError("--data-root is required with --manifest")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        images = [args.data_root / frame["camera"]["file"] for frame in manifest["frames"][: args.max_frames]]
    else:
        assert args.input_dir is not None
        images = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.max_frames]
    if not images:
        raise FileNotFoundError(f"no images in {args.input_dir}")
    detector = YoloDetector(args.weights, confidence=args.confidence)
    tracker = IouTracker()
    selector = CameraLeadSelector(RiskConfig().ego_corridor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for frame_id, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"cannot read {image_path}")
            timestamp_ms = round(frame_id * 1000 / args.fps)
            detections = detector.detect_frame(image, frame_id=frame_id, timestamp_ms=timestamp_ms)
            observations = tracker.update(detections, frame_width=image.shape[1], frame_height=image.shape[0])
            lead = selector.select(observations)
            stream.write(json.dumps({"image": image_path.name, **lead.as_dict()}) + "\n")
            print(f"frame={frame_id} detections={len(detections)} lead={lead.track_id} reason={lead.selection_reason}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
