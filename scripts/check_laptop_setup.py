"""Check that a machine can run the offline six-clip baseline without downloading data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/nexar_samples"))
    arguments = parser.parse_args()

    failures: list[str] = []
    try:
        import cv2
        import torch
        import ultralytics
    except ImportError as error:
        print(f"FAIL: missing laptop dependency: {error}")
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if not arguments.weights.is_file():
        failures.append(f"missing model weights: {arguments.weights}")
    metadata = arguments.data_root / "metadata.csv"
    if not metadata.is_file():
        failures.append(f"missing six-clip manifest: {metadata}")
    else:
        for clip_id, category in (
            ("00015", "positive"), ("00026", "positive"), ("00054", "positive"),
            ("01042", "negative"), ("01079", "negative"), ("01102", "negative"),
        ):
            video = arguments.data_root / category / f"{clip_id}.mp4"
            if not video.is_file():
                failures.append(f"missing sample video: {video}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: offline laptop baseline prerequisites are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
