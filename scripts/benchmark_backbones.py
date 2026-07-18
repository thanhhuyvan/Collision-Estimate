"""Measure YOLO backbone latency against GuardianCo-Pilot's Fast Path budget."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2

from guardian_perception.adapters.yolo import YoloDetector

FAST_PATH_FIXED_MS = 60.0  # Proposal: capture 15 + planning 20 + safety 10 + UI/CAN 15.
INFERENCE_BUDGET_MS = 35.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no measurements")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def load_frame(path: Path):
    capture = cv2.VideoCapture(str(path))
    readable, frame = capture.read()
    capture.release()
    if not readable:
        raise RuntimeError(f"cannot read a frame from {path}")
    return frame


def measure_model(weights: str, frame, arguments: argparse.Namespace) -> dict[str, object]:
    load_started = time.perf_counter()
    detector = YoloDetector(weights, image_size=arguments.image_size, device=arguments.device)
    model_load_ms = (time.perf_counter() - load_started) * 1000

    cold_started = time.perf_counter()
    cold_detections = detector.detect_frame(frame, frame_id=0, timestamp_ms=0)
    cold_inference_ms = (time.perf_counter() - cold_started) * 1000
    for index in range(arguments.warmup):
        detector.detect_frame(frame, frame_id=index + 1, timestamp_ms=index + 1)

    measurements: list[float] = []
    detection_counts: list[int] = []
    for index in range(arguments.iterations):
        started = time.perf_counter()
        detections = detector.detect_frame(frame, frame_id=index, timestamp_ms=index)
        measurements.append((time.perf_counter() - started) * 1000)
        detection_counts.append(len(detections))

    p95_ms = percentile(measurements, 0.95)
    estimated_fast_path_ms = FAST_PATH_FIXED_MS + p95_ms
    return {
        "weights": weights,
        "device": arguments.device or "auto",
        "image_size": arguments.image_size,
        "model_load_ms": round(model_load_ms, 3),
        "cold_inference_ms": round(cold_inference_ms, 3),
        "warm_iterations": arguments.warmup,
        "measured_iterations": arguments.iterations,
        "mean_inference_ms": round(statistics.fmean(measurements), 3),
        "p50_inference_ms": round(percentile(measurements, 0.50), 3),
        "p95_inference_ms": round(p95_ms, 3),
        "max_inference_ms": round(max(measurements), 3),
        "estimated_fast_path_p95_ms": round(estimated_fast_path_ms, 3),
        "inference_headroom_ms": round(INFERENCE_BUDGET_MS - p95_ms, 3),
        "effective_p95_fps": round(1000 / p95_ms, 2),
        "detection_count": round(statistics.fmean(detection_counts), 2),
        "cold_detection_count": len(cold_detections),
        "meets_35ms_inference_budget": p95_ms <= INFERENCE_BUDGET_MS,
        "meets_100ms_fast_path_estimate": estimated_fast_path_ms <= 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", default=["yolo11n.pt"])
    parser.add_argument("--input", type=Path, default=Path("data/nexar_samples/positive/00015.mp4"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Use cpu on laptop; use 0 on the NVIDIA PC.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/backbone_benchmarks"))
    arguments = parser.parse_args()
    if arguments.image_size <= 0 or arguments.warmup < 0 or arguments.iterations <= 0:
        raise ValueError("image size and iteration counts must be positive")

    frame = load_frame(arguments.input)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    results = [measure_model(weights, frame, arguments) for weights in arguments.weights]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = arguments.output_dir / f"backbone_latency_{stamp}.json"
    csv_path = arguments.output_dir / f"backbone_latency_{stamp}.csv"
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "input": str(arguments.input),
        "frame_shape": list(frame.shape),
        "proposal_inference_budget_ms": INFERENCE_BUDGET_MS,
        "proposal_fixed_fast_path_budget_ms": FAST_PATH_FIXED_MS,
        "results": results,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    for result in results:
        status = "PASS" if result["meets_35ms_inference_budget"] else "FAIL"
        print(
            f"{status} {result['weights']}: p95={result['p95_inference_ms']} ms, "
            f"estimated Fast Path={result['estimated_fast_path_p95_ms']} ms, "
            f"headroom={result['inference_headroom_ms']} ms"
        )
    print(json_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
