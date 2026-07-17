"""Run the detector -> tracker -> TTC/risk pipeline over the six Nexar starter clips."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from guardian_perception import IouTracker, RiskConfig, RiskEngine, RiskLevel
from guardian_perception.adapters.yolo import YoloDetector
from guardian_perception.geometry import bottom_center_is_in_corridor


@dataclass(frozen=True)
class ClipMetadata:
    file_name: str
    label: int
    time_of_event: float | None
    time_of_alert: float | None
    light_conditions: str
    weather: str
    scene: str


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def load_metadata(path: Path) -> list[ClipMetadata]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return [
            ClipMetadata(
                file_name=row["file_name"],
                label=int(row["label"]),
                time_of_event=optional_float(row["time_of_event"]),
                time_of_alert=optional_float(row["time_of_alert"]),
                light_conditions=row["light_conditions"],
                weather=row["weather"],
                scene=row["scene"],
            )
            for row in csv.DictReader(input_file)
        ]


def test_window(metadata: ClipMetadata) -> tuple[float, float]:
    if metadata.label:
        assert metadata.time_of_alert is not None and metadata.time_of_event is not None
        return max(0.0, metadata.time_of_alert - 5.0), metadata.time_of_event + 2.0
    return 10.0, 20.0


def draw_corridor(frame, config: RiskConfig) -> None:
    height, width = frame.shape[:2]
    points = [tuple(map(int, (x * width, y * height))) for x, y in config.ego_corridor]
    polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon], True, (255, 200, 0), 2)


def draw_frame(frame, observations, decision, config: RiskConfig, metadata: ClipMetadata, timestamp_s: float) -> None:
    draw_corridor(frame, config)
    selected = decision.selected_track_id
    for observation in observations:
        box = observation.detection.bbox
        left, top, right, bottom = map(round, (box.left, box.top, box.right, box.bottom))
        is_selected = observation.track_id == selected
        colour = (0, 0, 255) if is_selected and decision.risk is RiskLevel.WARNING else (0, 210, 255) if is_selected else (40, 200, 90)
        cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
        label = f"{observation.track_id} {observation.detection.class_name} {observation.detection.detector_confidence:.2f}"
        cv2.putText(frame, label, (left, max(54, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

    ttc = "--" if decision.ttc_estimate_s is None else f"{decision.ttc_estimate_s:.2f}s"
    header = f"RISK={decision.risk.value.upper()}  TTC={ttc}  target={selected or '--'}  t={timestamp_s:.2f}s"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (18, 18, 18), -1)
    cv2.putText(frame, header, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if metadata.label:
        reference = f"REFERENCE: alert={metadata.time_of_alert:.3f}s event={metadata.time_of_event:.3f}s"
    else:
        reference = "REFERENCE: normal-driving negative clip"
    cv2.putText(frame, reference, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def run_clip(
    metadata: ClipMetadata,
    *,
    detector: YoloDetector,
    sample_fps: float,
    output_root: Path,
    source_root: Path,
) -> dict[str, object]:
    source = source_root / ("positive" if metadata.label else "negative") / metadata.file_name
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot read {source}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_s, end_s = test_window(metadata)
    start_frame = round(start_s * source_fps)
    end_frame = round(end_s * source_fps)
    stride = max(1, round(source_fps / sample_fps))
    config = RiskConfig(max_observation_gap_ms=round(1000 / sample_fps * 1.5))
    tracker = IouTracker(maximum_gap_ms=400)
    engine = RiskEngine(config)

    clip_root = output_root / metadata.file_name.removesuffix(".mp4")
    clip_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = clip_root / "decisions.jsonl"
    video_path = clip_root / "annotated.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    processed = 0
    total_detections = 0
    in_corridor_observations = 0
    observations_by_track: dict[str, int] = {}
    last_processed_s: float | None = None
    first_warning_s: float | None = None
    minimum_ttc_s: float | None = None
    warning_frames = 0
    run_started = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as decisions_file:
        while frame_id <= end_frame:
            readable, frame = capture.read()
            if not readable:
                break
            timestamp_ms = round(frame_id * 1000 / source_fps)
            inference_started = time.perf_counter()
            detections = detector.detect_frame(frame, frame_id=frame_id, timestamp_ms=timestamp_ms)
            inference_ms = (time.perf_counter() - inference_started) * 1000
            observations = tracker.update(detections, frame_width=frame_width, frame_height=frame_height)
            for observation in observations:
                observations_by_track[observation.track_id] = observations_by_track.get(observation.track_id, 0) + 1
                if bottom_center_is_in_corridor(
                    observation.detection.bbox,
                    frame_width,
                    frame_height,
                    config.ego_corridor,
                ):
                    in_corridor_observations += 1
            decision = engine.evaluate_frame(observations, frame_id=frame_id, timestamp_ms=timestamp_ms)
            timestamp_s = timestamp_ms / 1000
            last_processed_s = timestamp_s
            if decision.ttc_estimate_s is not None:
                minimum_ttc_s = min(minimum_ttc_s, decision.ttc_estimate_s) if minimum_ttc_s is not None else decision.ttc_estimate_s
            if decision.risk is RiskLevel.WARNING:
                warning_frames += 1
                first_warning_s = first_warning_s if first_warning_s is not None else timestamp_s
            draw_frame(frame, observations, decision, config, metadata, timestamp_s)
            writer.write(frame)
            decisions_file.write(
                json.dumps(
                    {
                        **decision.as_dict(),
                        "timestamp_s": timestamp_s,
                        "detection_count": len(detections),
                        "track_count": len(observations),
                        "inference_ms": round(inference_ms, 3),
                        "observations": [
                            {
                                "track_id": observation.track_id,
                                "class_name": observation.detection.class_name,
                                "detector_confidence": round(observation.detection.detector_confidence, 4),
                                "tracker_confidence": round(observation.tracker_confidence, 4),
                                "in_ego_corridor": bottom_center_is_in_corridor(
                                    observation.detection.bbox,
                                    frame_width,
                                    frame_height,
                                    config.ego_corridor,
                                ),
                                "bbox_xyxy": [round(value, 1) for value in (
                                    observation.detection.bbox.left,
                                    observation.detection.bbox.top,
                                    observation.detection.bbox.right,
                                    observation.detection.bbox.bottom,
                                )],
                            }
                            for observation in observations
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            processed += 1
            total_detections += len(detections)
            for _ in range(stride - 1):
                if frame_id >= end_frame:
                    break
                capture.grab()
                frame_id += 1
            frame_id += 1
    capture.release()
    writer.release()

    max_track_observations = max(observations_by_track.values(), default=0)
    if metadata.label:
        classification = "warning_before_event" if first_warning_s is not None and first_warning_s <= metadata.time_of_event else "missed_or_late_warning"
        if classification == "warning_before_event":
            failure_category = ""
        elif total_detections == 0:
            failure_category = "detector_miss"
        elif max_track_observations < config.min_track_age_frames:
            failure_category = "id_switch"
        elif in_corridor_observations == 0:
            failure_category = "corridor_error"
        elif minimum_ttc_s is None or minimum_ttc_s > config.warning_ttc_s:
            failure_category = "threshold_issue"
        else:
            failure_category = "ttc_instability"
    else:
        classification = "false_warning" if first_warning_s is not None else "no_warning"
        failure_category = "" if classification == "no_warning" else "threshold_issue"
    return {
        "file_name": metadata.file_name,
        "label": metadata.label,
        "scene": metadata.scene,
        "weather": metadata.weather,
        "light_conditions": metadata.light_conditions,
        "reference_alert_s": metadata.time_of_alert or "",
        "reference_event_s": metadata.time_of_event or "",
        "window_start_s": round(start_s, 3),
        "window_end_s": round(end_s, 3),
        "processed_frames": processed,
        "last_processed_s": "" if last_processed_s is None else round(last_processed_s, 3),
        "total_detections": total_detections,
        "created_tracks": tracker.created_track_count,
        "max_track_observations": max_track_observations,
        "in_corridor_observations": in_corridor_observations,
        "first_warning_s": first_warning_s or "",
        "minimum_ttc_s": "" if minimum_ttc_s is None else round(minimum_ttc_s, 3),
        "warning_duration_s": round(warning_frames / sample_fps, 3),
        "classification": classification,
        "preliminary_failure_category": failure_category,
        "elapsed_s": round(time.perf_counter() - run_started, 3),
        "annotated_video": str(video_path),
        "decision_log": str(jsonl_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/nexar_samples/metadata.csv"))
    parser.add_argument("--source-root", type=Path, default=Path("data/nexar_samples"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/laptop_test"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--clip-id", help="Optional clip stem, for example 00015")
    arguments = parser.parse_args()
    rows = load_metadata(arguments.metadata)
    if arguments.clip_id:
        rows = [row for row in rows if row.file_name.removesuffix(".mp4") == arguments.clip_id]
        if not rows:
            raise ValueError(f"clip {arguments.clip_id} is not in {arguments.metadata}")
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    detector = YoloDetector(arguments.weights)
    summaries = [
        run_clip(
            row,
            detector=detector,
            sample_fps=arguments.sample_fps,
            output_root=arguments.output_root,
            source_root=arguments.source_root,
        )
        for row in rows
    ]
    summary_path = arguments.output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
