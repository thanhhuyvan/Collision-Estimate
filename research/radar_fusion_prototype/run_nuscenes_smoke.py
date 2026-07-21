"""Run the complete hackathon MVP on a small calibrated nuScenes sequence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from guardian_perception import CameraMeasurement
from guardian_perception.adapters.yolo import YoloDetector
from guardian_perception.config import RiskConfig
from guardian_perception.tracker import IouTracker, KalmanTracker

from camera_adapter import CameraLeadSelector
from lead_object_system import LeadObjectCollisionSystem, LeadObjectDecision, LeadObjectInput, LeadRisk
from radar_adapter import RadarTemporalFilter, associate_radar_to_lead


def draw_overlay(image, observations, lead, radar, decision, corridor) -> None:
    height, width = image.shape[:2]
    polygon = [(round(x * width), round(y * height)) for x, y in corridor]
    cv2.polylines(image, [np.array(polygon)], True, (255, 200, 0), 2, cv2.LINE_AA)
    # Draw every tracked detection.  A cyan box is the object selected for collision
    # reasoning; grey boxes are still detected/tracked, but are outside the current
    # lead-object decision path.
    for observation in observations:
        box = observation.detection.bbox
        left, top, right, bottom = map(round, (box.left, box.top, box.right, box.bottom))
        selected = observation.track_id == lead.track_id
        colour = (0, 200, 255) if selected else (160, 160, 160)
        thickness = 2 if selected else 1
        cv2.rectangle(image, (left, top), (right, bottom), colour, thickness)
        label = f"{observation.detection.class_name} {observation.track_id}"
        cv2.putText(image, label, (left, max(92, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)
    if lead.bbox_xyxy:
        left, top, right, bottom = map(round, lead.bbox_xyxy)
        colour = (0, 200, 255) if decision.radar_used else (0, 130, 255)
        cv2.putText(image, f"lead={lead.track_id} age={lead.track_age_frames}", (left, max(58, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (width, 78), (15, 15, 15), -1)
    radar_text = f"radar matched={radar.matched_point_count} used={decision.radar_used}" if radar else "radar unavailable"
    cv2.putText(image, f"risk={decision.risk.value} state={decision.evidence_status} TTC={decision.fused_ttc_s} reliability={decision.reliability:.2f}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, radar_text + f" | {decision.reason}", (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "nuscenes_smoke")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--scene", help="Optional scene name, for example scene-0061.")
    parser.add_argument("--max-frames", type=int, help="Optional cap after scene filtering.")
    parser.add_argument("--tracker", choices=("iou", "kalman"), default="kalman")
    parser.add_argument("--association", choices=("point_box", "cluster_geometry", "cluster_geometry_temporal", "cluster_geometry_pose_temporal"), default="point_box")
    parser.add_argument(
        "--tracker-max-gap-ms",
        type=int,
        help="Optional track expiry age. Defaults to 1.5x the selected sensor cadence, at least 400 ms.",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = manifest["frames"]
    if args.scene:
        frames = [frame for frame in frames if frame.get("scene") == args.scene]
    if args.max_frames:
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError("No frames selected. Check --scene and the manifest.")
    # Video playback can be 5 FPS for comfortable review, while every policy decision
    # must use the physical capture time supplied by nuScenes (normally about 2 Hz).
    source_times_ms = [round((frame["timestamp_us"] - frames[0]["timestamp_us"]) / 1000) for frame in frames]
    deltas = [later - earlier for earlier, later in zip(source_times_ms, source_times_ms[1:]) if later > earlier]
    nominal_cadence_ms = round(float(np.median(deltas))) if deltas else 400
    tracker_gap_ms = args.tracker_max_gap_ms or max(400, round(nominal_cadence_ms * 1.5))
    detector = YoloDetector(args.weights, confidence=args.confidence)
    tracker = (KalmanTracker if args.tracker == "kalman" else IouTracker)(maximum_gap_ms=tracker_gap_ms)
    config = RiskConfig()
    selector, collision = CameraLeadSelector(config.ego_corridor), LeadObjectCollisionSystem()
    temporal_radar = RadarTemporalFilter() if args.association in {"cluster_geometry_temporal", "cluster_geometry_pose_temporal"} else None
    decisions_path = args.output_dir / "decisions.jsonl"
    writer = None
    replay_records: list[dict[str, object]] = []
    print(f"tracker={args.tracker} sensor_cadence_ms={nominal_cadence_ms} tracker_max_gap_ms={tracker_gap_ms}")
    with decisions_path.open("w", encoding="utf-8") as stream:
        for frame_id, frame in enumerate(frames):
            image = cv2.imread(str(args.data_root / frame["camera"]["file"]))
            timestamp_ms = source_times_ms[frame_id]
            detections = detector.detect_frame(image, frame_id=frame_id, timestamp_ms=timestamp_ms)
            observations = tracker.update(detections, frame_width=image.shape[1], frame_height=image.shape[0])
            lead = selector.select(observations)
            radar = associate_radar_to_lead(data_root=args.data_root, radar_record=frame["radar"], camera_record=frame["camera"], bbox_xyxy=lead.bbox_xyxy, return_id=f"radar-{frame_id}", mode=args.association, timestamp_ms=timestamp_ms, temporal_filter=temporal_radar)
            if lead.track_id and lead.range_proxy_m:
                camera = CameraMeasurement(lead.track_id, lead.range_proxy_m, 0.0, lead.detector_confidence or 0.0, lead.tracker_confidence or 0.0)
                decision = collision.evaluate(LeadObjectInput(frame_id, timestamp_ms, camera, radar.measurement, lead.in_ego_corridor, lead.track_age_frames))
            else:
                # Preserve the decision schema even when camera has no eligible lead target.
                decision = LeadObjectDecision(frame_id, timestamp_ms, LeadRisk.NONE, 0.0, None, 0.0, False, lead.selection_reason)
            draw_overlay(image, observations, lead, radar, decision, config.ego_corridor)
            if writer is None:
                writer = cv2.VideoWriter(str(args.output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (image.shape[1], image.shape[0]))
            writer.write(image)
            tracks = [
                {
                    "track_id": item.track_id,
                    "class_name": item.detection.class_name,
                    "detector_confidence": item.detection.detector_confidence,
                    "bbox_xyxy": [item.detection.bbox.left, item.detection.bbox.top, item.detection.bbox.right, item.detection.bbox.bottom],
                }
                for item in observations
            ]
            stream.write(json.dumps({"image": frame["camera"]["file"], "tracks": tracks, "lead": lead.as_dict(), "radar": {"raw_point_count": radar.raw_point_count, "projected_point_count": radar.projected_point_count, "matched_point_count": radar.matched_point_count, "association_confidence": radar.association_confidence, "temporal_track_age": radar.temporal_track_age, "temporal_consistency": radar.temporal_consistency}, "decision": decision.as_dict()}) + "\n")
            replay_records.append({
                "scene": frame.get("scene", "unknown"),
                "frame_id": frame_id,
                "track_count": len(tracks),
                "lead_track_id": lead.track_id or "",
                "risk": decision.risk.value,
                "evidence_status": decision.evidence_status,
                "radar_matched_points": radar.matched_point_count,
            })
            print(f"frame={frame_id} lead={lead.track_id} matched={radar.matched_point_count} risk={decision.risk.value}")
    assert writer is not None
    writer.release()
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as summary_stream:
        columns = ["scene", "frames", "detection_frames", "lead_frames", "radar_supported_lead_frames", "warning_frames", "caution_frames", "uncertain_frames", "radar_confirmed_frames", "camera_only_frames"]
        row = {
            "scene": replay_records[0]["scene"],
            "frames": len(replay_records),
            "detection_frames": sum(record["track_count"] > 0 for record in replay_records),
            "lead_frames": sum(bool(record["lead_track_id"]) for record in replay_records),
            "radar_supported_lead_frames": sum(bool(record["lead_track_id"]) and record["radar_matched_points"] > 0 for record in replay_records),
            "warning_frames": sum(record["risk"] == "warning" for record in replay_records),
            "caution_frames": sum(record["risk"] == "caution" for record in replay_records),
            "uncertain_frames": sum(record["risk"] == "uncertain" for record in replay_records),
            "radar_confirmed_frames": sum(record["evidence_status"] == "radar_confirmed" for record in replay_records),
            "camera_only_frames": sum(record["evidence_status"] == "camera_only" for record in replay_records),
        }
        csv.DictWriter(summary_stream, fieldnames=columns).writeheader()
        csv.DictWriter(summary_stream, fieldnames=columns).writerow(row)
    print((args.output_dir / "annotated.mp4").resolve())
    print(decisions_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
