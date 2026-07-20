"""Show an explainable camera-radar late-fusion pass on a nuScenes mini scene.

This intentionally uses nuScenes GT 3D boxes as *oracle camera detections* so the
visualisation can isolate the geometric association and fusion logic.  A real run
replaces ``oracle detection`` with YOLO boxes; GT is never runtime input.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from guardian_perception import CameraMeasurement, RadarMeasurement, ReliabilityAwareFuser
from render_nuscenes_fusion_preview import (
    project_camera,
    read_pcd,
    rotation_matrix,
    transform_global_to_camera,
    transform_sensor_to_global,
)


@dataclass
class Candidate:
    category: str
    bbox: tuple[int, int, int, int]
    camera_range_m: float
    radar_indexes: np.ndarray
    score: float


def annotation_bbox(annotation: dict, camera: dict, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Project a labelled 3D box to an image rectangle, solely as a detector stand-in."""

    width, height = image_size
    box_width, box_length, box_height = annotation["size_m"]
    corners = np.array(
        [
            [box_length / 2, box_length / 2, -box_length / 2, -box_length / 2, box_length / 2, box_length / 2, -box_length / 2, -box_length / 2],
            [box_width / 2, -box_width / 2, -box_width / 2, box_width / 2, box_width / 2, -box_width / 2, -box_width / 2, box_width / 2],
            [box_height / 2, box_height / 2, box_height / 2, box_height / 2, -box_height / 2, -box_height / 2, -box_height / 2, -box_height / 2],
        ]
    )
    global_points = rotation_matrix(annotation["rotation"]) @ corners + np.asarray(annotation["translation_m"])[:, None]
    ego_rotation = rotation_matrix(camera["ego_pose"]["rotation"])
    ego_translation = np.asarray(camera["ego_pose"]["translation"])
    calibration_rotation = rotation_matrix(camera["calibration"]["rotation"])
    calibration_translation = np.asarray(camera["calibration"]["translation"])
    ego_points = ego_rotation.T @ (global_points - ego_translation[:, None])
    camera_points = calibration_rotation.T @ (ego_points - calibration_translation[:, None])
    pixels, visible = project_camera(camera_points, camera["calibration"]["camera_intrinsic"])
    if not np.all(visible):
        return None
    x1, y1 = np.floor(np.min(pixels, axis=1)).astype(int)
    x2, y2 = np.ceil(np.max(pixels, axis=1)).astype(int)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return x1, y1, x2, y2


def choose_candidate(frame: dict, pixels: np.ndarray, valid: np.ndarray, image_size: tuple[int, int]) -> Candidate | None:
    """Apply a deliberately simple association gate and pick an explainable target."""

    width, height = image_size
    candidates: list[Candidate] = []
    for annotation in frame["annotations"]:
        if not annotation["category"].startswith("vehicle."):
            continue
        bbox = annotation_bbox(annotation, frame["camera"], image_size)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        # A small expansion admits radar points that are geometrically near a box.
        padding = 14
        point_mask = valid & (pixels[0] >= x1 - padding) & (pixels[0] <= x2 + padding) & (pixels[1] >= y1 - padding) & (pixels[1] <= y2 + padding)
        indexes = np.flatnonzero(point_mask)
        focal_y = float(frame["camera"]["calibration"]["camera_intrinsic"][1][1])
        assumed_height = 1.6  # rough vehicle prior; a monocular range proxy, not truth.
        camera_range = focal_y * assumed_height / max(1, y2 - y1)
        centre_penalty = abs(((x1 + x2) / 2) / width - 0.5)
        # Prefer an object with radar support, in front of the camera, and visibly large.
        score = len(indexes) * 10 + min(5, (y2 - y1) / 25) - centre_penalty * 3
        candidates.append(Candidate(annotation["category"], bbox, camera_range, indexes, score))
    return max(candidates, key=lambda candidate: candidate.score) if candidates else None


def panel(width: int, height: int, title: str) -> np.ndarray:
    result = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
    cv2.rectangle(result, (0, 0), (width, 54), (14, 14, 14), -1)
    cv2.putText(result, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def text_lines(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], start_y: int = 92) -> None:
    y = start_y
    for line, colour in lines:
        cv2.putText(image, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, colour, 1, cv2.LINE_AA)
        y += 35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_rule_based_fusion.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_rule_based_fusion_preview.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_rule_based_fusion.jsonl"))
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    manifest = json.loads((args.manifest or args.data_root / "fusion_subset_manifest.json").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    fuser = ReliabilityAwareFuser()
    with args.report.open("w", encoding="utf-8") as report:
        for frame_index, frame in enumerate(manifest["frames"]):
            camera = frame["camera"]
            radar_record = frame["radar"]
            image = cv2.imread(str(args.data_root / camera["file"]))
            if image is None:
                raise RuntimeError(f"cannot read {camera['file']}")
            radar = read_pcd(args.data_root / radar_record["file"])
            radar_sensor_points = np.vstack((radar["x"], radar["y"], radar["z"]))
            global_points = transform_sensor_to_global(radar_sensor_points, radar_record["calibration"], radar_record["ego_pose"])
            camera_points = transform_global_to_camera(global_points, camera["calibration"], camera["ego_pose"])
            pixels, valid = project_camera(camera_points, camera["calibration"]["camera_intrinsic"])
            candidate = choose_candidate(frame, pixels, valid, (image.shape[1], image.shape[0]))
            camera_panel = image.copy()
            measurement_panel = panel(960, 540, "OBJECT-LEVEL RULE-BASED FUSION")
            result_panel = panel(960, 540, "WHY THIS IS NOT A TRAINED MODEL")
            record: dict[str, object] = {"frame": frame_index, "target_found": candidate is not None}
            if candidate is None:
                text_lines(measurement_panel, [("No eligible labelled vehicle in view.", (0, 130, 255))])
                text_lines(result_panel, [("Camera fallback only.", (0, 130, 255))])
            else:
                x1, y1, x2, y2 = candidate.bbox
                cv2.rectangle(camera_panel, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(camera_panel, "oracle camera box (replace with YOLO)", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1, cv2.LINE_AA)
                for point_index in candidate.radar_indexes:
                    x, y = np.rint(pixels[:, point_index]).astype(int)
                    cv2.circle(camera_panel, (x, y), 5, (0, 255, 0), -1)
                radar_measurement = None
                if len(candidate.radar_indexes):
                    selected = radar[candidate.radar_indexes]
                    radar_range = float(np.median(np.linalg.norm(np.vstack((selected["x"], selected["y"], selected["z"])), axis=0)))
                    # x is forward in this radar frame; negative compensated vx means distance closing.
                    closing_speed = max(0.0, -float(np.median(selected["vx_comp"])))
                    quality = float(np.mean(selected["is_quality_valid"].astype(float)))
                    association = min(1.0, 0.40 + 0.12 * len(selected))
                    radar_measurement = RadarMeasurement(f"radar-frame-{frame_index}", radar_range, closing_speed, quality, association)
                camera_measurement = CameraMeasurement(f"oracle-track-{frame_index}", candidate.camera_range_m, 0.0, 0.90, 0.85)
                fused = fuser.fuse(camera_measurement, radar_measurement)
                record.update({
                    "category": candidate.category,
                    "associated_radar_points": int(len(candidate.radar_indexes)),
                    "camera_range_proxy_m": round(camera_measurement.range_m, 2),
                    "radar_range_m": None if radar_measurement is None else round(radar_measurement.range_m, 2),
                    "radar_closing_speed_mps": None if radar_measurement is None else round(radar_measurement.closing_speed_mps, 2),
                    "fused_range_m": round(fused.range_m, 2), "fused_ttc_s": None if fused.ttc_s is None else round(fused.ttc_s, 2),
                    "radar_used": fused.radar_used, "reliability": round(fused.reliability, 2), "explanation": fused.explanation,
                })
                radar_line = "no associated radar return → camera fallback" if radar_measurement is None else f"radar range: {radar_measurement.range_m:.1f}m | closing speed: {radar_measurement.closing_speed_mps:.1f}m/s"
                text_lines(measurement_panel, [
                    (f"Target: {candidate.category}", (255, 80, 255)),
                    ("MAGENTA box: camera detector output (oracle in this demo)", (255, 80, 255)),
                    (f"camera range proxy: {camera_measurement.range_m:.1f}m  (box-height prior)", (255, 220, 80)),
                    (f"GREEN dots: {len(candidate.radar_indexes)} associated radar returns", (0, 255, 0)),
                    (radar_line, (0, 220, 170)),
                    (f"association confidence: {0.0 if radar_measurement is None else radar_measurement.association_confidence:.2f}  |  gate: 0.50", (220, 220, 220)),
                ])
                text_lines(result_panel, [
                    (f"FUSED range: {fused.range_m:.1f}m", (215, 140, 255)),
                    (f"FUSED TTC: {'not closing' if fused.ttc_s is None else f'{fused.ttc_s:.1f}s'}", (215, 140, 255)),
                    (f"weights — camera {fused.camera_weight:.0%} | radar {fused.radar_weight:.0%}", (215, 140, 255)),
                    (f"reliability: {fused.reliability:.2f} | {fused.explanation}", (215, 140, 255)),
                    ("", (255, 255, 255)),
                    ("Camera provides: object/class/box and a rough range proxy.", (255, 220, 80)),
                    ("Radar provides: more accurate range + relative velocity.", (0, 220, 0)),
                    ("Rule: only use radar after geometric association passes gate.", (0, 145, 255)),
                    ("GT selects the camera box here solely to demonstrate the data flow.", (0, 130, 255)),
                ])
            camera_panel = cv2.resize(camera_panel, (960, 540))
            cv2.rectangle(camera_panel, (0, 0), (960, 48), (14, 14, 14), -1)
            cv2.putText(camera_panel, "CAMERA + ASSOCIATED RADAR POINTS", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            rendered = cv2.vconcat([cv2.hconcat([camera_panel, measurement_panel]), cv2.hconcat([panel(960, 540, "RAW RADAR REMAINS SPARSE"), result_panel])])
            # Use a cropped radar BEV from the existing display only when preserving the key data.
            from render_nuscenes_fusion_preview import draw_bev
            rendered[540:1080, 0:960] = draw_bev(radar, (960, 540))
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot write {args.output}")
            if frame_index == 0 and not cv2.imwrite(str(args.preview), rendered):
                raise RuntimeError(f"cannot write {args.preview}")
            writer.write(rendered)
            report.write(json.dumps(record) + "\n")
    assert writer is not None
    writer.release()
    print(args.output.resolve())
    print(args.preview.resolve())
    print(args.report.resolve())


if __name__ == "__main__":
    main()
