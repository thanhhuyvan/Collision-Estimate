"""Compare naive 2D radar association against an oracle 3D GT gate.

The 3D gate is evaluation-only.  It makes false 2D associations visible and
defines what a runtime depth/motion/cluster gate should approximate later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from render_nuscenes_fusion_preview import (
    draw_bev,
    project_camera,
    read_pcd,
    rotation_matrix,
    transform_global_to_camera,
    transform_sensor_to_global,
)
from run_nuscenes_rule_based_fusion_demo import annotation_bbox, choose_candidate, panel, text_lines


def matching_annotation(frame: dict, target_bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> dict:
    for annotation in frame["annotations"]:
        if annotation["category"].startswith("vehicle.") and annotation_bbox(annotation, frame["camera"], image_size) == target_bbox:
            return annotation
    raise RuntimeError("target annotation was not found")


def oracle_object_points(global_points: np.ndarray, annotation: dict, margin_m: float = 0.0) -> np.ndarray:
    """Return radar points within the labelled 3D cuboid (evaluation only)."""

    local = rotation_matrix(annotation["rotation"]).T @ (global_points - np.asarray(annotation["translation_m"])[:, None])
    width, length, height = annotation["size_m"]
    return (
        (np.abs(local[0]) <= length / 2 + margin_m)
        & (np.abs(local[1]) <= width / 2 + margin_m)
        & (np.abs(local[2]) <= height / 2 + margin_m)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_association_gate_experiment.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_association_gate_experiment_preview.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_association_gate_experiment.jsonl"))
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    manifest = json.loads((args.data_root / "fusion_subset_manifest.json").read_text(encoding="utf-8"))
    for path in (args.output, args.preview, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    with args.report.open("w", encoding="utf-8") as report:
        for index, frame in enumerate(manifest["frames"]):
            image = cv2.imread(str(args.data_root / frame["camera"]["file"]))
            radar = read_pcd(args.data_root / frame["radar"]["file"])
            points = np.vstack((radar["x"], radar["y"], radar["z"]))
            global_points = transform_sensor_to_global(points, frame["radar"]["calibration"], frame["radar"]["ego_pose"])
            camera_points = transform_global_to_camera(global_points, frame["camera"]["calibration"], frame["camera"]["ego_pose"])
            pixels, visible = project_camera(camera_points, frame["camera"]["calibration"]["camera_intrinsic"])
            target = choose_candidate(frame, pixels, visible, (image.shape[1], image.shape[0]))
            left = image.copy()
            info = panel(960, 540, "ASSOCIATION GATE EXPERIMENT")
            result: dict[str, object] = {"frame": index, "target_found": target is not None}
            if target is not None:
                annotation = matching_annotation(frame, target.bbox, (image.shape[1], image.shape[0]))
                oracle_mask = oracle_object_points(global_points, annotation)
                naive_ids = target.radar_indexes
                oracle_ids = np.flatnonzero(oracle_mask)
                x1, y1, x2, y2 = target.bbox
                cv2.rectangle(left, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # Red is everything admitted by a 2D box. Green is the subset truly within
                # the labelled 3D object; green is intentionally drawn on top.
                for point_index in naive_ids:
                    x, y = np.rint(pixels[:, point_index]).astype(int)
                    cv2.circle(left, (x, y), 5, (0, 0, 255), -1)
                for point_index in oracle_ids:
                    if visible[point_index]:
                        x, y = np.rint(pixels[:, point_index]).astype(int)
                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            cv2.circle(left, (x, y), 6, (0, 255, 0), -1)
                naive_range = float(np.median(np.linalg.norm(points[:, naive_ids], axis=0))) if len(naive_ids) else None
                oracle_range = float(np.median(np.linalg.norm(points[:, oracle_ids], axis=0))) if len(oracle_ids) else None
                ego_position = np.asarray(frame["radar"]["ego_pose"]["translation"])
                center_range = float(np.linalg.norm(np.asarray(annotation["translation_m"]) - ego_position))
                contamination = 1 - len(oracle_ids) / max(1, len(naive_ids))
                result.update({"category": target.category, "naive_2d_point_count": int(len(naive_ids)), "oracle_3d_point_count": int(len(oracle_ids)), "naive_median_range_m": naive_range, "oracle_median_range_m": oracle_range, "gt_center_range_m": center_range, "naive_2d_contamination_fraction": contamination})
                text_lines(info, [
                    (f"Target: {target.category}", (255, 255, 255)),
                    (f"RED: naive 2D box gate = {len(naive_ids)} radar points", (0, 0, 255)),
                    (f"GREEN: points inside GT 3D object = {len(oracle_ids)}", (0, 255, 0)),
                    (f"2D gate contamination: {contamination:.0%}", (0, 130, 255)),
                    (f"naive median range: {'none' if naive_range is None else f'{naive_range:.1f}m'}", (0, 0, 255)),
                    (f"oracle median range: {'none' if oracle_range is None else f'{oracle_range:.1f}m'}", (0, 255, 0)),
                    (f"GT object center range: {center_range:.1f}m", (255, 220, 80)),
                    ("", (255, 255, 255)),
                    ("The 3D green gate is NOT usable at runtime.", (0, 145, 255)),
                    ("It is an oracle test: runtime must approximate it with", (0, 145, 255)),
                    ("radar clustering + depth consistency + temporal motion.", (0, 145, 255)),
                ])
            else:
                text_lines(info, [("No eligible target.", (0, 130, 255))])
            left = cv2.resize(left, (960, 540))
            cv2.rectangle(left, (0, 0), (960, 48), (14, 14, 14), -1)
            cv2.putText(left, "CAMERA VIEW | red=2D gate, green=oracle 3D object", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            rendered = cv2.vconcat([cv2.hconcat([left, info]), cv2.hconcat([draw_bev(radar, (960, 540)), panel(960, 540, "INTERPRETATION")])])
            text_lines(rendered[540:1080, 960:1920], [
                ("This experiment tests the earlier diagnosis.", (255, 255, 255)),
                ("If green is a much smaller subset than red, 2D overlap", (0, 130, 255)),
                ("alone is not an acceptable radar-to-object association rule.", (0, 130, 255)),
                ("", (255, 255, 255)),
                ("Calibration is tested by systematic projection displacement.", (255, 220, 80)),
                ("Association is tested by red-vs-green contamination.", (0, 255, 0)),
                ("These are separate failure modes.", (215, 140, 255)),
            ])
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080))
            writer.write(rendered)
            if index == 0:
                cv2.imwrite(str(args.preview), rendered)
            report.write(json.dumps(result) + "\n")
    assert writer is not None
    writer.release()
    print(args.output.resolve())
    print(args.preview.resolve())
    print(args.report.resolve())


if __name__ == "__main__":
    main()
