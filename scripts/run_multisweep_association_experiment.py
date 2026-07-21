"""Compare one radar sweep with ego-motion-compensated three-sweep association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from guardian_perception.radar import RadarPoint, cluster_radar_points
from render_association_gate_experiment import matching_annotation, oracle_object_points
from render_nuscenes_fusion_preview import project_camera, read_pcd, rotation_matrix, transform_global_to_camera, transform_sensor_to_global
from run_nuscenes_rule_based_fusion_demo import choose_candidate, panel, text_lines


def collect_sweeps(frame: dict, root: Path, count: int, *, motion_compensate: bool = False) -> tuple[np.ndarray, np.ndarray, list[RadarPoint], np.ndarray]:
    """Return global points, image pixels, reference-ego points and quality flags."""

    reference_ego = frame["camera"]["ego_pose"]
    reference_rotation = rotation_matrix(reference_ego["rotation"])
    reference_translation = np.asarray(reference_ego["translation"])[:, None]
    global_batches, quality_batches, velocity_batches = [], [], []
    reference_timestamp_us = int(frame["radar_sweeps"][0]["timestamp_us"])
    for sweep in frame["radar_sweeps"][:count]:
        raw = read_pcd(root / sweep["file"])
        sensor = np.vstack((raw["x"], raw["y"], raw["z"]))
        global_points = transform_sensor_to_global(sensor, sweep["calibration"], sweep["ego_pose"])
        if motion_compensate and sweep is not frame["radar_sweeps"][0]:
            # Experimental convention: compensated Doppler velocity is transformed
            # from the radar sensor frame into global coordinates, then propagated
            # from the older sweep to the reference sweep time. This must later be
            # validated against a sensor-specific velocity convention on real hardware.
            velocity_sensor = np.vstack((raw["vx_comp"], raw["vy_comp"], np.zeros(len(raw))))
            velocity_global = rotation_matrix(sweep["ego_pose"]["rotation"]) @ rotation_matrix(sweep["calibration"]["rotation"]) @ velocity_sensor
            global_points = global_points + velocity_global * ((reference_timestamp_us - int(sweep["timestamp_us"])) / 1_000_000)
        global_batches.append(global_points)
        quality_batches.append(raw["is_quality_valid"].astype(bool))
        velocity_batches.append(np.vstack((raw["vx_comp"], raw["vy_comp"])))
    global_points = np.hstack(global_batches)
    reference_ego_points = reference_rotation.T @ (global_points - reference_translation)
    pixels, visible = project_camera(
        transform_global_to_camera(global_points, frame["camera"]["calibration"], frame["camera"]["ego_pose"]),
        frame["camera"]["calibration"]["camera_intrinsic"],
    )
    velocities = np.hstack(velocity_batches)
    qualities = np.concatenate(quality_batches)
    radar_points = [RadarPoint(float(point[0]), float(point[1]), float(point[2]), float(velocity[0]), float(velocity[1]), bool(quality)) for point, velocity, quality in zip(reference_ego_points.T, velocities.T, qualities, strict=True)]
    return global_points, pixels, radar_points, visible


def select_cluster(points: list[RadarPoint], pixels: np.ndarray, visible: np.ndarray, bbox: tuple[int, int, int, int]):
    clusters = cluster_radar_points(points)
    x1, y1, x2, y2 = bbox
    supported = []
    for cluster in clusters:
        support = sum(visible[i] and x1 - 12 <= pixels[0, i] <= x2 + 12 and y1 - 12 <= pixels[1, i] <= y2 + 12 for i in cluster.point_indexes) / len(cluster.point_indexes)
        if support >= 0.60:
            supported.append((cluster.range_m, cluster, support))
    return min(supported, default=(None, None, 0.0), key=lambda item: float("inf") if item[0] is None else item[0]), clusters


def metrics(cluster, oracle: set[int]) -> tuple[float, float]:
    if cluster is None:
        return 0.0, 0.0
    selected = set(cluster.point_indexes)
    return len(selected & oracle) / len(selected), len(selected & oracle) / max(1, len(oracle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/multisweep_temporal_association.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/nuscenes_fusion/multisweep_temporal_association_preview.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes_fusion/multisweep_temporal_association.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/multisweep_temporal_association_summary.json"))
    parser.add_argument("--motion-compensate", action="store_true", help="Propagate older sweeps with compensated Doppler velocity.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for path in (args.output, args.preview, args.report, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    writer, records = None, []
    with args.report.open("w", encoding="utf-8") as report:
        for index, frame in enumerate(manifest["frames"]):
            image = cv2.imread(str(args.data_root / frame["camera"]["file"]))
            all_global, all_pixels, all_points, all_visible = collect_sweeps(frame, args.data_root, 3, motion_compensate=args.motion_compensate)
            current_global, current_pixels, current_points, current_visible = collect_sweeps(frame, args.data_root, 1)
            target = choose_candidate(frame, all_pixels, all_visible, (image.shape[1], image.shape[0]))
            record = {"frame": index, "scene": frame["scene"], "current_points": len(current_points), "multisweep_points": len(all_points), "target_found": target is not None}
            overlay = image.copy()
            info = panel(960, 540, "TEMPORAL MULTI-SWEEP EXPERIMENT")
            if target is not None:
                annotation = matching_annotation(frame, target.bbox, (image.shape[1], image.shape[0]))
                x1, y1, x2, y2 = target.bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 3)
                (_, single_cluster, single_support), _ = select_cluster(current_points, current_pixels, current_visible, target.bbox)
                (_, multi_cluster, multi_support), _ = select_cluster(all_points, all_pixels, all_visible, target.bbox)
                single_oracle = set(np.flatnonzero(oracle_object_points(current_global, annotation)).tolist())
                multi_oracle = set(np.flatnonzero(oracle_object_points(all_global, annotation)).tolist())
                single_precision, single_recall = metrics(single_cluster, single_oracle)
                multi_precision, multi_recall = metrics(multi_cluster, multi_oracle)
                if single_cluster:
                    for point_id in single_cluster.point_indexes:
                        if current_visible[point_id]:
                            x, y = np.rint(current_pixels[:, point_id]).astype(int)
                            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
                if multi_cluster:
                    for point_id in multi_cluster.point_indexes:
                        if all_visible[point_id]:
                            x, y = np.rint(all_pixels[:, point_id]).astype(int)
                            cv2.circle(overlay, (x, y), 3, (255, 255, 0), -1)
                record.update({"single_precision": single_precision, "single_recall": single_recall, "multi_precision": multi_precision, "multi_recall": multi_recall, "single_selected": single_cluster is not None, "multi_selected": multi_cluster is not None})
                text_lines(info, [
                    (f"{frame['scene']} | target: {target.category} | MAGENTA box = oracle camera", (255, 80, 255)),
                    (f"single sweep: {len(current_points)} points | RED cluster: {'none' if single_cluster is None else len(single_cluster.point_indexes)}", (0, 0, 255)),
                    (f"single eval: precision {single_precision:.0%} | recall {single_recall:.0%} | support {single_support:.2f}", (0, 0, 255)),
                    (f"3 sweeps: {len(all_points)} points | CYAN cluster: {'none' if multi_cluster is None else len(multi_cluster.point_indexes)}", (0, 255, 255)),
                    (f"multi eval: precision {multi_precision:.0%} | recall {multi_recall:.0%} | support {multi_support:.2f}", (0, 255, 255)),
                    ("", (255, 255, 255)),
                    ("Points are ego-motion-compensated into the current camera frame.", (0, 145, 255)),
                    (("Doppler/object-motion compensation: ENABLED (experimental)." if args.motion_compensate else "Dynamic-object motion compensation is not enabled."), (0, 130, 255)),
                ])
            left = cv2.resize(overlay, (960, 540))
            cv2.rectangle(left, (0, 0), (960, 48), (14, 14, 14), -1)
            cv2.putText(left, "RED = single-sweep cluster | CYAN = multi-sweep cluster", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
            lower = panel(960, 540, "INTERPRETATION")
            text_lines(lower, [
                ("More points help only if association precision does not collapse.", (0, 255, 255)),
                ("If recall rises while precision falls, accumulation adds clutter.", (0, 130, 255)),
                ("Next step if needed: Doppler/object-motion compensation or learned score.", (255, 220, 80)),
            ])
            rendered = cv2.vconcat([cv2.hconcat([left, info]), cv2.hconcat([np.full((540, 960, 3), (24, 24, 24), dtype=np.uint8), lower])])
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (1920, 1080))
            writer.write(rendered)
            if index == 0:
                cv2.imwrite(str(args.preview), rendered)
            records.append(record)
            report.write(json.dumps(record) + "\n")
    assert writer is not None
    writer.release()
    valid = [row for row in records if "single_precision" in row]
    summary = {"frames": len(records), "motion_compensation": args.motion_compensate, "mean_density_multiplier": float(np.mean([row["multisweep_points"] / row["current_points"] for row in valid])), "single": {"precision": float(np.mean([row["single_precision"] for row in valid])), "recall": float(np.mean([row["single_recall"] for row in valid]))}, "three_sweeps": {"precision": float(np.mean([row["multi_precision"] for row in valid])), "recall": float(np.mean([row["multi_recall"] for row in valid]))}}
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(args.summary.resolve())


if __name__ == "__main__":
    main()
