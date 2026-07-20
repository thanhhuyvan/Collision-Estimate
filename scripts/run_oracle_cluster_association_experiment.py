"""Run the improved cluster-and-track association baseline with GT boxes as camera oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from guardian_perception.radar import RadarClusterTracker, RadarPoint, cluster_radar_points
from render_association_gate_experiment import matching_annotation, oracle_object_points
from render_nuscenes_fusion_preview import draw_bev, project_camera, read_pcd, transform_global_to_camera, transform_sensor_to_global
from run_nuscenes_rule_based_fusion_demo import choose_candidate, panel, text_lines


def cluster_score(cluster_indexes: tuple[int, ...], pixels: np.ndarray, visible: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """2D-only runtime score: fraction of a cluster supporting the camera box."""

    x1, y1, x2, y2 = bbox
    padding = 12
    inside = sum(visible[index] and x1 - padding <= pixels[0, index] <= x2 + padding and y1 - padding <= pixels[1, index] <= y2 + padding for index in cluster_indexes)
    return inside / len(cluster_indexes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Defaults to the original single-scene manifest under data-root.")
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_cluster_association_v2.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_cluster_association_v2_preview.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_cluster_association_v2.jsonl"))
    parser.add_argument("--strategy", choices=("projection_only", "frontmost_supported"), default="frontmost_supported")
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    manifest = json.loads((args.manifest or args.data_root / "fusion_subset_manifest.json").read_text(encoding="utf-8"))
    for path in (args.output, args.preview, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    tracker = RadarClusterTracker(max_match_distance_m=4.0, max_age_ms=800)
    previous_scene: str | None = None
    writer = None
    with args.report.open("w", encoding="utf-8") as report:
        for frame_index, frame in enumerate(manifest["frames"]):
            scene = frame.get("scene")
            if scene != previous_scene:
                # Never carry temporal state across independent drives.
                tracker = RadarClusterTracker(max_match_distance_m=4.0, max_age_ms=800)
                previous_scene = scene
            image = cv2.imread(str(args.data_root / frame["camera"]["file"]))
            raw_radar = read_pcd(args.data_root / frame["radar"]["file"])
            radar_points = [RadarPoint(float(point["x"]), float(point["y"]), float(point["z"]), float(point["vx_comp"]), float(point["vy_comp"]), bool(point["is_quality_valid"])) for point in raw_radar]
            clusters = cluster_radar_points(radar_points, radius_m=1.5, min_points=2)
            tracks = tracker.update(clusters, int(frame["timestamp_us"] // 1000))
            coordinates = np.vstack((raw_radar["x"], raw_radar["y"], raw_radar["z"]))
            global_points = transform_sensor_to_global(coordinates, frame["radar"]["calibration"], frame["radar"]["ego_pose"])
            camera_points = transform_global_to_camera(global_points, frame["camera"]["calibration"], frame["camera"]["ego_pose"])
            pixels, visible = project_camera(camera_points, frame["camera"]["calibration"]["camera_intrinsic"])
            target = choose_candidate(frame, pixels, visible, (image.shape[1], image.shape[0]))
            left = image.copy()
            info = panel(960, 540, "IMPROVED BASELINE | CLUSTER + TEMPORAL ASSOCIATION")
            result: dict[str, object] = {"frame": frame_index, "scene": scene, "cluster_count": len(clusters), "target_found": target is not None}
            if target is not None:
                annotation = matching_annotation(frame, target.bbox, (image.shape[1], image.shape[0]))
                scored = [(cluster_score(cluster.point_indexes, pixels, visible, target.bbox), cluster) for cluster in clusters]
                supported = [(score, cluster) for score, cluster in scored if score >= 0.60]
                if args.strategy == "projection_only":
                    best_score, best_cluster = max(supported, default=(0.0, None), key=lambda item: item[0])
                else:
                    # When several clusters fully support a large 2D box, prefer the
                    # nearest supported return. This is a conservative visible-surface
                    # heuristic, not a claim that all closest returns are the object.
                    best_score, best_cluster = min(supported, default=(0.0, None), key=lambda item: (item[1].range_m, -item[0]))
                selected = best_cluster
                oracle_mask = oracle_object_points(global_points, annotation)
                x1, y1, x2, y2 = target.bbox
                cv2.rectangle(left, (x1, y1), (x2, y2), (255, 0, 255), 3)
                for cluster in clusters:
                    colour = (130, 130, 130)
                    if cluster is selected:
                        colour = (255, 255, 0)
                    for point_index in cluster.point_indexes:
                        if visible[point_index]:
                            x, y = np.rint(pixels[:, point_index]).astype(int)
                            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                                cv2.circle(left, (x, y), 4, colour, -1)
                for point_index in np.flatnonzero(oracle_mask):
                    if visible[point_index]:
                        x, y = np.rint(pixels[:, point_index]).astype(int)
                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            cv2.circle(left, (x, y), 2, (0, 255, 0), -1)
                oracle_ids = set(np.flatnonzero(oracle_mask).tolist())
                selected_ids = set() if selected is None else set(selected.point_indexes)
                precision = len(selected_ids & oracle_ids) / len(selected_ids) if selected_ids else 0.0
                recall = len(selected_ids & oracle_ids) / len(oracle_ids) if oracle_ids else 0.0
                selected_range = None if selected is None else selected.range_m
                track_id, track_age = (-1, 0) if selected is None else tracks[selected.cluster_id]
                result.update({"strategy": args.strategy, "target_category": target.category, "selected_cluster": None if selected is None else selected.cluster_id, "selected_cluster_track": None if selected is None else track_id, "track_age_frames": track_age, "association_score": best_score, "selected_range_m": selected_range, "oracle_point_count": len(oracle_ids), "oracle_precision": precision, "oracle_recall": recall})
                text_lines(info, [
                    (f"Target: {target.category} | MAGENTA = oracle camera box", (255, 80, 255)),
                    (f"{len(clusters)} radar clusters after quality filter + BEV clustering", (220, 220, 220)),
                    ("CYAN = selected cluster | GREEN = GT oracle points (evaluation only)", (0, 255, 255)),
                    (f"strategy: {args.strategy} | selected: {'none' if selected is None else selected.cluster_id} | score: {best_score:.2f}", (0, 255, 255)),
                    (f"cluster track: {'none' if selected is None else f'#{track_id}, age {track_age} frames'}", (0, 255, 255)),
                    (f"selected range: {'none' if selected_range is None else f'{selected_range:.1f}m'}", (0, 255, 255)),
                    (f"EVAL precision: {precision:.0%} | recall: {recall:.0%}", (0, 255, 0)),
                    ("", (255, 255, 255)),
                    ("Runtime uses only box + projection + cluster support + temporal age.", (0, 145, 255)),
                    ("GT green dots score the choice; they never select the cluster.", (0, 145, 255)),
                ])
            left = cv2.resize(left, (960, 540))
            cv2.rectangle(left, (0, 0), (960, 48), (14, 14, 14), -1)
            cv2.putText(left, "CAMERA + RADAR CLUSTERS", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            lower = panel(960, 540, "WHAT IMPROVED")
            text_lines(lower, [
                ("Before: all radar points within a 2D box were fused.", (0, 0, 255)),
                ("Now: filter → cluster → supported cluster → nearest tie-break → track age.", (0, 255, 255)),
                ("Still missing: depth gate, robust velocity gate, multi-object assignment.", (255, 220, 80)),
                ("This remains an oracle-box experiment until YOLO is introduced.", (255, 80, 255)),
            ])
            rendered = cv2.vconcat([cv2.hconcat([left, info]), cv2.hconcat([draw_bev(raw_radar, (960, 540)), lower])])
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080))
            writer.write(rendered)
            if frame_index == 0:
                cv2.imwrite(str(args.preview), rendered)
            report.write(json.dumps(result) + "\n")
    assert writer is not None
    writer.release()
    print(args.output.resolve())
    print(args.preview.resolve())
    print(args.report.resolve())


if __name__ == "__main__":
    main()
