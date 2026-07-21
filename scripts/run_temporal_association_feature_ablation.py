"""Controlled base-vs-temporal association scorer experiment.

Both variants receive exactly the same camera-object/radar-cluster candidates.
Only the temporal scorer additionally sees range-residual, camera box-growth and
Doppler features.  GT 3D boxes create labels after feature extraction only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from guardian_perception.radar import RadarClusterTracker, cluster_radar_points
from render_association_gate_experiment import oracle_object_points
from run_learned_association_ablation import candidate_features, fit_logistic, probability
from run_multisweep_association_experiment import collect_sweeps
from run_nuscenes_rule_based_fusion_demo import annotation_bbox


def lead_instance(frame: dict, max_range_m: float = 60.0, half_width_m: float = 1.75) -> str | None:
    from evaluate_collision_target_association import ego_position

    candidates = []
    for annotation in frame["annotations"]:
        if not annotation["category"].startswith("vehicle."):
            continue
        x_m, y_m, _ = ego_position(annotation, frame)
        if 2.0 < x_m <= max_range_m and abs(y_m) <= half_width_m + annotation["size_m"][0] / 2:
            candidates.append((x_m, annotation["instance_token"]))
    return min(candidates, default=(None, None))[1]


def build_rows(manifest: dict, root: Path) -> list[dict]:
    rows, tracker, active_scene, history = [], RadarClusterTracker(), None, {}
    for frame_index, frame in enumerate(manifest["frames"]):
        if frame["scene"] != active_scene:
            tracker, active_scene, history = RadarClusterTracker(), frame["scene"], {}
        global_points, pixels, radar_points, visible = collect_sweeps(frame, root, 1)
        clusters = cluster_radar_points(radar_points)
        ages = tracker.update(clusters, int(frame["timestamp_us"] // 1000))
        timestamp = int(frame["timestamp_us"])
        pending = {}
        for annotation in frame["annotations"]:
            if not annotation["category"].startswith("vehicle."):
                continue
            bbox = annotation_bbox(annotation, frame["camera"], (1600, 900))
            if bbox is None:
                continue
            instance = annotation["instance_token"]
            height = bbox[3] - bbox[1]
            previous = history.get(instance)
            growth = 0.0 if previous is None else max(0.0, (height - previous[0]) / max(0.001, (timestamp - previous[1]) / 1_000_000))
            pending[instance] = (height, timestamp)
            oracle = set(np.flatnonzero(oracle_object_points(global_points, annotation)).tolist())
            focal_y = float(frame["camera"]["calibration"]["camera_intrinsic"][1][1])
            camera_range_proxy = focal_y * 1.6 / max(1.0, height)
            for cluster in clusters:
                result = candidate_features(cluster, pixels, visible, bbox, ages[cluster.cluster_id][1])
                if result is None or result[0][0] < 0.15:
                    continue
                base_features, _ = result
                overlap = len(set(cluster.point_indexes) & oracle) / len(cluster.point_indexes)
                state = "positive" if overlap >= 0.5 else "negative" if overlap <= 0.1 else "ignore"
                closing = min(1.0, max(0.0, -cluster.vx_mps) / 15.0)
                growth_norm = min(1.0, growth / 160.0)
                temporal_features = np.concatenate((base_features, np.array([
                    min(1.0, abs(cluster.range_m - camera_range_proxy) / 40.0),
                    closing, growth_norm, closing * growth_norm,
                ])))
                rows.append({"scene": frame["scene"], "frame": frame_index, "instance": instance, "cluster": cluster.cluster_id, "label_state": state, "base": base_features, "temporal": temporal_features})
        history.update(pending)
    return rows


def evaluate(rows: list[dict], frames: dict[int, dict], weights: np.ndarray, mean: np.ndarray, std: np.ndarray, key: str, threshold: float) -> dict:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[row["frame"]].append(row)
    targets = assignments = correct = wrong = ambiguous = 0
    for frame_index, frame in frames.items():
        instance = lead_instance(frame)
        if instance is None:
            continue
        targets += 1
        candidates = [row for row in by_frame.get(frame_index, []) if row["instance"] == instance]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: probability(row[key], weights, mean, std))
        if probability(best[key], weights, mean, std) < threshold:
            continue
        assignments += 1
        correct += best["label_state"] == "positive"
        wrong += best["label_state"] == "negative"
        ambiguous += best["label_state"] == "ignore"
    return {"lead_frames": targets, "assignments": assignments, "correct": correct, "wrong": wrong, "ambiguous": ambiguous, "precision": correct / assignments if assignments else 0.0, "safe_success": correct / targets if targets else 0.0}


def fit(rows: list[dict], key: str):
    clean = [row for row in rows if row["label_state"] != "ignore"]
    return fit_logistic(np.stack([row[key] for row in clean]), np.array([row["label_state"] == "positive" for row in clean], dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/temporal_feature_ablation_40f_1s.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = build_rows(manifest, args.data_root)
    scenes = list(manifest["scenes"])
    frame_map = {index: frame for index, frame in enumerate(manifest["frames"])}
    results = {}
    for test_scene in scenes:
        validation_scene = next(scene for scene in scenes if scene not in {test_scene, "scene-1077"}) if test_scene == "scene-1077" else "scene-1077"
        train = [row for row in rows if row["scene"] not in {test_scene, validation_scene}]
        validation = [row for row in rows if row["scene"] == validation_scene]
        test = [row for row in rows if row["scene"] == test_scene]
        validation_frames = {index: frame for index, frame in frame_map.items() if frame["scene"] == validation_scene}
        test_frames = {index: frame for index, frame in frame_map.items() if frame["scene"] == test_scene}
        variants = {}
        for key in ("base", "temporal"):
            weights, mean, std = fit(train, key)
            grid = [round(value / 100, 2) for value in range(35, 91, 5)]
            threshold, validation_metrics = max(((value, evaluate(validation, validation_frames, weights, mean, std, key, value)) for value in grid), key=lambda item: item[1]["safe_success"])
            variants[key] = {"threshold": threshold, "validation": validation_metrics, "held_out": evaluate(test, test_frames, weights, mean, std, key, threshold)}
        results[test_scene] = variants
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"features": {"base": ["projection_support", "box_center_similarity", "cluster_track_age", "normalized_range", "normalized_cluster_size"], "temporal_extra": ["camera_range_proxy_residual", "radar_doppler_closing", "camera_box_growth", "closing_growth_agreement"]}, "rows": len(rows), "results": results}
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    for scene, outcome in results.items():
        print(scene, "base", outcome["base"]["held_out"], "temporal", outcome["temporal"]["held_out"])


if __name__ == "__main__":
    main()
