"""Compare camera-only, hard radar match and PDAF radar fusion for the lead target.

This is an oracle-camera-track diagnostic: GT 3D boxes supply camera proposals and
lead identity, but never select a radar measurement.  GT radar overlap is only
used after each update to score association correctness and position error.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from guardian_perception.probabilistic import (
    GaussianState,
    linear_update,
    pdaf_position_update,
    pdaf_weights,
    predict_constant_velocity,
    squared_mahalanobis,
)
from guardian_perception.radar import cluster_radar_points
from evaluate_collision_target_association import ego_position
from render_association_gate_experiment import oracle_object_points
from render_nuscenes_fusion_preview import rotation_matrix
from run_learned_association_ablation import candidate_features
from run_multisweep_association_experiment import collect_sweeps
from run_nuscenes_rule_based_fusion_demo import annotation_bbox


POSITION_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)


def camera_ground_measurement(bbox: tuple[int, int, int, int], camera: dict) -> np.ndarray | None:
    """Intersect the bottom-centre camera ray with an ego-frame flat-ground plane."""

    x1, _, x2, y2 = bbox
    intrinsic = np.asarray(camera["calibration"]["camera_intrinsic"], dtype=float)
    ray_camera = np.linalg.solve(intrinsic, np.array([(x1 + x2) / 2, y2, 1.0]))
    origin_ego = np.asarray(camera["calibration"]["translation"], dtype=float)
    ray_ego = rotation_matrix(camera["calibration"]["rotation"]) @ ray_camera
    if abs(ray_ego[2]) < 1e-6:
        return None
    scale = -origin_ego[2] / ray_ego[2]
    if scale <= 0:
        return None
    point = origin_ego + scale * ray_ego
    if not 1.0 < point[0] < 100.0 or abs(point[1]) > 30.0:
        return None
    return point[:2]


def lead_annotation(frame: dict) -> dict | None:
    candidates = []
    for annotation in frame["annotations"]:
        if not annotation["category"].startswith("vehicle."):
            continue
        x_m, y_m, _ = ego_position(annotation, frame)
        if 2.0 < x_m <= 60.0 and abs(y_m) <= 1.75 + annotation["size_m"][0] / 2:
            candidates.append((x_m, annotation))
    return min(candidates, default=(None, None), key=lambda item: float("inf") if item[0] is None else item[0])[1]


def initial_state(position: np.ndarray) -> GaussianState:
    return GaussianState(np.array([position[0], position[1], 0.0, 0.0]), np.diag([36.0, 9.0, 64.0, 25.0]))


def camera_covariance(position: np.ndarray) -> np.ndarray:
    range_std = max(2.0, 0.12 * float(np.linalg.norm(position)))
    return np.diag([range_std**2, max(1.5, range_std * 0.55) ** 2])


def frame_error(state: GaussianState, truth: np.ndarray) -> float:
    return float(np.linalg.norm(state.mean[:2] - truth[:2]))


def metric_summary(records: list[dict], key: str) -> dict:
    errors = [record[f"{key}_position_error_m"] for record in records]
    if key == "camera":
        return {
            "frames": len(records),
            "mean_position_error_m": float(np.mean(errors)) if errors else None,
            "p95_position_error_m": float(np.percentile(errors, 95)) if errors else None,
            "radar_used_frames": 0,
            "radar_association_precision": None,
            "radar_abstain_frames": len(records),
        }
    used = [record for record in records if record[f"{key}_radar_used"]]
    correct = sum(record[f"{key}_radar_correct"] for record in used)
    return {
        "frames": len(records),
        "mean_position_error_m": float(np.mean(errors)) if errors else None,
        "p95_position_error_m": float(np.percentile(errors, 95)) if errors else None,
        "radar_used_frames": len(used),
        "radar_association_precision": correct / len(used) if used else None,
        "radar_abstain_frames": len(records) - len(used),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/classical_probabilistic_fusion_40f.json"))
    parser.add_argument("--gate", type=float, default=9.21, help="Chi-square 99% gate for 2D position.")
    parser.add_argument("--pdaf-minimum-mass", type=float, default=0.60)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, states, previous_timestamp, active_scene, active_instance = [], {}, None, None, None
    for frame_index, frame in enumerate(manifest["frames"]):
        annotation = lead_annotation(frame)
        if annotation is None:
            continue
        bbox = annotation_bbox(annotation, frame["camera"], (1600, 900))
        camera_position = None if bbox is None else camera_ground_measurement(bbox, frame["camera"])
        if camera_position is None:
            continue
        truth = ego_position(annotation, frame)[:2]
        timestamp = int(frame["timestamp_us"])
        reset = frame["scene"] != active_scene or annotation["instance_token"] != active_instance
        if reset:
            states = {key: initial_state(camera_position) for key in ("camera", "hard", "pdaf")}
            previous_timestamp, active_scene, active_instance = timestamp, frame["scene"], annotation["instance_token"]
        else:
            dt_s = max(0.05, (timestamp - previous_timestamp) / 1_000_000)
            states = {key: predict_constant_velocity(value, dt_s) for key, value in states.items()}
            previous_timestamp = timestamp
        covariance_camera = camera_covariance(camera_position)
        states = {key: linear_update(value, camera_position, POSITION_H, covariance_camera) for key, value in states.items()}

        global_points, pixels, radar_points, visible = collect_sweeps(frame, args.data_root, 1)
        clusters = cluster_radar_points(radar_points)
        oracle = set(np.flatnonzero(oracle_object_points(global_points, annotation)).tolist())
        candidates = []
        radar_covariance = np.diag([2.5**2, 1.5**2])
        for cluster in clusters:
            # Projection support is only a weak candidate filter. The association
            # decision below uses predicted physical state and its covariance.
            feature_result = candidate_features(cluster, pixels, visible, bbox, age=1)
            if feature_result is None or feature_result[0][0] < 0.15:
                continue
            measurement = np.array([cluster.x_m, cluster.y_m])
            precision = len(set(cluster.point_indexes) & oracle) / len(cluster.point_indexes)
            candidates.append({"cluster": cluster, "measurement": measurement, "positive": precision >= 0.5})

        def gated(state: GaussianState) -> list[dict]:
            return [item | {"distance": squared_mahalanobis(state, item["measurement"], POSITION_H, radar_covariance)} for item in candidates if squared_mahalanobis(state, item["measurement"], POSITION_H, radar_covariance) <= args.gate]

        hard_candidates, pdaf_candidates = gated(states["hard"]), gated(states["pdaf"])
        hard_selected = min(hard_candidates, default=None, key=lambda item: float("inf") if item is None else item["distance"])
        if hard_selected is not None:
            states["hard"] = linear_update(states["hard"], hard_selected["measurement"], POSITION_H, radar_covariance)
        distances = [item["distance"] for item in pdaf_candidates]
        weights, missed = pdaf_weights(distances) if distances else (np.empty(0), 1.0)
        pdaf_used = float(weights.sum()) >= args.pdaf_minimum_mass
        pdaf_selected = None
        if pdaf_used:
            states["pdaf"] = pdaf_position_update(states["pdaf"], [item["measurement"] for item in pdaf_candidates], weights, radar_covariance)
            pdaf_selected = pdaf_candidates[int(np.argmax(weights))]
        record = {
            "frame": frame_index, "scene": frame["scene"], "truth_forward_m": float(truth[0]), "truth_lateral_m": float(truth[1]),
            "candidate_count": len(candidates), "hard_gate_count": len(hard_candidates), "pdaf_gate_count": len(pdaf_candidates),
            "pdaf_association_mass": float(weights.sum()), "pdaf_missed_probability": missed,
        }
        for key in ("camera", "hard", "pdaf"):
            record[f"{key}_position_error_m"] = frame_error(states[key], truth)
        record.update({
            "hard_radar_used": hard_selected is not None,
            "hard_radar_correct": False if hard_selected is None else hard_selected["positive"],
            "pdaf_radar_used": pdaf_used,
            "pdaf_radar_correct": False if pdaf_selected is None else pdaf_selected["positive"],
        })
        records.append(record)
    by_scene = defaultdict(list)
    for record in records:
        by_scene[record["scene"]].append(record)
    summary = {
        "scope": "oracle camera lead-track diagnostic; radar never receives GT at runtime",
        "gate": args.gate, "pdaf_minimum_mass": args.pdaf_minimum_mass,
        "overall": {key: metric_summary(records, key) for key in ("camera", "hard", "pdaf")},
        "by_scene": {scene: {key: metric_summary(scene_records, key) for key in ("camera", "hard", "pdaf")} for scene, scene_records in by_scene.items()},
        "records": records,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.summary.resolve())
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
