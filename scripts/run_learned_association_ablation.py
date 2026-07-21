"""Train a tiny association scorer on three scenes and test it on dense traffic scene-0796."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from guardian_perception.association import AssociationCandidate, greedy_one_to_one_assignment
from guardian_perception.radar import RadarClusterTracker, cluster_radar_points
from render_association_gate_experiment import oracle_object_points
from run_multisweep_association_experiment import collect_sweeps
from run_nuscenes_rule_based_fusion_demo import annotation_bbox


def candidate_features(cluster, pixels, visible, bbox, age):
    x1, y1, x2, y2 = bbox
    projected = [i for i in cluster.point_indexes if visible[i]]
    if not projected:
        return None
    support = sum(x1 - 12 <= pixels[0, i] <= x2 + 12 and y1 - 12 <= pixels[1, i] <= y2 + 12 for i in projected) / len(cluster.point_indexes)
    centroid = np.mean(pixels[:, projected], axis=1)
    center = max(0.0, 1.0 - np.hypot((centroid[0] - (x1 + x2) / 2) / max(1, (x2 - x1) / 2), (centroid[1] - (y1 + y2) / 2) / max(1, (y2 - y1) / 2)))
    features = np.array([support, center, min(1.0, age / 3), min(1.0, cluster.range_m / 80), min(1.0, len(cluster.point_indexes) / 12)], dtype=float)
    rule_score = 0.65 * support + 0.20 * center + 0.15 * min(1.0, age / 3)
    return features, rule_score


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = features.mean(axis=0), features.std(axis=0) + 1e-6
    x = (features - mean) / std
    weights = np.zeros(x.shape[1] + 1)
    positive_weight = len(labels) / max(1, 2 * labels.sum())
    negative_weight = len(labels) / max(1, 2 * (len(labels) - labels.sum()))
    sample_weights = np.where(labels == 1, positive_weight, negative_weight)
    for _ in range(1200):
        logits = x @ weights[:-1] + weights[-1]
        prediction = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        error = (prediction - labels) * sample_weights
        weights[:-1] -= 0.08 * (x.T @ error / len(x) + 0.001 * weights[:-1])
        weights[-1] -= 0.08 * error.mean()
    return weights, mean, std


def probability(features, weights, mean, std):
    logit = ((features - mean) / std) @ weights[:-1] + weights[-1]
    return float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))


def score_assignments(candidates, labels, threshold):
    assignments = greedy_one_to_one_assignment(candidates, minimum_score=threshold)
    selected_labels = [labels[(item.camera_id, item.cluster_id)] for item in assignments.values()]
    return {"assignments": len(assignments), "precision": float(np.mean(selected_labels)) if selected_labels else 0.0, "good_rate": float(np.mean([label >= 0.5 for label in selected_labels])) if selected_labels else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-scene", default="scene-0796")
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/learned_association_ablation.png"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/learned_association_ablation.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_rows, tracker, current_scene = [], RadarClusterTracker(), None
    for frame in manifest["frames"]:
        if frame["scene"] != current_scene:
            tracker, current_scene = RadarClusterTracker(), frame["scene"]
        global_points, pixels, radar_points, visible = collect_sweeps(frame, args.data_root, 3, motion_compensate=True)
        clusters = cluster_radar_points(radar_points)
        ages = tracker.update(clusters, int(frame["timestamp_us"] // 1000))
        for object_index, annotation in enumerate(frame["annotations"]):
            if not annotation["category"].startswith("vehicle."):
                continue
            bbox = annotation_bbox(annotation, frame["camera"], (1600, 900))
            if bbox is None:
                continue
            oracle = set(np.flatnonzero(oracle_object_points(global_points, annotation)).tolist())
            for cluster in clusters:
                result = candidate_features(cluster, pixels, visible, bbox, ages[cluster.cluster_id][1])
                if result is None:
                    continue
                features, rule_score = result
                precision = len(set(cluster.point_indexes) & oracle) / len(cluster.point_indexes)
                all_rows.append({"scene": frame["scene"], "frame": frame["sample_token"], "camera_id": str(object_index), "cluster_id": cluster.cluster_id, "features": features, "label": float(precision >= 0.5), "precision": precision, "rule_score": rule_score})
    train = [row for row in all_rows if row["scene"] != args.test_scene]
    test_by_frame = defaultdict(list)
    for row in all_rows:
        if row["scene"] == args.test_scene:
            test_by_frame[row["frame"]].append(row)
    weights, mean, std = fit_logistic(np.stack([row["features"] for row in train]), np.array([row["label"] for row in train]))
    totals = {"rule": [], "learned": []}
    for rows in test_by_frame.values():
        labels = {(row["camera_id"], row["cluster_id"]): row["precision"] for row in rows}
        rule = [AssociationCandidate(row["camera_id"], row["cluster_id"], row["rule_score"], row["features"][0], row["features"][1]) for row in rows]
        learned = [AssociationCandidate(row["camera_id"], row["cluster_id"], probability(row["features"], weights, mean, std), row["features"][0], row["features"][1]) for row in rows]
        totals["rule"].append(score_assignments(rule, labels, 0.68))
        totals["learned"].append(score_assignments(learned, labels, 0.50))
    summary = {"train_scenes": sorted({row["scene"] for row in train}), "test_scene": args.test_scene, "train_candidates": len(train), "test_frames": len(test_by_frame), "rule": {key: float(np.mean([row[key] for row in totals["rule"]])) for key in totals["rule"][0]}, "learned": {key: float(np.mean([row[key] for row in totals["learned"]])) for key in totals["learned"][0]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    chart = np.full((520, 1100, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(chart, "Learned association ablation | held-out dense traffic", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(chart, f"Train: {', '.join(summary['train_scenes'])} | Test: {args.test_scene} | same 3-sweep Doppler data and clustering", (35, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1, cv2.LINE_AA)
    for index, key in enumerate(("precision", "good_rate", "assignments")):
        y = 150 + index * 110
        cv2.putText(chart, key.replace("_", " "), (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1, cv2.LINE_AA)
        for column, (name, colour) in enumerate((("rule", (70, 90, 255)), ("learned", (0, 230, 120)))):
            value = summary[name][key]
            x = 300 + column * 360
            scale = 1 if key != "assignments" else 0.25
            cv2.rectangle(chart, (x, y - 28), (x + 240, y + 8), (60, 60, 60), -1)
            cv2.rectangle(chart, (x, y - 28), (x + int(240 * min(1, value * scale)), y + 8), colour, -1)
            cv2.putText(chart, f"{name}: {value:.2f}", (x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    cv2.putText(chart, "Experimental only: small data and oracle boxes. The ablation isolates score learning, not end-to-end deployment.", (35, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.output), chart)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(args.summary.resolve())


if __name__ == "__main__":
    main()
