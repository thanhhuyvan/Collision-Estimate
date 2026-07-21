"""Build scene-split temporal radar-camera association samples from oracle labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from guardian_perception.radar import RadarClusterTracker, cluster_radar_points
from render_association_gate_experiment import oracle_object_points
from run_learned_association_ablation import candidate_features
from run_multisweep_association_experiment import collect_sweeps
from run_nuscenes_rule_based_fusion_demo import annotation_bbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-scene", default="scene-0796")
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--positive-precision", type=float, default=0.5)
    parser.add_argument("--negative-precision", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/association_training_samples.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/association_training_samples_summary.json"))
    args = parser.parse_args()
    if not 0 <= args.negative_precision < args.positive_precision <= 1:
        raise ValueError("require 0 <= negative precision < positive precision <= 1")
    if args.sweeps < 1:
        raise ValueError("sweeps must be at least one")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tracker, active_scene, counts = RadarClusterTracker(), None, Counter()
    with args.output.open("w", encoding="utf-8") as output:
        for frame_index, frame in enumerate(manifest["frames"]):
            if frame["scene"] != active_scene:
                tracker, active_scene = RadarClusterTracker(), frame["scene"]
            global_points, pixels, radar_points, visible = collect_sweeps(
                frame, args.data_root, args.sweeps, motion_compensate=True
            )
            clusters = cluster_radar_points(radar_points)
            ages = tracker.update(clusters, int(frame["timestamp_us"] // 1000))
            for annotation in frame["annotations"]:
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
                    # Keep weak geometric candidates as hard negatives, but drop pairs
                    # with zero image evidence to avoid a trivial all-negative dataset.
                    if features[0] < 0.15:
                        continue
                    precision = len(set(cluster.point_indexes) & oracle) / len(cluster.point_indexes)
                    # A radar cluster straddling an object boundary is not a clean
                    # positive or negative association.  Keep it for visual audit,
                    # but exclude it from supervised fitting later rather than
                    # forcing noisy binary targets.
                    if precision >= args.positive_precision:
                        label_state, label_reason = "positive", "majority_of_cluster_inside_gt_3d_box"
                    elif precision <= args.negative_precision:
                        label_state, label_reason = "negative", "cluster_outside_gt_3d_box"
                    else:
                        label_state, label_reason = "ignore", "ambiguous_gt_3d_box_overlap"
                    sample = {
                        "split": "test" if frame["scene"] == args.test_scene else "train",
                        "scene": frame["scene"], "frame_index": frame_index,
                        "instance_id": annotation["instance_token"], "category": annotation["category"],
                        "cluster_id": cluster.cluster_id, "features": [round(float(value), 6) for value in features],
                        "rule_score": round(float(rule_score), 6), "oracle_cluster_precision": round(float(precision), 6),
                        "label_state": label_state,
                        "label_reason": label_reason,
                        # Retained for compatibility with the initial ablation.
                        "label_match": int(label_state == "positive"),
                    }
                    output.write(json.dumps(sample) + "\n")
                    counts[f"{sample['split']}_samples"] += 1
                    counts[f"{sample['split']}_{label_state}"] += 1
    summary = {
        "test_scene": args.test_scene,
        "radar_sweeps": args.sweeps,
        **counts,
        "feature_schema": ["projection_support", "box_center_similarity", "cluster_track_age", "normalized_range", "normalized_cluster_size"],
        "label_contract": {
            "positive": f"cluster overlap with the GT object 3D box >= {args.positive_precision:.2f}",
            "negative": f"cluster overlap with the GT object 3D box <= {args.negative_precision:.2f}",
            "ignore": "overlap is ambiguous; retained for audit and excluded from fitting",
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(args.summary.resolve())


if __name__ == "__main__":
    main()
