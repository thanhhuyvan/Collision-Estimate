"""Render the three-state association labels for manual data-contract audit.

Green means a radar cluster is a clean match to the GT 3D object box, red is a
clean non-match, and amber is deliberately ignored because the geometric
overlap is ambiguous.  GT is used only to inspect labels, never as inference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from guardian_perception.radar import RadarClusterTracker, cluster_radar_points
from render_association_gate_experiment import oracle_object_points
from run_learned_association_ablation import candidate_features
from run_multisweep_association_experiment import collect_sweeps
from run_nuscenes_rule_based_fusion_demo import annotation_bbox


COLORS = {"positive": (0, 220, 0), "negative": (0, 0, 235), "ignore": (0, 190, 255)}


def label_state(precision: float, positive: float, negative: float) -> str:
    if precision >= positive:
        return "positive"
    if precision <= negative:
        return "negative"
    return "ignore"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/association_label_audit.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/nuscenes_fusion/association_label_audit_preview.jpg"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/association_label_audit_summary.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for path in (args.output, args.preview, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    tracker, active_scene, totals = RadarClusterTracker(), None, Counter()
    for frame_index, frame in enumerate(manifest["frames"]):
        if frame["scene"] != active_scene:
            tracker, active_scene = RadarClusterTracker(), frame["scene"]
        image = cv2.imread(str(args.data_root / frame["camera"]["file"]))
        if image is None:
            raise FileNotFoundError(frame["camera"]["file"])
        global_points, pixels, radar_points, visible = collect_sweeps(frame, args.data_root, args.sweeps, motion_compensate=True)
        clusters = cluster_radar_points(radar_points)
        ages = tracker.update(clusters, int(frame["timestamp_us"] // 1000))
        overlay, frame_counts = image.copy(), Counter()
        for annotation in frame["annotations"]:
            if not annotation["category"].startswith("vehicle."):
                continue
            bbox = annotation_bbox(annotation, frame["camera"], (image.shape[1], image.shape[0]))
            if bbox is None:
                continue
            oracle = set(np.flatnonzero(oracle_object_points(global_points, annotation)).tolist())
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
            for cluster in clusters:
                result = candidate_features(cluster, pixels, visible, bbox, ages[cluster.cluster_id][1])
                if result is None or result[0][0] < 0.15:
                    continue
                precision = len(set(cluster.point_indexes) & oracle) / len(cluster.point_indexes)
                state = label_state(precision, 0.5, 0.1)
                frame_counts[state] += 1
                totals[state] += 1
                representative = next((i for i in cluster.point_indexes if visible[i]), None)
                if representative is None:
                    continue
                px, py = np.rint(pixels[:, representative]).astype(int)
                color = COLORS[state]
                cv2.circle(overlay, (px, py), 7, color, -1)
                cv2.putText(overlay, f"C{cluster.cluster_id} {state[0].upper()} {precision:.0%}", (px + 8, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 82), (20, 20, 20), -1)
        cv2.putText(overlay, f"LABEL AUDIT | {frame['scene']} | frame {frame_index}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"GREEN positive={frame_counts['positive']} | RED negative={frame_counts['negative']} | AMBER ignore={frame_counts['ignore']} | white=GT camera box", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        rendered = cv2.resize(overlay, (960, 540))
        if writer is None:
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (960, 540))
        writer.write(rendered)
        if frame_index == 0:
            cv2.imwrite(str(args.preview), rendered)
    if writer is not None:
        writer.release()
    args.summary.write_text(json.dumps({"frames": len(manifest["frames"]), "sweeps": args.sweeps, "candidate_label_counts": totals, "contract": "green=positive >=50% GT-3D-box overlap; red=negative <=10%; amber=ignore"}, indent=2, default=int) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
