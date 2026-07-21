"""Evaluate radar association only for the forward collision target.

Camera detections are GT-box oracle proposals in this diagnostic.  Therefore the
camera-only lead choice is a scope reference, not a detector benchmark; the
measured question is whether radar can be safely attached to that one target.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def ego_position(annotation: dict, frame: dict) -> np.ndarray:
    from render_nuscenes_fusion_preview import rotation_matrix

    rotation = rotation_matrix(frame["camera"]["ego_pose"]["rotation"])
    return rotation.T @ (np.asarray(annotation["translation_m"]) - np.asarray(frame["camera"]["ego_pose"]["translation"]))


def lead_target(frame: dict, *, max_range_m: float, corridor_half_width_m: float) -> tuple[str, float] | None:
    candidates = []
    for annotation in frame["annotations"]:
        if not annotation["category"].startswith("vehicle."):
            continue
        x_m, y_m, _ = ego_position(annotation, frame)
        half_width = annotation["size_m"][0] / 2
        if 2.0 < x_m <= max_range_m and abs(y_m) <= corridor_half_width_m + half_width:
            candidates.append((float(x_m), annotation["instance_token"]))
    return min(candidates, default=None)


def evaluate(frames: list[dict], rows_by_frame: dict[int, list[dict]], threshold: float, max_range_m: float, corridor_half_width_m: float) -> dict:
    targets = camera_visible_targets = radar_assignments = correct = wrong = ambiguous = 0
    records = []
    for index, frame in enumerate(frames):
        target = lead_target(frame, max_range_m=max_range_m, corridor_half_width_m=corridor_half_width_m)
        if target is None:
            continue
        range_m, instance_id = target
        targets += 1
        options = [row for row in rows_by_frame.get(index, []) if row["instance_id"] == instance_id]
        if not options:
            records.append({"frame": index, "scene": frame["scene"], "lead_instance": instance_id, "lead_range_m": range_m, "state": "no_radar_candidate"})
            continue
        camera_visible_targets += 1
        best = max(options, key=lambda row: float(row["rule_score"]))
        if float(best["rule_score"]) < threshold:
            records.append({"frame": index, "scene": frame["scene"], "lead_instance": instance_id, "lead_range_m": range_m, "state": "abstain", "score": best["rule_score"]})
            continue
        radar_assignments += 1
        state = best["label_state"]
        correct += state == "positive"
        wrong += state == "negative"
        ambiguous += state == "ignore"
        records.append({"frame": index, "scene": frame["scene"], "lead_instance": instance_id, "lead_range_m": range_m, "state": state, "score": best["rule_score"], "cluster": best["cluster_id"]})
    precision = correct / radar_assignments if radar_assignments else 0.0
    coverage = radar_assignments / targets if targets else 0.0
    safe_success = correct / targets if targets else 0.0
    return {"lead_target_frames": targets, "camera_visible_leads": camera_visible_targets, "radar_assignments": radar_assignments, "correct_radar_associations": correct, "wrong_radar_associations": wrong, "ambiguous_radar_associations": ambiguous, "radar_precision_when_used": precision, "radar_coverage": coverage, "safe_radar_success": safe_success, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--validation-scene", default="scene-1077")
    parser.add_argument("--test-scene", default="scene-0796")
    parser.add_argument("--max-range-m", type=float, default=60.0)
    parser.add_argument("--corridor-half-width-m", type=float, default=1.75)
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/collision_target_association_40f_1s.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows_by_frame: dict[int, list[dict]] = defaultdict(list)
    for line in args.samples.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows_by_frame[row["frame_index"]].append(row)
    validation_frames = [frame for frame in manifest["frames"] if frame["scene"] == args.validation_scene]
    test_frames = [frame for frame in manifest["frames"] if frame["scene"] == args.test_scene]
    # Re-index the saved global row frame indices after scene filtering.
    frame_indexes = {id(frame): index for index, frame in enumerate(manifest["frames"])}
    def selected_rows(frames: list[dict]) -> dict[int, list[dict]]:
        return {local: rows_by_frame.get(frame_indexes[id(frame)], []) for local, frame in enumerate(frames)}
    grid = [round(value / 100, 2) for value in range(35, 96, 5)]
    validation_rows = selected_rows(validation_frames)
    threshold, validation = max(
        ((value, evaluate(validation_frames, validation_rows, value, args.max_range_m, args.corridor_half_width_m)) for value in grid),
        key=lambda item: item[1]["safe_radar_success"],
    )
    test = evaluate(test_frames, selected_rows(test_frames), threshold, args.max_range_m, args.corridor_half_width_m)
    summary = {
        "scope": {"lead": "nearest vehicle in forward ego corridor", "max_range_m": args.max_range_m, "corridor_half_width_m": args.corridor_half_width_m, "camera_note": "GT camera proposals; this evaluates radar attachment, not detector accuracy."},
        "threshold_selected_on": {"scene": args.validation_scene, "threshold": threshold, "metrics": {key: value for key, value in validation.items() if key != "records"}},
        "held_out": {"scene": args.test_scene, **{key: value for key, value in test.items() if key != "records"}},
        "held_out_records": test["records"],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.summary.resolve())
    print(json.dumps(summary["held_out"], indent=2))


if __name__ == "__main__":
    main()
