"""Evaluate a reproducible rule association baseline from saved labelled candidates.

The runtime score was generated without GT.  GT labels appear only after global
one-to-one assignment to measure precision, recall and ambiguous selections.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from guardian_perception.association import AssociationCandidate, greedy_one_to_one_assignment


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score(rows: list[dict], threshold: float) -> dict[str, float]:
    by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[(row["scene"], row["frame_index"])].append(row)
    selected_total = positive_selected = negative_selected = ignored_selected = positives = 0
    for frame_rows in by_frame.values():
        positives += sum(row["label_state"] == "positive" for row in frame_rows)
        candidates = [
            AssociationCandidate(
                camera_id=row["instance_id"], cluster_id=int(row["cluster_id"]),
                score=float(row["rule_score"]), support=float(row["features"][0]),
                center_similarity=float(row["features"][1]),
            )
            for row in frame_rows
        ]
        by_pair = {(row["instance_id"], int(row["cluster_id"])): row for row in frame_rows}
        for assignment in greedy_one_to_one_assignment(candidates, minimum_score=threshold).values():
            selected_total += 1
            state = by_pair[(assignment.camera_id, assignment.cluster_id)]["label_state"]
            if state == "positive":
                positive_selected += 1
            elif state == "negative":
                negative_selected += 1
            else:
                ignored_selected += 1
    precision = positive_selected / selected_total if selected_total else 0.0
    recall = positive_selected / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "frames": len(by_frame), "positive_candidates": positives, "assignments": selected_total,
        "true_positive": positive_selected, "false_positive": negative_selected,
        "ambiguous_selected": ignored_selected, "precision": precision, "recall": recall, "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--validation-scene", default="scene-1077")
    parser.add_argument("--test-scene", default="scene-0796")
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/rule_association_baseline_40f_1s.json"))
    args = parser.parse_args()
    rows = load_rows(args.samples)
    validation = [row for row in rows if row["scene"] == args.validation_scene]
    test = [row for row in rows if row["scene"] == args.test_scene]
    if not validation or not test:
        raise ValueError("validation or test scene has no candidates")
    candidates = [round(value / 100, 2) for value in range(35, 96, 5)]
    threshold, validation_metrics = max(
        ((value, score(validation, value)) for value in candidates), key=lambda item: item[1]["f1"]
    )
    summary = {
        "runtime": "projection support + box-center similarity + radar-cluster track age; global greedy one-to-one assignment",
        "selection": {"validation_scene": args.validation_scene, "threshold_grid": candidates, "selected_threshold": threshold, "validation": validation_metrics},
        "held_out": {"scene": args.test_scene, **score(test, threshold)},
        # Diagnostic only: this curve must not be used to retune the selected
        # threshold, because the test scene remains held out.
        "held_out_threshold_curve_diagnostic": [
            {"threshold": value, **score(test, value)} for value in candidates
        ],
        "label_note": "ambiguous/ignore candidates remain visible to runtime and are reported separately; they are not used as true positives.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.summary.resolve())
    print(json.dumps(summary["held_out"], indent=2))


if __name__ == "__main__":
    main()
