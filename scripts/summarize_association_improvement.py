"""Summarise V1/V2 oracle association reports in a compact comparison chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_metrics(path: Path) -> dict[str, float]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    valid = [record for record in records if record.get("oracle_point_count", 0) > 0 and record.get("selected_cluster") is not None]
    # Macro scores make each frame equally visible; they are not a production safety metric.
    return {
        "frames": len(valid),
        "precision": float(np.mean([record["oracle_precision"] for record in valid])),
        "recall": float(np.mean([record["oracle_recall"] for record in valid])),
        "good_match_rate": float(np.mean([record["oracle_precision"] >= 0.5 for record in valid])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_cluster_association.jsonl"))
    parser.add_argument("--v2", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_cluster_association_v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/association_improvement_chart.png"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/association_improvement_summary.json"))
    args = parser.parse_args()
    v1, v2 = load_metrics(args.v1), load_metrics(args.v2)
    metrics = [("Oracle precision", "precision"), ("Oracle recall", "recall"), ("Good-match frames", "good_match_rate")]
    chart = np.full((720, 1280, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(chart, "Oracle association: V1 vs V2", (45, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(chart, f"Evaluation-only: {int(v1['frames'])} frames with GT radar returns. V2 adds a nearest-supported-cluster tie-break.", (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1, cv2.LINE_AA)
    labels = ("V1 projection only", "V2 frontmost supported")
    colours = ((80, 90, 255), (0, 230, 150))
    for row, (label, key) in enumerate(metrics):
        y = 180 + row * 160
        cv2.putText(chart, label, (45, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 1, cv2.LINE_AA)
        for column, (values, colour) in enumerate(((v1, colours[0]), (v2, colours[1]))):
            x = 310 + column * 370
            value = values[key]
            bar_width = int(270 * value)
            cv2.rectangle(chart, (x, y), (x + 270, y + 42), (60, 60, 60), -1)
            cv2.rectangle(chart, (x, y), (x + bar_width, y + 42), colour, -1)
            cv2.putText(chart, f"{labels[column]}: {value:.0%}", (x, y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)
    cv2.putText(chart, "Interpretation: V2 is still rule-based and tested on a small four-scene subset. It is a directional diagnostic, not a safety claim.", (45, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 180, 255), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), chart)
    args.summary.write_text(json.dumps({"v1_projection_only": v1, "v2_frontmost_supported": v2}, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(args.summary.resolve())


if __name__ == "__main__":
    main()
