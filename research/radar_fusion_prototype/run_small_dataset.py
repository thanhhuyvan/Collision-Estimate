"""Replay the deterministic small dataset and enforce MVP behaviour expectations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from guardian_perception import CameraMeasurement, RadarMeasurement

from lead_object_system import LeadObjectCollisionSystem, LeadObjectInput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data" / "lead_object_small" / "scenarios.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "small_dataset")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.output_dir / "decisions.jsonl"
    summary_path = args.output_dir / "summary.csv"
    summaries: list[dict[str, str | int | bool]] = []
    with decisions_path.open("w", encoding="utf-8") as decisions:
        for scenario in dataset["scenarios"]:
            system = LeadObjectCollisionSystem()
            final = None
            for frame_id, frame in enumerate(scenario["frames"]):
                camera = CameraMeasurement(scenario["id"], frame["camera_range_m"], frame["camera_closing_speed_mps"], 0.90, 0.85)
                radar = RadarMeasurement(f"{scenario['id']}-{frame_id}", frame["radar_range_m"], frame["radar_closing_speed_mps"], 0.95, frame["association_confidence"])
                final = system.evaluate(LeadObjectInput(frame_id, frame_id * 200, camera, radar, frame["in_ego_corridor"], frame_id + 1))
                record = {"scenario": scenario["id"], **final.as_dict()}
                decisions.write(json.dumps(record) + "\n")
            assert final is not None
            passed = final.risk.value == scenario["expected_final_risk"]
            summaries.append({"scenario": scenario["id"], "expected_final_risk": scenario["expected_final_risk"], "actual_final_risk": final.risk.value, "passed": passed, "frames": len(scenario["frames"])})
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for result in summaries:
        print(f"{result['scenario']}: {result['actual_final_risk']} ({'PASS' if result['passed'] else 'FAIL'})")
    if not all(bool(result["passed"]) for result in summaries):
        raise SystemExit("small dataset expectations failed")
    print(decisions_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
