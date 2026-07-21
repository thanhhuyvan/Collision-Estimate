"""Run an offline, dependency-free hackathon lead-object replay."""

from __future__ import annotations

import json
from pathlib import Path

from guardian_perception import CameraMeasurement, RadarMeasurement

from lead_object_system import LeadObjectCollisionSystem, LeadObjectInput


def main() -> None:
    system = LeadObjectCollisionSystem()
    output = Path(__file__).parent / "outputs" / "lead_object_demo.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    # In a real demo these values come from the selected YOLO track and its associated radar cluster.
    replay = [
        (32, 31, 6.0, 0.9, True),
        (28, 27, 6.2, 0.9, True),
        (24, 23, 6.3, 0.9, True),
        (20, 20, 6.5, 0.9, True),
        (16, 16, 6.8, 0.25, True),  # Ambiguous radar: force safe fallback.
        (13, 13, 7.0, 0.9, True),
        (10, 10, 7.2, 0.9, True),
    ]
    with output.open("w", encoding="utf-8") as stream:
        for frame_id, (camera_range, radar_range, closing_speed, association, in_corridor) in enumerate(replay):
            camera = CameraMeasurement("lead-1", camera_range, closing_speed * 0.7, 0.90, 0.85)
            radar = RadarMeasurement(f"radar-{frame_id}", radar_range, closing_speed, 0.95, association)
            decision = system.evaluate(LeadObjectInput(frame_id, frame_id * 200, camera, radar, in_corridor, frame_id + 1))
            stream.write(json.dumps(decision.as_dict()) + "\n")
            print(f"frame={frame_id} risk={decision.risk.value} ttc={decision.fused_ttc_s} reason={decision.reason}")
    print(output.resolve())


if __name__ == "__main__":
    main()
