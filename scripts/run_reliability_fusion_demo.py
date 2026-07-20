"""Show reliability-aware late fusion behavior without GPU, model weights, or a dataset."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from guardian_perception import CameraMeasurement, RadarMeasurement, ReliabilityAwareFuser


def main() -> None:
    fuser = ReliabilityAwareFuser()
    cases = {
        "agreement": (
            CameraMeasurement("lead", 30.0, 6.0, 0.92, 0.95),
            RadarMeasurement("radar-01", 29.2, 6.2, 0.90, 0.92),
        ),
        "dark_camera": (
            CameraMeasurement("lead", 34.0, 4.0, 0.70, 0.25),
            RadarMeasurement("radar-01", 29.5, 6.1, 0.92, 0.94),
        ),
        "ambiguous_radar": (
            CameraMeasurement("lead", 30.0, 6.0, 0.92, 0.95),
            RadarMeasurement("radar-02", 18.0, 11.0, 0.90, 0.20),
        ),
        "sensor_disagreement": (
            CameraMeasurement("lead", 30.0, 6.0, 0.92, 0.95),
            RadarMeasurement("radar-01", 50.0, 13.0, 0.90, 0.92),
        ),
    }
    results = {name: asdict(fuser.fuse(camera, radar)) for name, (camera, radar) in cases.items()}
    output = Path("outputs/reliability_fusion/demo.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, result in results.items():
        ttc = "--" if result["ttc_s"] is None else f"{result['ttc_s']:.2f}s"
        print(
            f"{name}: range={result['range_m']:.2f}m TTC={ttc} "
            f"reliability={result['reliability']:.2f} {result['explanation']}"
        )
    print(output.resolve())


if __name__ == "__main__":
    main()
