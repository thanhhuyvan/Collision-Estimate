"""Extract only CAM_FRONT/RADAR_FRONT files needed for a multi-scene mini evaluation."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["scene-0061", "scene-0553", "scene-0796", "scene-1077"])
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    metadata = args.data_root / "v1.0-mini"
    scenes = {row["name"]: row for row in json.loads((metadata / "scene.json").read_text(encoding="utf-8"))}
    samples = {row["token"]: row for row in json.loads((metadata / "sample.json").read_text(encoding="utf-8"))}
    sample_data = json.loads((metadata / "sample_data.json").read_text(encoding="utf-8"))
    calibrated = {row["token"]: row for row in json.loads((metadata / "calibrated_sensor.json").read_text(encoding="utf-8"))}
    sensors = {row["token"]: row for row in json.loads((metadata / "sensor.json").read_text(encoding="utf-8"))}
    by_sample: dict[str, list[dict]] = {}
    for record in sample_data:
        by_sample.setdefault(record["sample_token"], []).append(record)
    requested: set[str] = set()
    for scene_name in args.scenes:
        token = scenes[scene_name]["first_sample_token"]
        count = 0
        while token and count < args.frames:
            for record in by_sample[token]:
                channel = sensors[calibrated[record["calibrated_sensor_token"]]["sensor_token"]]["channel"]
                if record["is_key_frame"] and channel in {"CAM_FRONT", "RADAR_FRONT"}:
                    requested.add(record["filename"])
            token = samples[token]["next"]
            count += 1
    missing = [name for name in requested if not (args.data_root / name).is_file()]
    with tarfile.open(args.archive, "r:gz") as archive:
        members = {member.name: member for member in archive if member.name in missing}
        not_found = set(missing) - set(members)
        if not_found:
            raise FileNotFoundError(f"archive lacks requested sensor files: {sorted(not_found)[:3]}")
        for name in missing:
            archive.extract(members[name], args.data_root)
    print(f"requested={len(requested)} already_present={len(requested) - len(missing)} extracted={len(missing)}")


if __name__ == "__main__":
    main()
