"""Attach time-ordered RADAR_FRONT sweeps to an existing camera-radar manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write sweep references before raw PCD files are extracted (for extraction planning only).",
    )
    args = parser.parse_args()
    metadata = args.data_root / "v1.0-mini"
    sample_data = json.loads((metadata / "sample_data.json").read_text(encoding="utf-8"))
    calibrated = {row["token"]: row for row in json.loads((metadata / "calibrated_sensor.json").read_text(encoding="utf-8"))}
    poses = {row["token"]: row for row in json.loads((metadata / "ego_pose.json").read_text(encoding="utf-8"))}
    by_filename = {row["filename"]: row for row in sample_data}
    by_token = {row["token"]: row for row in sample_data}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for frame in manifest["frames"]:
        record = by_filename[frame["radar"]["file"]]
        sweeps = []
        for _ in range(args.sweeps):
            file = args.data_root / record["filename"]
            if not file.is_file() and not args.allow_missing:
                raise FileNotFoundError(file)
            sweeps.append({"file": record["filename"], "timestamp_us": record["timestamp"], "calibration": calibrated[record["calibrated_sensor_token"]], "ego_pose": poses[record["ego_pose_token"]]})
            if not record["prev"]:
                break
            record = by_token[record["prev"]]
        frame["radar_sweeps"] = sweeps
    manifest["radar_sweep_count_requested"] = args.sweeps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
