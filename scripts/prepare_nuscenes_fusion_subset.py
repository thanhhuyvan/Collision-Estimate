"""Create a compact, inspectable camera-radar subset manifest from nuScenes metadata.

The script does not copy raw sensor files. It expects the requested CAM_FRONT and
RADAR_FRONT files to have already been extracted beside the nuScenes JSON metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_table(root: Path, name: str) -> list[dict]:
    return json.loads((root / "v1.0-mini" / f"{name}.json").read_text(encoding="utf-8"))


def table_by_token(root: Path, name: str) -> dict[str, dict]:
    return {row["token"]: row for row in load_table(root, name)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene", default="scene-0061", help="One scene name (kept for backwards compatibility).")
    parser.add_argument("--scenes", nargs="+", help="One or more scenes to combine into an evaluation manifest.")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.frames <= 0:
        raise ValueError("frames must be positive")

    root = arguments.data_root
    output = arguments.output or root / "fusion_subset_manifest.json"
    scenes = {row["name"]: row for row in load_table(root, "scene")}
    requested_scenes = arguments.scenes or [arguments.scene]
    missing = set(requested_scenes) - set(scenes)
    if missing:
        raise ValueError(f"scene(s) absent from {root}: {sorted(missing)}")
    samples = table_by_token(root, "sample")
    sample_data = table_by_token(root, "sample_data")
    calibrated = table_by_token(root, "calibrated_sensor")
    sensors = table_by_token(root, "sensor")
    poses = table_by_token(root, "ego_pose")
    annotations = table_by_token(root, "sample_annotation")
    instances = table_by_token(root, "instance")
    categories = table_by_token(root, "category")

    sample_data_by_sample: dict[str, list[dict]] = {}
    for record in sample_data.values():
        sample_data_by_sample.setdefault(record["sample_token"], []).append(record)
    annotations_by_sample: dict[str, list[dict]] = {}
    for annotation in annotations.values():
        annotations_by_sample.setdefault(annotation["sample_token"], []).append(annotation)
    frames: list[dict] = []
    for scene_name in requested_scenes:
        scene = scenes[scene_name]
        token = scene["first_sample_token"]
        scene_frame_count = 0
        while token and scene_frame_count < arguments.frames:
            sample = samples[token]
            channel_records: dict[str, dict] = {}
            for record in sample_data_by_sample.get(token, []):
                channel = sensors[calibrated[record["calibrated_sensor_token"]]["sensor_token"]]["channel"]
                if record["is_key_frame"] and channel in {"CAM_FRONT", "RADAR_FRONT"}:
                    channel_records[channel] = record
            if set(channel_records) == {"CAM_FRONT", "RADAR_FRONT"}:
                camera, radar = channel_records["CAM_FRONT"], channel_records["RADAR_FRONT"]
                for record in (camera, radar):
                    file_path = root / record["filename"]
                    if not file_path.is_file():
                        raise FileNotFoundError(f"subset is missing extracted sensor file: {file_path}")
                frame_annotations = []
                for annotation in annotations_by_sample.get(token, []):
                    instance = instances[annotation["instance_token"]]
                    category = categories[instance["category_token"]]
                    frame_annotations.append({"annotation_token": annotation["token"], "instance_token": annotation["instance_token"], "category": category["name"], "translation_m": annotation["translation"], "size_m": annotation["size"], "rotation": annotation["rotation"]})
                frames.append({"scene": scene_name, "scene_description": scene["description"], "timestamp_us": sample["timestamp"], "sample_token": token, "camera": {"file": camera["filename"], "calibration": calibrated[camera["calibrated_sensor_token"]], "ego_pose": poses[camera["ego_pose_token"]]}, "radar": {"file": radar["filename"], "calibration": calibrated[radar["calibrated_sensor_token"]], "ego_pose": poses[radar["ego_pose_token"]]}, "annotations": frame_annotations})
                scene_frame_count += 1
            token = sample["next"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": "nuScenes v1.0-mini",
                "scenes": requested_scenes,
                "frame_count": len(frames),
                "frames": frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
