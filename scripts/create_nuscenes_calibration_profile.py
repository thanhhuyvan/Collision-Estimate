"""Extract a reusable CAM_FRONT ↔ RADAR_FRONT calibration profile from a subset manifest.

The values originate from nuScenes calibrated_sensor records.  This is not an
image-based re-calibration: nuScenes already supplies the authoritative rig
calibration, which is the correct baseline for an offline experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from render_nuscenes_fusion_preview import rotation_matrix


def radar_to_camera_transform(camera_calibration: dict, radar_calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return p_camera = rotation @ p_radar + translation."""

    radar_to_ego = rotation_matrix(radar_calibration["rotation"])
    camera_to_ego = rotation_matrix(camera_calibration["rotation"])
    radar_translation = np.asarray(radar_calibration["translation"], dtype=float)
    camera_translation = np.asarray(camera_calibration["translation"], dtype=float)
    return camera_to_ego.T @ radar_to_ego, camera_to_ego.T @ (radar_translation - camera_translation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/calibration/nuscenes_scene0061_cam_front_radar_front.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes_fusion/calibration_sanity_report.json"))
    args = parser.parse_args()
    manifest_path = args.manifest or args.data_root / "fusion_subset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["frames"][0]
    camera = first["camera"]
    radar = first["radar"]
    rotation, translation = radar_to_camera_transform(camera["calibration"], radar["calibration"])
    offsets_ms: list[float] = []
    max_rotation_difference = 0.0
    max_translation_difference = 0.0
    for frame in manifest["frames"]:
        current_rotation, current_translation = radar_to_camera_transform(frame["camera"]["calibration"], frame["radar"]["calibration"])
        max_rotation_difference = max(max_rotation_difference, float(np.max(np.abs(rotation - current_rotation))))
        max_translation_difference = max(max_translation_difference, float(np.max(np.abs(translation - current_translation))))
        offsets_ms.append((int(frame["radar"]["ego_pose"]["timestamp"]) - int(frame["camera"]["ego_pose"]["timestamp"])) / 1000)
    profile = {
        "profile_name": "nuscenes_scene0061_cam_front_radar_front",
        "source": "nuScenes v1.0-mini calibrated_sensor metadata",
        "purpose": "offline camera-radar projection baseline; not a physical calibration for a different vehicle",
        "coordinate_convention": "p_camera = radar_to_camera_rotation @ p_radar + radar_to_camera_translation_m",
        "camera": {"name": "CAM_FRONT", "intrinsic": camera["calibration"]["camera_intrinsic"], "sensor_to_ego": camera["calibration"]},
        "radar": {"name": "RADAR_FRONT", "sensor_to_ego": radar["calibration"]},
        "radar_to_camera_rotation": rotation.round(12).tolist(),
        "radar_to_camera_translation_m": translation.round(12).tolist(),
        "runtime_gates": {"maximum_camera_radar_offset_ms": 100, "association_projection_padding_px": 14, "minimum_association_confidence": 0.50},
    }
    report = {
        "frames_checked": len(manifest["frames"]),
        "calibration_is_constant_across_subset": max_rotation_difference < 1e-9 and max_translation_difference < 1e-9,
        "max_rotation_element_difference": max_rotation_difference,
        "max_translation_difference_m": max_translation_difference,
        "radar_camera_time_offset_ms": {"min": min(offsets_ms), "max": max(offsets_ms), "mean": float(np.mean(offsets_ms))},
        "status": "pass" if max(abs(value) for value in offsets_ms) <= 100 else "review timestamp alignment",
        "warning": "This validates metadata consistency, not physical installation on another vehicle.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(args.report.resolve())


if __name__ == "__main__":
    main()
