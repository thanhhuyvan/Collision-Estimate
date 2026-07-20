"""Render a camera-radar fusion inspection video from a prepared nuScenes subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def rotation_matrix(quaternion: list[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix for nuScenes [w, x, y, z] quaternions."""

    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_sensor_to_global(points: np.ndarray, calibration: dict, ego_pose: dict) -> np.ndarray:
    sensor_to_ego = rotation_matrix(calibration["rotation"])
    ego_to_global = rotation_matrix(ego_pose["rotation"])
    in_ego = sensor_to_ego @ points + np.asarray(calibration["translation"])[:, None]
    return ego_to_global @ in_ego + np.asarray(ego_pose["translation"])[:, None]


def transform_global_to_camera(points: np.ndarray, calibration: dict, ego_pose: dict) -> np.ndarray:
    global_to_ego = rotation_matrix(ego_pose["rotation"]).T
    ego_to_camera = rotation_matrix(calibration["rotation"]).T
    in_ego = global_to_ego @ (points - np.asarray(ego_pose["translation"])[:, None])
    return ego_to_camera @ (in_ego - np.asarray(calibration["translation"])[:, None])


def project_camera(points: np.ndarray, intrinsic: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    valid = points[2] > 0.5
    projected = np.zeros((2, points.shape[1]), dtype=float)
    projected[:, valid] = (np.asarray(intrinsic) @ points[:, valid])[:2] / points[2, valid]
    return projected, valid


def read_pcd(path: Path) -> np.ndarray:
    """Read the binary PCD records used by nuScenes radar samples without extra packages."""

    with path.open("rb") as input_file:
        header_lines: list[str] = []
        while True:
            line = input_file.readline().decode("ascii").strip()
            header_lines.append(line)
            if line.startswith("DATA"):
                break
        payload = input_file.read()
    header = {line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1] for line in header_lines if " " in line}
    fields = header["FIELDS"].split()
    sizes = [int(value) for value in header["SIZE"].split()]
    kinds = header["TYPE"].split()
    counts = [int(value) for value in header["COUNT"].split()]
    type_codes = {("F", 4): "<f4", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4"}
    dtype = np.dtype(
        [
            (field, type_codes[(kind, size)] if count == 1 else (type_codes[(kind, size)], count))
            for field, size, kind, count in zip(fields, sizes, kinds, counts, strict=True)
        ]
    )
    return np.frombuffer(payload, dtype=dtype, count=int(header["POINTS"]))


def draw_gt_boxes(image: np.ndarray, annotations: list[dict], camera: dict) -> int:
    drawn = 0
    camera_rotation = rotation_matrix(camera["ego_pose"]["rotation"])
    camera_translation = np.asarray(camera["ego_pose"]["translation"])
    calibration_rotation = rotation_matrix(camera["calibration"]["rotation"])
    calibration_translation = np.asarray(camera["calibration"]["translation"])
    for annotation in annotations:
        if not annotation["category"].startswith(("vehicle.", "human.pedestrian.")):
            continue
        width, length, height = annotation["size_m"]
        corners = np.array(
            [
                [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2],
                [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2],
                [height / 2, height / 2, height / 2, height / 2, -height / 2, -height / 2, -height / 2, -height / 2],
            ]
        )
        global_points = rotation_matrix(annotation["rotation"]) @ corners + np.asarray(annotation["translation_m"])[:, None]
        ego_points = camera_rotation.T @ (global_points - camera_translation[:, None])
        camera_points = calibration_rotation.T @ (ego_points - calibration_translation[:, None])
        pixels, valid = project_camera(camera_points, camera["calibration"]["camera_intrinsic"])
        if not np.all(valid):
            continue
        if np.any(pixels[0] < 0) or np.any(pixels[0] >= image.shape[1]) or np.any(pixels[1] < 0) or np.any(pixels[1] >= image.shape[0]):
            continue
        colour = (0, 200, 255) if annotation["category"].startswith("vehicle.") else (255, 110, 0)
        pixels = np.rint(pixels.T).astype(np.int32)
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            cv2.line(image, tuple(pixels[first]), tuple(pixels[second]), colour, 1, cv2.LINE_AA)
        cv2.putText(image, "GT " + annotation["category"].split(".")[0], tuple(pixels[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        drawn += 1
    return drawn


def draw_bev(radar: np.ndarray, panel_size: tuple[int, int]) -> np.ndarray:
    width, height = panel_size
    panel = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(panel, "RADAR_FRONT BEV | range + relative velocity", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    origin = (width // 2, height - 32)
    cv2.circle(panel, origin, 7, (255, 255, 255), -1)
    cv2.putText(panel, "ego", (origin[0] + 10, origin[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for distance in (20, 40, 60, 80):
        row = origin[1] - int(distance / 80 * (height - 60))
        cv2.line(panel, (20, row), (width - 20, row), (55, 55, 55), 1)
        cv2.putText(panel, f"{distance}m", (8, row - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    for point in radar:
        forward, lateral = float(point["x"]), float(point["y"])
        if not 0 < forward < 80 or abs(lateral) > 30:
            continue
        x = origin[0] - int(lateral / 30 * (width / 2 - 28))
        y = origin[1] - int(forward / 80 * (height - 60))
        colour = (0, 220, 0) if point["is_quality_valid"] else (100, 100, 100)
        cv2.circle(panel, (x, y), 2, colour, -1)
        # vx_comp / vy_comp are radar's compensated relative velocity estimate.
        # Arrows make the extra information carried by radar visible; they are
        # intentionally not treated as an object velocity until association.
        if point["is_quality_valid"]:
            vx, vy = float(point["vx_comp"]), float(point["vy_comp"])
            arrow_end = (x + int(-vy * 3), y - int(vx * 3))
            cv2.arrowedLine(panel, (x, y), arrow_end, (0, 145, 255), 1, cv2.LINE_AA, tipLength=0.35)
    return panel


def titled(panel: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Add a consistently readable title bar without obscuring sensor data."""

    result = panel.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 48), (15, 15, 15), -1)
    cv2.putText(result, title, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(result, subtitle, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
    return result


def draw_explainer(radar: np.ndarray, projected_count: int, gt_count: int, offset_ms: float, panel_size: tuple[int, int]) -> np.ndarray:
    """Render an in-video legend explaining the role and limits of each sensor."""

    width, height = panel_size
    panel = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
    panel = titled(panel, "WHAT THE SYSTEM WOULD DO NEXT", "This is an inspection view; it does not issue a collision warning.")
    quality_count = int(np.count_nonzero(radar["is_quality_valid"]))
    ranges = np.linalg.norm(np.vstack((radar["x"], radar["y"], radar["z"])), axis=0)
    median_range = float(np.median(ranges)) if len(ranges) else 0.0
    lines = [
        ("1. CAMERA", "YOLO detects vehicle / pedestrian boxes and class.", (255, 220, 80)),
        ("2. RADAR", "Adds sparse range + relative velocity; it has no semantic class.", (0, 220, 0)),
        ("3. ASSOCIATION", "Keep only radar returns compatible with one camera object.", (0, 145, 255)),
        ("4. RELIABILITY FUSION", "Weight each measurement by quality and disagreement.", (215, 140, 255)),
        ("5. RISK", "Use fused range / closing speed for TTC, then safety gates.", (80, 190, 255)),
    ]
    y = 86
    for heading, detail, colour in lines:
        cv2.putText(panel, heading, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 1, cv2.LINE_AA)
        cv2.putText(panel, detail, (24, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (232, 232, 232), 1, cv2.LINE_AA)
        y += 76
    diagnostics = [
        f"this frame: {len(radar)} raw returns | {quality_count} quality-valid | {projected_count} project into camera",
        f"median radar range: {median_range:.1f}m | camera-radar time offset: {offset_ms:+.1f}ms",
        f"{gt_count} GT 3D boxes shown only for evaluation — replace with YOLO boxes at runtime.",
        "Watch for: dots off objects (calibration/time), clutter, wrong association, unstable velocity.",
    ]
    y = height - 118
    for line in diagnostics:
        cv2.putText(panel, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
        y += 25
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_inspection.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/nuscenes_fusion/scene-0061_preview.jpg"))
    parser.add_argument("--fps", type=float, default=2.0)
    arguments = parser.parse_args()
    manifest_path = arguments.manifest or arguments.data_root / "fusion_subset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for index, frame in enumerate(manifest["frames"]):
        camera = frame["camera"]
        radar_record = frame["radar"]
        original_image = cv2.imread(str(arguments.data_root / camera["file"]))
        if original_image is None:
            raise RuntimeError(f"cannot read {camera['file']}")
        image = original_image.copy()
        radar = read_pcd(arguments.data_root / radar_record["file"])
        radar_sensor_points = np.vstack((radar["x"], radar["y"], radar["z"]))
        global_points = transform_sensor_to_global(radar_sensor_points, radar_record["calibration"], radar_record["ego_pose"])
        camera_points = transform_global_to_camera(global_points, camera["calibration"], camera["ego_pose"])
        pixels, valid = project_camera(camera_points, camera["calibration"]["camera_intrinsic"])
        gt_count = draw_gt_boxes(image, frame["annotations"], camera)
        projected_count = 0
        for point_index in np.flatnonzero(valid):
            x, y = np.rint(pixels[:, point_index]).astype(int)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                distance = float(np.linalg.norm(radar_sensor_points[:, point_index]))
                colour = (0, 255, 0) if distance < 30 else (0, 220, 255) if distance < 60 else (255, 170, 0)
                cv2.circle(image, (x, y), 3, colour, -1)
                projected_count += 1
        offset_ms = (int(radar_record["ego_pose"]["timestamp"]) - int(camera["ego_pose"]["timestamp"])) / 1000
        cv2.rectangle(image, (0, 0), (image.shape[1], 52), (15, 15, 15), -1)
        cv2.putText(image, f"frame={index}  radar raw={len(radar)} projected={projected_count}  GT boxes={gt_count}  radar-camera offset={offset_ms:+.1f}ms", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        raw_camera = titled(cv2.resize(original_image, (960, 540)), "CAMERA ONLY", "Visual context and object appearance; range/velocity are uncertain from one frame.")
        projected_camera = titled(cv2.resize(image, (960, 540)), "CAMERA + PROJECTED RADAR", "Green/yellow/blue dots: radar range bands. Yellow/blue 3D boxes: evaluation-only GT.")
        bev = draw_bev(radar, (960, 540))
        explainer = draw_explainer(radar, projected_count, gt_count, offset_ms, (960, 540))
        rendered = cv2.vconcat([cv2.hconcat([raw_camera, projected_camera]), cv2.hconcat([bev, explainer])])
        if writer is None:
            writer = cv2.VideoWriter(str(arguments.output), cv2.VideoWriter_fourcc(*"mp4v"), arguments.fps, (rendered.shape[1], rendered.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"cannot write {arguments.output}")
        if index == 0:
            arguments.preview.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(arguments.preview), rendered):
                raise RuntimeError(f"cannot write {arguments.preview}")
        writer.write(rendered)
    assert writer is not None
    writer.release()
    print(arguments.output.resolve())
    print(arguments.preview.resolve())


if __name__ == "__main__":
    main()
