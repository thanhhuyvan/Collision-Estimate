"""Minimal radar replay adapter for a lead-object hackathon smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from guardian_perception import RadarMeasurement, RadarPoint, RadarCluster, RadarClusterTracker, cluster_radar_points

def rotation_matrix(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)


def transform_sensor_to_global(points: np.ndarray, calibration: dict, ego_pose: dict) -> np.ndarray:
    in_ego = rotation_matrix(calibration["rotation"]) @ points + np.asarray(calibration["translation"])[:, None]
    return rotation_matrix(ego_pose["rotation"]) @ in_ego + np.asarray(ego_pose["translation"])[:, None]


def transform_global_to_camera(points: np.ndarray, calibration: dict, ego_pose: dict) -> np.ndarray:
    in_ego = rotation_matrix(ego_pose["rotation"]).T @ (points - np.asarray(ego_pose["translation"])[:, None])
    return rotation_matrix(calibration["rotation"]).T @ (in_ego - np.asarray(calibration["translation"])[:, None])


def transform_clusters_to_global(clusters: list[RadarCluster], radar_record: dict) -> list[RadarCluster]:
    """Express radar-cluster centroids and compensated velocities in one world frame."""

    if not clusters:
        return []
    calibration, ego_pose = radar_record["calibration"], radar_record["ego_pose"]
    local_positions = np.array([[cluster.x_m, cluster.y_m, cluster.z_m] for cluster in clusters], dtype=float).T
    global_positions = transform_sensor_to_global(local_positions, calibration, ego_pose)
    rotation = rotation_matrix(ego_pose["rotation"]) @ rotation_matrix(calibration["rotation"])
    global_velocities = rotation @ np.array([[cluster.vx_mps, cluster.vy_mps, 0.0] for cluster in clusters], dtype=float).T
    return [
        RadarCluster(
            cluster_id=cluster.cluster_id,
            point_indexes=cluster.point_indexes,
            x_m=float(global_positions[0, index]),
            y_m=float(global_positions[1, index]),
            z_m=float(global_positions[2, index]),
            vx_mps=float(global_velocities[0, index]),
            vy_mps=float(global_velocities[1, index]),
        )
        for index, cluster in enumerate(clusters)
    ]


def project_camera(points: np.ndarray, intrinsic: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    valid = points[2] > 0.5
    pixels = np.zeros((2, points.shape[1]), dtype=float)
    pixels[:, valid] = (np.asarray(intrinsic) @ points[:, valid])[:2] / points[2, valid]
    return pixels, valid


def read_pcd(path: Path) -> np.ndarray:
    """Read nuScenes binary PCD radar records without a point-cloud dependency."""

    with path.open("rb") as source:
        header_lines = []
        while True:
            line = source.readline().decode("ascii").strip()
            header_lines.append(line)
            if line.startswith("DATA"):
                break
        payload = source.read()
    header = {line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1] for line in header_lines if " " in line}
    type_codes = {("F", 4): "<f4", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4"}
    dtype = np.dtype([(field, type_codes[(kind, int(size))] if int(count) == 1 else (type_codes[(kind, int(size))], int(count))) for field, size, kind, count in zip(header["FIELDS"].split(), header["SIZE"].split(), header["TYPE"].split(), header["COUNT"].split(), strict=True)])
    return np.frombuffer(payload, dtype=dtype, count=int(header["POINTS"]))


@dataclass(frozen=True)
class RadarLeadResult:
    measurement: RadarMeasurement | None
    raw_point_count: int
    projected_point_count: int
    matched_point_count: int
    association_confidence: float = 0.0
    temporal_track_age: int = 0
    temporal_consistency: float = 0.0


@dataclass
class _RadarTemporalState:
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    timestamp_ms: int


class RadarTemporalFilter:
    """Track radar clusters and score how physically consistent each sweep is.

    The score is deliberately a gate, not a learned claim of identity. A cluster needs
    three consistent sightings before it can receive full association confidence.
    """

    def __init__(self) -> None:
        self._tracker = RadarClusterTracker(max_match_distance_m=4.0, max_age_ms=1_200)
        self._states: dict[int, _RadarTemporalState] = {}

    def update(self, clusters: list[RadarCluster], timestamp_ms: int) -> dict[int, tuple[int, float]]:
        assignments = self._tracker.update(clusters, timestamp_ms)
        evidence: dict[int, tuple[int, float]] = {}
        for cluster in clusters:
            track_id, age = assignments[cluster.cluster_id]
            previous = self._states.get(track_id)
            if previous is None:
                consistency = 0.30
            else:
                dt_s = max(0.001, (timestamp_ms - previous.timestamp_ms) / 1000)
                predicted_x = previous.x_m + previous.vx_mps * dt_s
                predicted_y = previous.y_m + previous.vy_mps * dt_s
                position_error = float(np.hypot(cluster.x_m - predicted_x, cluster.y_m - predicted_y))
                velocity_error = float(np.hypot(cluster.vx_mps - previous.vx_mps, cluster.vy_mps - previous.vy_mps))
                # Four metres per radar interval is permissive enough for sparse
                # radar, while still suppressing a sudden switch to a nearby vehicle.
                consistency = max(0.0, 1.0 - position_error / 4.0) * max(0.0, 1.0 - velocity_error / 8.0)
            self._states[track_id] = _RadarTemporalState(cluster.x_m, cluster.y_m, cluster.vx_mps, cluster.vy_mps, timestamp_ms)
            evidence[cluster.cluster_id] = (age, consistency)
        return evidence


def associate_radar_to_lead(
    *,
    data_root: Path,
    radar_record: dict,
    camera_record: dict,
    bbox_xyxy: tuple[float, float, float, float] | None,
    return_id: str,
    padding_px: int = 12,
    mode: str = "point_box",
    timestamp_ms: int | None = None,
    temporal_filter: RadarTemporalFilter | None = None,
) -> RadarLeadResult:
    """Project one radar sweep and return a conservative object-level measurement.

    This geometric replay adapter is deliberately simple: its point count becomes an
    association-confidence *input*, never a guarantee that a radar point is correct.
    """

    if mode not in {"point_box", "cluster_geometry", "cluster_geometry_temporal", "cluster_geometry_pose_temporal"}:
        raise ValueError("unsupported association mode")
    radar = read_pcd(data_root / radar_record["file"])
    raw_points = np.vstack((radar["x"], radar["y"], radar["z"]))
    global_points = transform_sensor_to_global(raw_points, radar_record["calibration"], radar_record["ego_pose"])
    camera_points = transform_global_to_camera(global_points, camera_record["calibration"], camera_record["ego_pose"])
    pixels, visible = project_camera(camera_points, camera_record["calibration"]["camera_intrinsic"])
    projected = np.flatnonzero(visible)
    clusters: list[RadarCluster] | None = None
    temporal_evidence: dict[int, tuple[int, float]] = {}
    if mode in {"cluster_geometry_temporal", "cluster_geometry_pose_temporal"}:
        if temporal_filter is None or timestamp_ms is None:
            raise ValueError("temporal association requires temporal_filter and timestamp_ms")
        clusters = _clusters_from_radar(radar)
        tracking_clusters = transform_clusters_to_global(clusters, radar_record) if mode == "cluster_geometry_pose_temporal" else clusters
        temporal_evidence = temporal_filter.update(tracking_clusters, timestamp_ms)
    if bbox_xyxy is None:
        return RadarLeadResult(None, len(radar), len(projected), 0)
    left, top, right, bottom = bbox_xyxy
    matched = np.flatnonzero(
        visible
        & (pixels[0] >= left - padding_px)
        & (pixels[0] <= right + padding_px)
        & (pixels[1] >= top - padding_px)
        & (pixels[1] <= bottom + padding_px)
    )
    if mode in {"cluster_geometry", "cluster_geometry_temporal", "cluster_geometry_pose_temporal"}:
        return _associate_cluster_geometry(
            radar=radar,
            raw_points=raw_points,
            pixels=pixels,
            visible=visible,
            bbox_xyxy=bbox_xyxy,
            raw_point_count=len(radar),
            projected_point_count=len(projected),
            return_id=return_id,
            padding_px=padding_px,
            clusters=clusters,
            temporal_evidence=temporal_evidence,
        )
    if len(matched) == 0:
        return RadarLeadResult(None, len(radar), len(projected), 0)
    selected = radar[matched]
    range_m = float(np.median(np.linalg.norm(raw_points[:, matched], axis=0)))
    closing_speed_mps = max(0.0, -float(np.median(selected["vx_comp"])))
    signal_quality = float(np.mean(selected["is_quality_valid"].astype(float)))
    # A count-based geometry score is intentionally capped below certainty. Temporal
    # association must replace this heuristic after the hackathon MVP.
    association_confidence = min(0.90, 0.25 + 0.12 * len(matched)) * signal_quality
    measurement = RadarMeasurement(return_id, range_m, closing_speed_mps, signal_quality, association_confidence)
    return RadarLeadResult(measurement, len(radar), len(projected), len(matched), association_confidence)


def _clusters_from_radar(radar: np.ndarray) -> list[RadarCluster]:
    quality = radar["is_quality_valid"].astype(bool)
    points = [
        RadarPoint(float(radar["x"][index]), float(radar["y"][index]), float(radar["z"][index]), float(radar["vx_comp"][index]), float(radar["vy_comp"][index]), bool(quality[index]))
        for index in range(len(radar))
    ]
    return cluster_radar_points(points, radius_m=1.5, min_points=2)


def _associate_cluster_geometry(
    *, radar: np.ndarray, raw_points: np.ndarray, pixels: np.ndarray, visible: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float], raw_point_count: int,
    projected_point_count: int, return_id: str, padding_px: int,
    clusters: list[RadarCluster] | None = None,
    temporal_evidence: dict[int, tuple[int, float]] | None = None,
) -> RadarLeadResult:
    """Associate one coherent radar cluster, never an arbitrary mixture of points.

    Clustering happens in radar Cartesian coordinates (metres); image projection only
    provides camera support. This is a small BEV-first ablation, not full 3D detection.
    """
    clusters = clusters if clusters is not None else _clusters_from_radar(radar)
    left, top, right, bottom = bbox_xyxy
    box_diagonal = max(1.0, float(np.hypot(right - left, bottom - top)))
    box_center = np.array([(left + right) / 2, (top + bottom) / 2])
    candidates: list[tuple[float, tuple[int, ...], float, int]] = []
    for cluster in clusters:
        indexes = np.asarray(cluster.point_indexes, dtype=int)
        cluster_visible = indexes[visible[indexes]]
        if len(cluster_visible) == 0:
            continue
        inside = cluster_visible[
            (pixels[0, cluster_visible] >= left - padding_px)
            & (pixels[0, cluster_visible] <= right + padding_px)
            & (pixels[1, cluster_visible] >= top - padding_px)
            & (pixels[1, cluster_visible] <= bottom + padding_px)
        ]
        support = len(inside) / len(cluster_visible)
        if support < 0.35:
            continue
        cluster_center = np.median(pixels[:, cluster_visible], axis=1)
        center_similarity = max(0.0, 1.0 - float(np.linalg.norm(cluster_center - box_center)) / box_diagonal)
        score = 0.70 * support + 0.30 * center_similarity
        candidates.append((score, tuple(int(index) for index in inside), support, cluster.cluster_id))
    if not candidates:
        return RadarLeadResult(None, raw_point_count, projected_point_count, 0)
    score, selected_indexes, support, cluster_id = max(candidates, key=lambda item: item[0])
    if score < 0.55:
        return RadarLeadResult(None, raw_point_count, projected_point_count, 0)
    indexes = np.asarray(selected_indexes, dtype=int)
    range_m = float(np.median(np.linalg.norm(raw_points[:, indexes], axis=0)))
    closing_speed_mps = max(0.0, -float(np.median(radar["vx_comp"][indexes])))
    signal_quality = float(np.mean(radar["is_quality_valid"][indexes].astype(float)))
    age, temporal_consistency = (temporal_evidence or {}).get(cluster_id, (0, 1.0))
    temporal_gate = min(1.0, age / 3.0) * temporal_consistency
    association_confidence = min(0.90, score * (0.70 + 0.30 * support)) * signal_quality * temporal_gate
    measurement = RadarMeasurement(return_id, range_m, closing_speed_mps, signal_quality, association_confidence)
    return RadarLeadResult(measurement, raw_point_count, projected_point_count, len(indexes), association_confidence, age, temporal_consistency)
