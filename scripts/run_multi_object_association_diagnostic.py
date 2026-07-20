"""Evaluate multi-object radar association before introducing a learned backbone."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from guardian_perception.association import AssociationCandidate, greedy_one_to_one_assignment
from guardian_perception.radar import RadarClusterTracker, RadarPoint, cluster_radar_points
from render_association_gate_experiment import oracle_object_points
from render_nuscenes_fusion_preview import draw_bev, project_camera, read_pcd, transform_global_to_camera, transform_sensor_to_global
from run_nuscenes_rule_based_fusion_demo import annotation_bbox, panel, text_lines


def candidate_for_box(camera_id: str, cluster, pixels: np.ndarray, visible: np.ndarray, bbox: tuple[int, int, int, int], age: int) -> AssociationCandidate | None:
    x1, y1, x2, y2 = bbox
    ids = list(cluster.point_indexes)
    projected = [index for index in ids if visible[index]]
    if not projected:
        return None
    padding = 12
    support = sum(x1 - padding <= pixels[0, index] <= x2 + padding and y1 - padding <= pixels[1, index] <= y2 + padding for index in projected) / len(ids)
    if support < 0.50:
        return None
    centroid = np.mean(pixels[:, projected], axis=1)
    half_width, half_height = max(1, (x2 - x1) / 2), max(1, (y2 - y1) / 2)
    normalized_distance = np.hypot((centroid[0] - (x1 + x2) / 2) / half_width, (centroid[1] - (y1 + y2) / 2) / half_height)
    center_similarity = max(0.0, 1.0 - normalized_distance)
    temporal_stability = min(1.0, age / 3)
    score = 0.65 * support + 0.20 * center_similarity + 0.15 * temporal_stability
    return AssociationCandidate(camera_id, cluster.cluster_id, score, support, center_similarity)


def draw_chart(per_scene: dict[str, dict[str, float]], output: Path) -> None:
    chart = np.full((720, 1280, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(chart, "Multi-object association diagnostic", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(chart, "GT evaluates assignments only. Runtime score uses projection support, box geometry, and cluster age.", (45, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1, cv2.LINE_AA)
    for index, (scene, values) in enumerate(per_scene.items()):
        y = 150 + index * 130
        cv2.putText(chart, scene, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
        for offset, (label, key, colour) in enumerate((("assigned", "assignment_coverage", (0, 200, 255)), ("good", "good_assignment_rate", (0, 230, 100)), ("ambiguous", "ambiguity_rate", (0, 90, 255)))):
            x = 270 + offset * 300
            value = values[key]
            cv2.rectangle(chart, (x, y - 26), (x + 210, y + 8), (60, 60, 60), -1)
            cv2.rectangle(chart, (x, y - 26), (x + int(210 * value), y + 8), colour, -1)
            cv2.putText(chart, f"{label}: {value:.0%}", (x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.46, colour, 1, cv2.LINE_AA)
    cv2.putText(chart, "High ambiguity or low good-assignment rate means: do not fuse TTC; use camera fallback and improve association data processing.", (45, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), chart)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/nuscenes_fusion/multi_object_association.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/nuscenes_fusion/multi_object_association_preview.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/nuscenes_fusion/multi_object_association.jsonl"))
    parser.add_argument("--chart", type=Path, default=Path("outputs/nuscenes_fusion/multi_object_association_chart.png"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/nuscenes_fusion/multi_object_association_summary.json"))
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for path in (args.output, args.preview, args.report, args.chart, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    tracker, previous_scene, writer = RadarClusterTracker(), None, None
    records: list[dict] = []
    with args.report.open("w", encoding="utf-8") as report:
        for frame_index, frame in enumerate(manifest["frames"]):
            scene = frame["scene"]
            if scene != previous_scene:
                tracker, previous_scene = RadarClusterTracker(), scene
            image = cv2.imread(str(args.data_root / frame["camera"]["file"]))
            raw = read_pcd(args.data_root / frame["radar"]["file"])
            points = [RadarPoint(float(p["x"]), float(p["y"]), float(p["z"]), float(p["vx_comp"]), float(p["vy_comp"]), bool(p["is_quality_valid"])) for p in raw]
            clusters = cluster_radar_points(points)
            cluster_tracks = tracker.update(clusters, int(frame["timestamp_us"] // 1000))
            sensor_points = np.vstack((raw["x"], raw["y"], raw["z"]))
            global_points = transform_sensor_to_global(sensor_points, frame["radar"]["calibration"], frame["radar"]["ego_pose"])
            camera_points = transform_global_to_camera(global_points, frame["camera"]["calibration"], frame["camera"]["ego_pose"])
            pixels, visible = project_camera(camera_points, frame["camera"]["calibration"]["camera_intrinsic"])
            objects = []
            for annotation in frame["annotations"]:
                if not annotation["category"].startswith("vehicle."):
                    continue
                bbox = annotation_bbox(annotation, frame["camera"], (image.shape[1], image.shape[0]))
                if bbox is not None:
                    objects.append((annotation, bbox))
            candidates: list[AssociationCandidate] = []
            for object_index, (_, bbox) in enumerate(objects):
                for cluster in clusters:
                    candidate = candidate_for_box(str(object_index), cluster, pixels, visible, bbox, cluster_tracks[cluster.cluster_id][1])
                    if candidate:
                        candidates.append(candidate)
            assignments = greedy_one_to_one_assignment(candidates, minimum_score=0.68)
            ranked_by_object: dict[str, list[AssociationCandidate]] = defaultdict(list)
            for candidate in candidates:
                ranked_by_object[candidate.camera_id].append(candidate)
            ambiguous = sum(len(sorted(items, key=lambda item: item.score, reverse=True)) > 1 and sorted(items, key=lambda item: item.score, reverse=True)[0].score - sorted(items, key=lambda item: item.score, reverse=True)[1].score < 0.10 for items in ranked_by_object.values())
            assigned_precisions: list[float] = []
            for object_index, (annotation, bbox) in enumerate(objects):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(image, f"O{object_index}", (x1, max(58, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)
                assignment = assignments.get(str(object_index))
                if assignment:
                    cluster = clusters[assignment.cluster_id]
                    projected = [point_index for point_index in cluster.point_indexes if visible[point_index]]
                    point_center = np.mean(pixels[:, projected], axis=1).astype(int)
                    box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    oracle_ids = set(np.flatnonzero(oracle_object_points(global_points, annotation)).tolist())
                    precision = len(set(cluster.point_indexes) & oracle_ids) / len(cluster.point_indexes)
                    assigned_precisions.append(precision)
                    line_colour = (0, 255, 0) if precision >= 0.5 else (0, 0, 255)
                    cv2.line(image, box_center, tuple(point_center), line_colour, 2, cv2.LINE_AA)
                    cv2.putText(image, f"C{cluster.cluster_id} {precision:.0%}", tuple(point_center), cv2.FONT_HERSHEY_SIMPLEX, 0.40, line_colour, 1, cv2.LINE_AA)
            for cluster in clusters:
                for point_index in cluster.point_indexes:
                    if visible[point_index]:
                        x, y = np.rint(pixels[:, point_index]).astype(int)
                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            cv2.circle(image, (x, y), 2, (170, 170, 170), -1)
            left = cv2.resize(image, (960, 540))
            cv2.rectangle(left, (0, 0), (960, 48), (14, 14, 14), -1)
            cv2.putText(left, f"{scene} | multi-object association", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            info = panel(960, 540, "GLOBAL ASSIGNMENT DIAGNOSTIC")
            text_lines(info, [
                (f"camera vehicle boxes: {len(objects)} | radar clusters: {len(clusters)}", (255, 80, 255)),
                (f"candidate pairs: {len(candidates)} | one-to-one assignments: {len(assignments)}", (0, 255, 255)),
                (f"ambiguous objects (top-2 delta < 0.10): {ambiguous}", (0, 130, 255)),
                (f"GT eval mean precision of assignments: {np.mean(assigned_precisions) if assigned_precisions else 0:.0%}", (0, 255, 0)),
                ("MAGENTA = camera object; GREY = radar cluster; GREEN/RED line = GT evaluation", (220, 220, 220)),
                ("No line means no confident match → camera fallback, not forced fusion.", (0, 145, 255)),
            ])
            lower = panel(960, 540, "CONFLICT POLICY")
            text_lines(lower, [
                ("One cluster can be assigned to only one object per frame.", (0, 255, 255)),
                ("Low-score or ambiguous pairs remain unassigned.", (0, 145, 255)),
                ("TTC must ignore unassigned / unstable radar measurements.", (0, 130, 255)),
                ("This is data processing, before any larger detector or fusion backbone.", (255, 220, 80)),
            ])
            rendered = cv2.vconcat([cv2.hconcat([left, info]), cv2.hconcat([draw_bev(raw, (960, 540)), lower])])
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080))
            writer.write(rendered)
            if frame_index == 0:
                cv2.imwrite(str(args.preview), rendered)
            record = {"scene": scene, "frame": frame_index, "camera_objects": len(objects), "radar_clusters": len(clusters), "candidate_pairs": len(candidates), "assignments": len(assignments), "ambiguous_objects": int(ambiguous), "mean_assignment_precision": float(np.mean(assigned_precisions)) if assigned_precisions else None}
            records.append(record)
            report.write(json.dumps(record) + "\n")
    assert writer is not None
    writer.release()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["scene"]].append(record)
    per_scene: dict[str, dict[str, float]] = {}
    for scene, rows in grouped.items():
        total_objects = max(1, sum(row["camera_objects"] for row in rows))
        per_scene[scene] = {
            "assignment_coverage": sum(row["assignments"] for row in rows) / total_objects,
            "good_assignment_rate": float(np.mean([(row["mean_assignment_precision"] or 0) >= 0.5 for row in rows])),
            "ambiguity_rate": sum(row["ambiguous_objects"] for row in rows) / total_objects,
        }
    args.summary.write_text(json.dumps({"frames": len(records), "per_scene": per_scene}, indent=2) + "\n", encoding="utf-8")
    draw_chart(per_scene, args.chart)
    print(args.output.resolve())
    print(args.chart.resolve())
    print(args.summary.resolve())


if __name__ == "__main__":
    main()
