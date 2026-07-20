"""Small, dependency-free radar clustering and temporal cluster tracking."""

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class RadarPoint:
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    quality_valid: bool = True


@dataclass(frozen=True)
class RadarCluster:
    cluster_id: int
    point_indexes: tuple[int, ...]
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float

    @property
    def range_m(self) -> float:
        return hypot(self.x_m, self.y_m)


def cluster_radar_points(
    points: list[RadarPoint], *, radius_m: float = 1.5, min_points: int = 2
) -> list[RadarCluster]:
    """Group quality-valid points with transparent XY connected components."""

    if radius_m <= 0 or min_points < 1:
        raise ValueError("radius_m must be positive and min_points must be at least one")
    unseen = {index for index, point in enumerate(points) if point.quality_valid}
    clusters: list[RadarCluster] = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            source = points[current]
            neighbours = [
                other for other in unseen
                if hypot(source.x_m - points[other].x_m, source.y_m - points[other].y_m) <= radius_m
            ]
            for neighbour in neighbours:
                unseen.remove(neighbour)
                component.add(neighbour)
                frontier.append(neighbour)
        if len(component) < min_points:
            continue
        indexes = tuple(sorted(component))
        selected = [points[index] for index in indexes]
        count = len(selected)
        clusters.append(RadarCluster(
            cluster_id=len(clusters), point_indexes=indexes,
            x_m=sum(point.x_m for point in selected) / count,
            y_m=sum(point.y_m for point in selected) / count,
            z_m=sum(point.z_m for point in selected) / count,
            vx_mps=sum(point.vx_mps for point in selected) / count,
            vy_mps=sum(point.vy_mps for point in selected) / count,
        ))
    return clusters


@dataclass
class _TrackedCluster:
    track_id: int
    x_m: float
    y_m: float
    timestamp_ms: int
    age: int


class RadarClusterTracker:
    """Greedy nearest-centroid tracker for sparse 5 FPS diagnostic replay."""

    def __init__(self, *, max_match_distance_m: float = 4.0, max_age_ms: int = 800) -> None:
        self.max_match_distance_m = max_match_distance_m
        self.max_age_ms = max_age_ms
        self._tracks: dict[int, _TrackedCluster] = {}
        self._next_id = 1

    def update(self, clusters: list[RadarCluster], timestamp_ms: int) -> dict[int, tuple[int, int]]:
        self._tracks = {track_id: track for track_id, track in self._tracks.items() if timestamp_ms - track.timestamp_ms <= self.max_age_ms}
        assignments: dict[int, tuple[int, int]] = {}
        unused_tracks = set(self._tracks)
        for cluster in clusters:
            distance, track_id = min(
                ((hypot(cluster.x_m - self._tracks[candidate].x_m, cluster.y_m - self._tracks[candidate].y_m), candidate) for candidate in unused_tracks),
                default=(float("inf"), -1),
            )
            if distance > self.max_match_distance_m:
                track_id = self._next_id
                self._next_id += 1
                age = 1
            else:
                unused_tracks.remove(track_id)
                age = self._tracks[track_id].age + 1
            self._tracks[track_id] = _TrackedCluster(track_id, cluster.x_m, cluster.y_m, timestamp_ms, age)
            assignments[cluster.cluster_id] = (track_id, age)
        return assignments
