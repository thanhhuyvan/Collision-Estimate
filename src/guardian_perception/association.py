"""Explainable one-to-one camera-region to radar-cluster assignment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssociationCandidate:
    camera_id: str
    cluster_id: int
    score: float
    support: float
    center_similarity: float


def greedy_one_to_one_assignment(
    candidates: list[AssociationCandidate], *, minimum_score: float
) -> dict[str, AssociationCandidate]:
    """Choose highest-scoring non-conflicting pairs without forcing uncertain matches."""

    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be in [0, 1]")
    assigned_cameras: set[str] = set()
    assigned_clusters: set[int] = set()
    assignments: dict[str, AssociationCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate.score < minimum_score:
            break
        if candidate.camera_id in assigned_cameras or candidate.cluster_id in assigned_clusters:
            continue
        assignments[candidate.camera_id] = candidate
        assigned_cameras.add(candidate.camera_id)
        assigned_clusters.add(candidate.cluster_id)
    return assignments
