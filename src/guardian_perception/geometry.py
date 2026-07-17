"""Geometry helpers with no image-processing dependency."""

from .types import BoundingBox


def point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    """Return whether ``point`` is inside or on the boundary of a polygon."""

    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        intersects = (current_y > y) != (previous_y > y)
        if intersects:
            crossing_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def bottom_center_is_in_corridor(
    box: BoundingBox,
    frame_width: int,
    frame_height: int,
    corridor: tuple[tuple[float, float], ...],
) -> bool:
    """Evaluate the bottom-centre of a box against a normalized ego corridor."""

    x, y = box.bottom_center
    return point_in_polygon((x / frame_width, y / frame_height), corridor)
