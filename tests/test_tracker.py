import unittest

from guardian_perception import BoundingBox, Detection, IouTracker, KalmanTracker
from guardian_perception.tracker import intersection_over_union


def detection(timestamp_ms: int, left: float, *, class_name: str = "car") -> Detection:
    return Detection(
        frame_id=timestamp_ms // 200,
        timestamp_ms=timestamp_ms,
        bbox=BoundingBox(left, 200, left + 100, 300),
        class_name=class_name,
        detector_confidence=0.9,
    )


class IouTrackerTests(unittest.TestCase):
    def test_iou_is_zero_for_disjoint_boxes(self) -> None:
        self.assertEqual(
            intersection_over_union(BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)),
            0.0,
        )

    def test_matching_detection_keeps_track_id(self) -> None:
        tracker = IouTracker()
        first = tracker.update([detection(0, 100)], frame_width=1280, frame_height=720)
        second = tracker.update([detection(200, 108)], frame_width=1280, frame_height=720)
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_expired_track_gets_new_id(self) -> None:
        tracker = IouTracker(maximum_gap_ms=400)
        first = tracker.update([detection(0, 100)], frame_width=1280, frame_height=720)
        second = tracker.update([detection(500, 108)], frame_width=1280, frame_height=720)
        self.assertNotEqual(first[0].track_id, second[0].track_id)

    def test_same_position_different_classes_do_not_merge(self) -> None:
        tracker = IouTracker()
        observations = tracker.update(
            [detection(0, 100, class_name="car"), detection(0, 100, class_name="truck")],
            frame_width=1280,
            frame_height=720,
        )
        self.assertEqual(len({item.track_id for item in observations}), 2)


class KalmanTrackerTests(unittest.TestCase):
    def test_predicted_motion_keeps_id_when_iou_only_would_fail(self) -> None:
        tracker = KalmanTracker(maximum_gap_ms=750)
        first = tracker.update([detection(0, 100)], frame_width=1280, frame_height=720)
        second = tracker.update([detection(500, 130)], frame_width=1280, frame_height=720)
        third = tracker.update([detection(1000, 180)], frame_width=1280, frame_height=720)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].track_id, third[0].track_id)

    def test_same_position_different_classes_do_not_merge(self) -> None:
        tracker = KalmanTracker()
        observations = tracker.update(
            [detection(0, 100, class_name="car"), detection(0, 100, class_name="truck")],
            frame_width=1280,
            frame_height=720,
        )
        self.assertEqual(len({item.track_id for item in observations}), 2)


if __name__ == "__main__":
    unittest.main()
