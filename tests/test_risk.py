import unittest

from guardian_perception import (
    BoundingBox,
    Detection,
    RiskConfig,
    RiskEngine,
    RiskLevel,
    TrackObservation,
)
from guardian_perception.adapters.yolo import yolo_rows_to_detections


def observation(
    frame: int,
    height: float,
    *,
    track_id: str = "lead",
    center_x: float = 640,
    confidence: float = 0.9,
) -> TrackObservation:
    width = height * 1.5
    left = center_x - width / 2
    return TrackObservation(
        track_id=track_id,
        detection=Detection(
            frame_id=frame,
            timestamp_ms=frame * 100,
            bbox=BoundingBox(left, 600 - height, left + width, 600),
            class_name="car",
            detector_confidence=confidence,
        ),
        tracker_confidence=confidence,
        frame_width=1280,
        frame_height=720,
    )


class RiskEngineTests(unittest.TestCase):
    def test_yolo_adapter_normalizes_detector_contract(self) -> None:
        detections = yolo_rows_to_detections(
            [(10.0, 20.0, 50.0, 100.0, 0.88, 2)],
            {2: "car"},
            frame_id=7,
            timestamp_ms=230,
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "car")
        self.assertEqual(detections[0].frame_id, 7)
        self.assertEqual(detections[0].bbox.height, 80.0)

    def test_requires_stable_history_before_warning(self) -> None:
        engine = RiskEngine(RiskConfig(min_track_age_frames=4, warning_persistence_frames=2))
        decisions = [engine.evaluate_frame([observation(frame, 100 + frame * 5)]) for frame in range(4)]
        self.assertTrue(all(decision.risk is RiskLevel.NONE for decision in decisions[:3]))
        self.assertIsNotNone(decisions[-1].ttc_estimate_s)

    def test_warns_for_stable_rapid_approach(self) -> None:
        engine = RiskEngine(RiskConfig(min_track_age_frames=4, warning_persistence_frames=2))
        decisions = [engine.evaluate_frame([observation(frame, 100 + frame * 20)]) for frame in range(6)]
        self.assertIs(decisions[-1].risk, RiskLevel.WARNING)
        self.assertEqual(decisions[-1].selected_track_id, "lead")

    def test_rejects_adjacent_lane_target(self) -> None:
        engine = RiskEngine(RiskConfig(min_track_age_frames=4))
        decisions = [
            engine.evaluate_frame([observation(frame, 100 + frame * 20, center_x=100)])
            for frame in range(6)
        ]
        self.assertTrue(all(decision.selected_track_id is None for decision in decisions))

    def test_rejects_non_approaching_target(self) -> None:
        engine = RiskEngine(RiskConfig(min_track_age_frames=4))
        decisions = [engine.evaluate_frame([observation(frame, 160)]) for frame in range(6)]
        self.assertIs(decisions[-1].risk, RiskLevel.NONE)
        self.assertIsNone(decisions[-1].selected_track_id)

    def test_prefers_warning_over_caution(self) -> None:
        engine = RiskEngine(RiskConfig(min_track_age_frames=4, warning_persistence_frames=1))
        decision = None
        for frame in range(6):
            decision = engine.evaluate_frame(
                [
                    observation(frame, 100 + frame * 8, track_id="caution"),
                    observation(frame, 100 + frame * 20, track_id="warning"),
                ]
            )
        self.assertEqual(decision.selected_track_id, "warning")
        self.assertIs(decision.risk, RiskLevel.WARNING)


if __name__ == "__main__":
    unittest.main()
