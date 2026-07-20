import unittest

from guardian_perception import CameraMeasurement, RadarMeasurement, ReliabilityAwareFuser


def camera(*, quality: float = 0.9, range_m: float = 30.0, speed: float = 6.0):
    return CameraMeasurement("track-0001", range_m, speed, 0.9, quality)


def radar(
    *, quality: float = 0.9, association: float = 0.9, range_m: float = 29.0, speed: float = 6.2
):
    return RadarMeasurement("radar-001", range_m, speed, quality, association)


class ReliabilityFusionTests(unittest.TestCase):
    def test_high_quality_radar_dominates_noisy_camera_range(self) -> None:
        fused = ReliabilityAwareFuser().fuse(camera(), radar())
        self.assertTrue(fused.radar_used)
        self.assertGreater(fused.radar_weight, fused.camera_weight)
        self.assertAlmostEqual(fused.range_m, 29.0, delta=0.1)
        self.assertAlmostEqual(fused.ttc_s, 29.0 / 6.2, delta=0.1)

    def test_low_camera_quality_increases_radar_influence(self) -> None:
        fused = ReliabilityAwareFuser().fuse(camera(quality=0.2), radar())
        self.assertGreater(fused.radar_weight, 0.99)

    def test_low_association_rejects_radar_and_uses_camera_fallback(self) -> None:
        fused = ReliabilityAwareFuser().fuse(camera(), radar(association=0.2))
        self.assertFalse(fused.radar_used)
        self.assertEqual(fused.range_m, 30.0)
        self.assertIn("association", fused.explanation)

    def test_sensor_disagreement_reduces_reliability(self) -> None:
        agreeing = ReliabilityAwareFuser().fuse(camera(), radar())
        disagreeing = ReliabilityAwareFuser().fuse(camera(), radar(range_m=60.0, speed=14.0))
        self.assertTrue(disagreeing.disagreement)
        self.assertLess(disagreeing.reliability, agreeing.reliability)

    def test_non_closing_target_has_no_ttc(self) -> None:
        fused = ReliabilityAwareFuser().fuse(camera(speed=-0.5))
        self.assertIsNone(fused.ttc_s)


if __name__ == "__main__":
    unittest.main()
