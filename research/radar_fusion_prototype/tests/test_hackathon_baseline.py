import unittest

from guardian_perception import CameraMeasurement, RadarMeasurement
from hackathon_mvp.lead_object_system import LeadObjectCollisionSystem, LeadObjectInput, LeadRisk


def camera() -> CameraMeasurement:
    return CameraMeasurement("lead-1", 55.0, 0.0, 0.80, 0.80)


def radar(*, association: float = 0.90, range_m: float = 20.0, closing_speed_mps: float = 8.0) -> RadarMeasurement:
    return RadarMeasurement("radar-1", range_m, closing_speed_mps, 0.90, association)


def input_for(frame_id: int, *, radar_measurement: RadarMeasurement | None) -> LeadObjectInput:
    return LeadObjectInput(frame_id, frame_id * 500, camera(), radar_measurement, True, 3)


class HackathonBaselineTests(unittest.TestCase):
    def test_trusted_radar_is_primary_for_ttc_despite_camera_proxy_range(self) -> None:
        system = LeadObjectCollisionSystem(warning_persistence_frames=1)
        decision = system.evaluate(input_for(0, radar_measurement=radar()))
        self.assertEqual(decision.risk, LeadRisk.WARNING)
        self.assertTrue(decision.radar_used)
        self.assertEqual(decision.evidence_status, "radar_confirmed")
        self.assertAlmostEqual(decision.fused_range_m, 20.0)
        self.assertAlmostEqual(decision.fused_ttc_s, 2.5)

    def test_untrusted_radar_does_not_become_warning(self) -> None:
        system = LeadObjectCollisionSystem(warning_persistence_frames=1)
        decision = system.evaluate(input_for(0, radar_measurement=radar(association=0.20)))
        self.assertEqual(decision.risk, LeadRisk.UNCERTAIN)
        self.assertFalse(decision.radar_used)
        self.assertEqual(decision.evidence_status, "uncertain")

    def test_stable_camera_without_radar_is_visible_but_not_warning(self) -> None:
        system = LeadObjectCollisionSystem(warning_persistence_frames=1)
        decision = system.evaluate(input_for(0, radar_measurement=None))
        self.assertEqual(decision.risk, LeadRisk.UNCERTAIN)
        self.assertEqual(decision.evidence_status, "camera_only")

    def test_warning_requires_two_consecutive_low_ttc_observations(self) -> None:
        system = LeadObjectCollisionSystem(warning_persistence_frames=2)
        first = system.evaluate(input_for(0, radar_measurement=radar()))
        second = system.evaluate(input_for(1, radar_measurement=radar(range_m=19.0)))
        self.assertEqual(first.risk, LeadRisk.CAUTION)
        self.assertEqual(second.risk, LeadRisk.WARNING)


if __name__ == "__main__":
    unittest.main()
