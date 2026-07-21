import unittest

import numpy as np

from guardian_perception.probabilistic import (
    GaussianState,
    linear_update,
    pdaf_position_update,
    pdaf_weights,
    predict_constant_velocity,
    squared_mahalanobis,
)


class ProbabilisticFusionTests(unittest.TestCase):
    def setUp(self):
        self.state = GaussianState(np.array([10.0, 0.0, 2.0, 0.0]), np.eye(4))
        self.position = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.radar_covariance = np.diag([1.0, 1.0])

    def test_constant_velocity_prediction_advances_position(self):
        prediction = predict_constant_velocity(self.state, 0.5)
        self.assertAlmostEqual(prediction.mean[0], 11.0)
        self.assertGreater(prediction.covariance[0, 0], self.state.covariance[0, 0])

    def test_mahalanobis_rejects_distant_measurement(self):
        close = squared_mahalanobis(self.state, np.array([10.5, 0.0]), self.position, self.radar_covariance)
        distant = squared_mahalanobis(self.state, np.array([40.0, 0.0]), self.position, self.radar_covariance)
        self.assertLess(close, distant)

    def test_pdaf_keeps_missed_detection_probability_when_ambiguous(self):
        weights, missed = pdaf_weights([0.1, 0.2])
        self.assertLess(weights[0], 1.0)
        self.assertGreater(missed, 0.0)
        self.assertAlmostEqual(float(weights.sum() + missed), 1.0)

    def test_pdaf_update_moves_toward_weighted_measurement(self):
        weights, _ = pdaf_weights([0.0, 8.0])
        updated = pdaf_position_update(self.state, [np.array([12.0, 0.0]), np.array([30.0, 0.0])], weights, self.radar_covariance)
        self.assertGreater(updated.mean[0], self.state.mean[0])
        self.assertLess(updated.mean[0], 20.0)

    def test_linear_update_reduces_position_uncertainty(self):
        updated = linear_update(self.state, np.array([10.0, 0.0]), self.position, self.radar_covariance)
        self.assertLess(updated.covariance[0, 0], self.state.covariance[0, 0])


if __name__ == "__main__":
    unittest.main()
