import unittest

from guardian_perception.association import AssociationCandidate, greedy_one_to_one_assignment


class AssociationTests(unittest.TestCase):
    def test_global_assignment_prevents_cluster_reuse(self):
        candidates = [
            AssociationCandidate("car-a", 1, 0.9, 1.0, 0.8),
            AssociationCandidate("car-b", 1, 0.8, 1.0, 0.7),
            AssociationCandidate("car-b", 2, 0.7, 0.9, 0.6),
        ]
        assignments = greedy_one_to_one_assignment(candidates, minimum_score=0.5)
        self.assertEqual(assignments["car-a"].cluster_id, 1)
        self.assertEqual(assignments["car-b"].cluster_id, 2)

    def test_low_score_pair_is_left_unassigned(self):
        assignments = greedy_one_to_one_assignment(
            [AssociationCandidate("car-a", 1, 0.4, 0.4, 0.4)], minimum_score=0.5
        )
        self.assertEqual(assignments, {})
