import unittest

from guardian_perception.radar import RadarClusterTracker, RadarPoint, cluster_radar_points


class RadarClusterTests(unittest.TestCase):
    def test_groups_nearby_quality_points_and_ignores_noise(self):
        points = [RadarPoint(10, 0, 0, 0, 0), RadarPoint(10.7, 0.1, 0, 0, 0), RadarPoint(30, 0, 0, 0, 0), RadarPoint(10.2, 0.2, 0, 0, 0, False)]
        clusters = cluster_radar_points(points, radius_m=1, min_points=2)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].point_indexes, (0, 1))

    def test_tracker_keeps_id_for_nearby_cluster(self):
        tracker = RadarClusterTracker(max_match_distance_m=2)
        first = cluster_radar_points([RadarPoint(10, 0, 0, 0, 0), RadarPoint(10.5, 0, 0, 0, 0)])
        second = cluster_radar_points([RadarPoint(10.8, 0, 0, 0, 0), RadarPoint(11.2, 0, 0, 0, 0)])
        self.assertEqual(tracker.update(first, 0)[0][0], tracker.update(second, 200)[0][0])
