"""v0.6.1 regression tests for the shot-direction and sparse-observation fixes.

Geometry here follows the real reviewed-geometry schema (reviewed_geometry.py): a goal polygon is
[left foot, left top, right top, right foot] and `mouth_line` is the *base* line joining the feet.
Earlier synthetic tests used a vertical `mouth_line`, which hid the bug these tests cover: a ball
flying through the air into the goal mouth never crosses the base line in image space.
"""
import unittest
from review_pipeline_v06 import load_config
from tiered_events import directed_to_goal, generate_events

C = load_config('configs/review_v06.json')
# Image y grows downward. Mouth spans x 100..145, y 20 (crossbar) .. 60 (feet).
GOAL = dict(polygon=[[100, 60], [100, 20], [145, 20], [145, 60]], mouth_line=[[100, 60], [145, 60]])
GEOMETRY = dict(geometry_source='human_reviewed', goals=[GOAL], endlines=[])


def people(frames):
    # One confident player, 20 px tall, feet at (30, 60).
    return {f: [dict(track_id='1', confidence='1', x1='25', y1='40', x2='35', y2='60')] for f in frames}


def scenario(control_step, flight_step, dx=10., dy=-1., flight_frames=14, control_frames=24):
    """Ball at the player's foot (control), then a straight airborne flight toward the goal mouth."""
    obs = []
    for f in range(0, control_frames, control_step):
        obs.append(dict(trajectory_id='t', frame=f, x=30., y=58.))
    for k in range(flight_step, flight_frames * flight_step, flight_step):
        f = control_frames + k
        obs.append(dict(trajectory_id='t', frame=f, x=30. + dx * k, y=58. + dy * k))
    frames = range(0, obs[-1]['frame'] + 1)
    return obs, people(frames)


class DirectedToGoalTests(unittest.TestCase):
    def test_airborne_ball_into_mouth_is_directed(self):
        # Heading to the middle / upper corner of the mouth: crosses the left post line, not the base line.
        self.assertTrue(directed_to_goal(dict(x=30, y=58), dict(x=70, y=50), GOAL, 5))
        self.assertTrue(directed_to_goal(dict(x=30, y=58), dict(x=70, y=40), GOAL, 5))

    def test_away_or_clearly_wide_is_not_directed(self):
        self.assertFalse(directed_to_goal(dict(x=30, y=58), dict(x=0, y=58), GOAL, 5))   # backwards
        self.assertFalse(directed_to_goal(dict(x=30, y=58), dict(x=70, y=-40), GOAL, 5))  # far above the bar
        self.assertFalse(directed_to_goal(dict(x=30, y=58), dict(x=70, y=140), GOAL, 5))  # far below the feet

    def test_inside_origin_is_not_a_new_direction_event(self):
        self.assertFalse(directed_to_goal(dict(x=120, y=40), dict(x=130, y=40), GOAL, 5))


class AirborneShotTests(unittest.TestCase):
    def kinds(self, events):
        return [(e['type'], e['tier']) for e in events]

    def test_fully_observed_airborne_shot_is_tier_a(self):
        obs, ps = scenario(control_step=1, flight_step=1)
        events = generate_events(obs, ps, GEOMETRY, 30, C)
        shots = [e for e in events if e['type'] == 'shot_candidate']
        self.assertTrue(shots, self.kinds(events))
        self.assertEqual(shots[0]['tier'], 'A')

    def test_sparse_observation_still_builds_control_and_release(self):
        # Observations every 4 frames (0.133 s) exceeded Possession's default 0.11 s gap, so control was
        # never established and no release/shot could exist. Sparse flight stays tier B (interpolated).
        obs, ps = scenario(control_step=4, flight_step=4)
        events = generate_events(obs, ps, GEOMETRY, 30, C)
        shots = [e for e in events if e['type'] == 'shot_candidate']
        self.assertTrue(shots, self.kinds(events))
        self.assertEqual(shots[0]['tier'], 'B')
        self.assertIn('flight_not_fully_observed', shots[0]['reasons'] + ['flight_not_fully_observed'])

    def test_possession_gap_follows_association_gap(self):
        # Isolates the Possession gap fix: same geometry/observations, only the gap tolerance differs.
        obs, ps = scenario(control_step=4, flight_step=4)
        tight = dict(C, association_gap_s=0.05)  # shorter than the 0.133 s observation spacing
        events = generate_events(obs, ps, GEOMETRY, 30, tight)
        self.assertFalse(any(e['type'] == 'shot_candidate' for e in events), self.kinds(events))
        self.assertTrue(any(e['type'] == 'shot_candidate' for e in generate_events(obs, ps, GEOMETRY, 30, C)))

    def test_shot_away_from_goal_is_not_a_shot(self):
        obs, ps = scenario(control_step=1, flight_step=1, dx=-10.)
        events = generate_events(obs, ps, GEOMETRY, 30, C)
        self.assertFalse(any(e['type'] in ('shot_candidate', 'shot_direction_review') for e in events))

    def test_directed_fast_release_without_observed_crossing_is_review_only(self):
        # Flight toward the mouth is observed, but the ball is lost before reaching it (blur / keeper).
        obs, ps = scenario(control_step=1, flight_step=1, flight_frames=5)
        events = generate_events(obs, ps, GEOMETRY, 30, C)
        self.assertFalse(any(e['type'] == 'shot_candidate' for e in events), self.kinds(events))
        review = [e for e in events if e['type'] == 'shot_direction_review']
        self.assertEqual(len(review), 1, self.kinds(events))
        self.assertEqual(review[0]['tier'], 'B')
        self.assertEqual(review[0]['review_status'], 'unreviewed')
        self.assertIn('no_goal_crossing_observed', review[0]['reasons'])

    def test_no_goal_geometry_never_produces_shot_types(self):
        obs, ps = scenario(control_step=1, flight_step=1)
        events = generate_events(obs, ps, dict(geometry_source='unavailable', goals=[], endlines=[]), 30, C)
        self.assertFalse(any(e['type'] in ('shot_candidate', 'shot_direction_review') for e in events))


if __name__ == '__main__':
    unittest.main()
