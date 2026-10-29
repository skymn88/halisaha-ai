import unittest
import numpy as np
from tracking_helpers import TemporalBall, IDTransitions, jersey_feature


class TemporalTests(unittest.TestCase):
    def test_stationary_ball_away_from_markings(self):
        image = np.zeros((80,80,3),np.uint8)
        t = TemporalBall()
        statuses = [t.update(f,[([20,20,28,28],.8)],image)[0][3] for f in range(5)]
        self.assertEqual(statuses[2:],['accepted']*3)

    def test_marking_and_stopped_confirmed_ball(self):
        image = np.full((80,80,3),255,np.uint8)
        t = TemporalBall()
        for f in range(5):
            self.assertEqual(t.update(f,[([20,20,28,28],.8)],image)[0][3],'marking_unconfirmed')
        self.assertEqual(t.update(5,[([26,20,34,28],.8)],image)[0][3],'accepted')
        self.assertEqual(t.update(6,[([26,20,34,28],.8)],image)[0][3],'accepted')
        self.assertEqual(t.update(20,[([26,20,34,28],.8)],image)[0][3],'marking_unconfirmed')

    def test_no_duplicate_association_or_fabricated_detection(self):
        t = TemporalBall(hits=1)
        image = np.zeros((80,80,3),np.uint8)
        rows = t.update(0,[([20,20,28,28],.8),([21,20,29,28],.7)],image)
        self.assertNotEqual(rows[0][2],rows[1][2])
        self.assertEqual(t.update(1,[],image),[])

    def test_possible_id_transition(self):
        t = IDTransitions()
        self.assertEqual(t.update(0,[(1,[10,10,20,30])]),[])
        self.assertEqual(t.update(1,[(2,[11,10,21,30])])[0][1:3],[1,2])

    def test_color_descriptor_has_no_team_assumption(self):
        values = jersey_feature(np.zeros((40,40,3),np.uint8),[0,0,30,40])
        self.assertEqual(len(values),3)


if __name__ == '__main__': unittest.main()
