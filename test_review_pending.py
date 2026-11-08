"""Unselected proposals must survive partial frame review as suggestions."""
import tempfile
import unittest
from review_store import ReviewStore
from test_review_workflow import fixture


class PendingProposalTests(unittest.TestCase):
    def test_omitted_proposals_are_not_labels_and_remain_available(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(fixture(directory))
            store.save_frame(dict(window_id='w', frame=0, annotation_source='human',
                                  reviewer='test', reviewed=True, ball_state='uncertain',
                                  people_complete=False, identities_complete=False,
                                  detections=[]))
            result = store.frame('w', 0)
            self.assertEqual(len(result['detections']), 3)
            self.assertTrue(all(r['ui_pending'] for r in result['detections']))
            from review_schema import read_rows
            self.assertEqual(read_rows(store.directory/'detection_labels.csv'), [])
