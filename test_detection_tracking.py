"""Behavioral tests for detector-independent tracking and ball recovery."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from detection_io import Detection, RAW_FIELDS, BASE_FIELDS, read_detections
from player_tracking import PlayerTracker
from calibration import SceneGeometry
from ball_tracking import PitchBallTracker
from pitch_context import PitchBackground
from ball_detection import BallSearch, tile_starts, suppress_duplicates
from track_csv import replay, parser


def config():
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    result = YAML.load(check_yaml('botsort.yaml'))
    result['gmc_method'] = 'none'
    return result


def person(frame, confidence=.8, box=(10,10,30,50)):
    return Detection(frame,frame/30,'person',confidence,box)


class PlayerTests(unittest.TestCase):
    def test_unknown_backend_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            PlayerTracker({'tracker_type':'unknown'})

    def test_weak_observation_keeps_identity(self):
        tracker = PlayerTracker(config())
        first,_ = tracker.update(0,[person(0)])
        second,_ = tracker.update(1,[person(1,.15,(11,10,31,50))])
        self.assertEqual(first[0]['track_id'],second[0]['track_id'])
        self.assertEqual(tracker.last_diagnostics['low_conf_associated'],1)

    def test_new_unconfirmed_person_remains_visible_without_fabricated_id(self):
        tracker = PlayerTracker(config())
        tracker.update(0,[person(0)])
        rows,_ = tracker.update(1,[person(1),person(1,.9,(100,10,120,50))])
        pending = [r for r in rows if r['state']=='detected_unassigned']
        self.assertEqual(len(pending),1)
        self.assertEqual(pending[0]['track_id'],-1)
        rows,_ = tracker.update(2,[person(2),person(2,.9,(101,10,121,50))])
        self.assertEqual(sum(r['track_id']>=0 for r in rows),2)

    def test_predictions_are_separate_and_expire(self):
        tracker = PlayerTracker(config(),hold_frames=2)
        first,_ = tracker.update(0,[person(0)])
        rows,predicted = tracker.update(1,[])
        self.assertEqual(rows,[])
        self.assertEqual(predicted[0]['track_id'],first[0]['track_id'])
        self.assertEqual(predicted[0]['state'],'predicted')
        self.assertEqual(tracker.update(3,[]),( [],[] ))

    def test_pitch_filter_uses_foot_not_box_center(self):
        tracker = PlayerTracker(config())
        geometry = SceneGeometry({'pitch_polygon':[[0,0],[100,0],[100,100],[0,100]]})
        rows,_ = tracker.update(0,[person(0,box=(10,80,30,110))],geometry)
        self.assertEqual(rows,[])
        self.assertEqual(tracker.last_diagnostics['outside_pitch'],1)

    def test_pitch_polygon_can_be_derived_from_configured_dimensions(self):
        geometry = SceneGeometry(dict(image_points=[[0,0],[100,0],[100,100],[0,100]],
                                      pitch_points_m=[[0,0],[10,0],[10,20],[0,20]],pitch_size_m=[10,20]))
        self.assertTrue(geometry.contains((50,50)))
        self.assertFalse(geometry.contains((101,50)))


class CsvTests(unittest.TestCase):
    def test_original_schema_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.csv'
            with path.open('w',newline='') as stream:
                writer = csv.writer(stream);writer.writerow(BASE_FIELDS);writer.writerow(person(0).csv_row()[:11])
            self.assertEqual(read_detections(path)[0].box,person(0).box)

    def test_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.csv'
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(RAW_FIELDS)
                row=person(0).csv_row();row[4]='nan';writer.writerow(row)
            with self.assertRaises(ValueError):
                read_detections(path)

    def test_scene_metadata_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.csv'
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(RAW_FIELDS);writer.writerow(person(0).csv_row())
            path.with_suffix('.meta.json').write_text(json.dumps(dict(fps=30,frame_count=2,stride=1,
                scenes=[dict(scene_id=1,start_frame=0,end_frame=2)])))
            args=parser().parse_args(['--detections',str(path),'--output',str(Path(directory)/'out')])
            with self.assertRaises(ValueError):
                replay(args)

    def test_replay_without_video_and_empty_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.csv'
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(RAW_FIELDS)
                writer.writerows([person(0).csv_row(),person(1,.15).csv_row()])
            path.with_suffix('.meta.json').write_text(json.dumps(dict(fps=30,frame_count=3,stride=1,
                image_size=[200,100],person_det_conf=.1,record_kind='raw_model_observations',
                scenes=[dict(scene_id=0,start_frame=0,end_frame=3)])))
            output=Path(directory)/'out'
            args=parser().parse_args(['--detections',str(path),'--output',str(output)])
            with contextlib.redirect_stdout(io.StringIO()):
                summary=replay(args)
            with (output/'detections.csv').open(encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['track_id'],rows[1]['track_id'])
            self.assertEqual(summary['processed_frames'],3)
            self.assertEqual(summary['predicted'],1)
            self.assertFalse((output/'tracked.mp4').exists())


class RecoveryTests(unittest.TestCase):
    def test_far_candidate_requires_consistent_reacquisition(self):
        image=np.zeros((100,160,3),np.uint8)
        background=PitchBackground(image)
        tracker=PitchBallTracker(hits=1,radius=12)
        first=tracker.update(0,[([20,60,28,68],.8)],image,background)[0][2]
        self.assertEqual(tracker.update(1,[([110,60,118,68],.8)],image,background),[])
        self.assertEqual(tracker.update(2,[([111,60,119,68],.8)],image,background),[])
        row=tracker.update(3,[([112,60,120,68],.8)],image,background)[0]
        self.assertNotEqual(row[2],first)

    def test_small_white_spot_inside_padded_box_is_detected(self):
        image=np.zeros((100,100,3),np.uint8)
        image[:]=(30,110,30);image[48:52,48:52]=255
        self.assertTrue(PitchBackground(image).evidence(image,[44,44,56,56])['static_marking'])

    def test_tiles_cover_edges_and_nms_keeps_best(self):
        starts=tile_starts(640,270)
        self.assertEqual(starts[0],0);self.assertEqual(starts[-1]+270,640)
        a=Detection(0,0,'sports ball',.4,(10,10,20,20),detector_pass='full')
        b=Detection(0,0,'sports ball',.8,(11,10,21,20),detector_pass='roi')
        self.assertEqual(suppress_duplicates([a,b]),[b])

    def test_challenger_gets_roi_even_when_active_has_detection(self):
        class Detector:
            def __init__(self): self.calls=[]
            def detect_ball_crop(self,crop,frame,time_s,scene,confidence,imgsz,offset,source):
                self.calls.append(offset)
                x,y=offset
                return [Detection(frame,time_s,'sports ball',.8,(x+5,y+5,x+10,y+10),scene,source)]
        detector=Detector();search=BallSearch(detector,interval=0)
        image=np.zeros((100,200,3),np.uint8);background=PitchBackground(image)
        active=dict(center=np.array([30.,50.]),velocity=np.zeros(2),frame=0,hits=3,confidence=.8)
        challenger=dict(center=np.array([150.,50.]),velocity=np.zeros(2),frame=0,hits=1,confidence=.4)
        class Tracker: pass
        tracker=Tracker();tracker.active=active;tracker.hypotheses=[active,challenger];tracker.gap=8;tracker.radius=25
        full=[Detection(1,1/30,'sports ball',.8,(26,46,34,54))]
        result=search.propose(image,1,1/30,0,full,tracker,background,.15)
        self.assertEqual(len(detector.calls),1)
        self.assertTrue(any(r.detector_pass=='roi' and r.box[0]>100 for r in result))


if __name__=='__main__': unittest.main()
