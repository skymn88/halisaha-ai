"""Synthetic regression tests, not human football labels or an accuracy benchmark."""
import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch
import cv2
import numpy as np
from ball_cache import collect_evidence, sha256_file
from ball_tracking import PitchBallTracker
from detection_io import Detection, RAW_FIELDS
from event_candidates import Possession, generate
from evaluate_review import evaluate, match_pairs
from export_trackeval import export
from pitch_context import PitchBackground
from replay_ball import build_csv_cache, replay
from review_app import make_server
from review_schema import TABLES, read_rows, validate_frame, write_rows
from review_store import ReviewStore
from audit_players import audit


def fixture(directory):
    root = Path(directory)
    run = root/'run'
    run.mkdir()
    raw = [Detection(0,0.,'person',.9,(0,0,10,20)),
           Detection(0,0.,'person',.2,(0,0,10,20)),
           Detection(0,0.,'sports ball',.8,(18,18,22,22)),
           Detection(1,.1,'sports ball',.8,(3,3,7,7)),
           Detection(2,.2,'sports ball',.8,(3,3,7,7))]
    with (run/'raw_detections.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(RAW_FIELDS);writer.writerows(r.csv_row() for r in raw)
    with (run/'detections.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(RAW_FIELDS)
        row=raw[0].csv_row();row[2]=11;writer.writerow(row)
        row=raw[1].csv_row();row[2]=12;writer.writerow(row)
    from replay_ball import TRACK_FIELDS,CANDIDATE_FIELDS
    write_rows(run/'ball_track.csv',TRACK_FIELDS,[
        dict(frame=0,time_s=0,scene_id=0,ball_id=-1,state='missing',confidence='',center_x='',center_y=''),
        dict(frame=1,time_s=.1,scene_id=0,ball_id=1,state='observed',confidence=.8,center_x=5,center_y=5),
        dict(frame=2,time_s=.2,scene_id=0,ball_id=1,state='observed',confidence=.8,center_x=5,center_y=5)])
    write_rows(run/'ball_candidates.csv',CANDIDATE_FIELDS,[dict(frame=d.frame,scene_id=0,
        confidence=d.confidence,x1=d.box[0],y1=d.box[1],x2=d.box[2],y2=d.box[3],ball_id=1,
        status='accepted',white_marking_cue=0,background_marking_fraction=0.,foreground_fraction=1.,
        innovation_px='') for d in raw if not d.is_person])
    params=dict(ball_hits=1,ball_gap=8,ball_radius=45,ball_reacquire_hits=3,ball_mode='pitch',
                background_difference=30.,foreground_min=.25)
    (run/'summary.json').write_text(json.dumps(dict(parameters=params)),encoding='utf-8')
    (run/'raw_detections.meta.json').write_text(json.dumps(dict(source_video='unused',fps=10,
        image_size=[40,40],frame_count=3,stride=1,scenes=[dict(scene_id=0,start_frame=0,end_frame=3)])),encoding='utf-8')
    config=dict(person_iou=.5,ball_center_tolerance_px=2.,event_tolerances_s=[.5,1.])
    (root/'evaluation_config.json').write_text(json.dumps(config),encoding='utf-8')
    window=dict(window_id='w',source_id='s',scene_id=0,start_frame=0,end_frame=3,kind='detections',
                selection='synthetic',split='dev',split_group='s:0',angle='synthetic',angle_reviewed=False,clip='none')
    event_window=dict(window,window_id='e',kind='events',selection='random_audit')
    dataset=dict(schema_version=1,sources=[dict(source_id='s',run=str(run),video='unused',fps=10,
        frame_count=3,image_size=[40,40],scenes=[dict(scene_id=0,start_frame=0,end_frame=3)],
        model={'version':'synthetic'},source_sha256='synthetic',cache_sha256='synthetic',invocation={})],
        windows=[window,event_window],evaluation_config_sha256=sha256_file(root/'evaluation_config.json'),
        limitations=['Synthetic test only'])
    (root/'dataset.json').write_text(json.dumps(dataset),encoding='utf-8')
    for name,fields in TABLES.items():write_rows(root/name,fields,[])
    return root


def payload(frame=0, source='human', ball_state='visible'):
    def box(identifier,kind,xy):
        return dict(annotation_id=identifier,object_type=kind,decision='accepted',
                    x1=xy[0],y1=xy[1],x2=xy[2],y2=xy[3],center_x=(xy[0]+xy[2])/2,
                    center_y=(xy[1]+xy[3])/2,proposal_id='',error_groups='on_line' if kind=='ball' else '',
                    fp_group='',identity_id='',team_id='',role='unknown',track_hint='')
    return dict(window_id='w',frame=frame,annotation_source=source,reviewer='synthetic_test',
                reviewed=True,ball_state=ball_state,people_complete=True,identities_complete=False,
                detections=([box('p','player',(0,0,10,20))] if frame==0 else [])+
                ([box('b','ball',(18,18,22,22))] if ball_state=='visible' else []),notes='Synthetic fixture')


class CacheTests(unittest.TestCase):
    def test_online_exports_are_replayable_without_detector_inference(self):
        import run_tracking
        class FakeDetector:
            def __init__(self, *args, **kwargs):pass
            def detect(self, image, frame, time_s, scene_id, *args):
                return [Detection(frame,time_s,'sports ball',.9,(5+frame,10,9+frame,14),scene_id)]
            def provenance(self):return {'implementation':'synthetic_test','weights_sha256':None}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);video=root/'source.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(40,40))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(8):writer.write(np.full((40,40,3),(20,100,20),np.uint8))
            finally:writer.release()
            with patch('run_tracking.YoloDetector',FakeDetector),patch('sys.argv',[
                    'run_tracking.py','--video',str(video),'--output',str(root/'online')]):
                with contextlib.redirect_stdout(io.StringIO()):run_tracking.main()
            result=replay(root/'online/ball_evidence.csv',root/'replay',root/'online')
            self.assertTrue(result['reference_match'])
            self.assertEqual(result['processed_frames'],8)
    def test_image_and_evidence_paths_match_and_recompute_innovation(self):
        image=np.full((60,80,3),(20,100,20),np.uint8)
        bg=PitchBackground(image)
        a,b=PitchBallTracker(hits=1),PitchBallTracker(hits=1)
        for frame in range(4):
            proposals=[((10+frame,10,14+frame,14),.8),((30,30,34,34),.7)]
            evidence=collect_evidence(proposals,image,bg,people=[(0,0,20,30)])
            for row in evidence:row['innovation_px']=99999
            self.assertEqual(a.update(frame,proposals,image,bg,people=[(0,0,20,30)]),
                             b.update_evidence(frame,evidence))
            self.assertEqual(a.last_diagnostics,b.last_diagnostics)
            self.assertNotEqual(b.last_diagnostics[0]['innovation_px'],99999)

    def test_csv_import_and_replay_do_not_open_images_or_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp)
            with patch('cv2.VideoCapture',side_effect=AssertionError('video opened')):
                cache=build_csv_cache(root/'run',root/'cache','pitch_max_core_v1')
                result=replay(cache,root/'replay')
            self.assertEqual(result['candidates'],3)
            self.assertFalse(result['image_or_detector_used'])
            self.assertEqual(result['accepted_ball_rows'],3)

    def test_unknown_historical_feature_definition_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp)
            with self.assertRaises(ValueError):build_csv_cache(root/'run',root/'cache','old_mean_features')

    def test_timeline_scene_mismatch_is_rejected(self):
        from ball_cache import read_evidence
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);path=build_csv_cache(root/'run',root/'cache','pitch_max_core_v1')
            metadata=json.loads(path.with_suffix('.meta.json').read_text())
            metadata['timeline'][0]['scene_id']=3
            path.with_suffix('.meta.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):read_evidence(path)

    def test_image_threshold_changes_require_rebuilding_evidence(self):
        from ball_cache import read_evidence
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);path=build_csv_cache(root/'run',root/'cache','pitch_max_core_v1')
            metadata=json.loads(path.with_suffix('.meta.json').read_text())
            metadata['parameters']['foreground_min']=.75
            path.with_suffix('.meta.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):read_evidence(path)


class ReviewTests(unittest.TestCase):
    def test_uncertain_event_interval_is_not_a_false_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);store=ReviewStore(root)
            store.save_event(dict(event_id='unclear',window_id='e',frame=1,event_type='unknown',
                decision='uncertain',annotation_source='human',reviewer='synthetic_test',
                interval_start_frame=0,interval_end_frame=3))
            store.complete_window(dict(window_id='e',events_complete=True,annotation_source='human',
                                       reviewer='synthetic_test'))
            write_rows(root/'event_candidates.csv',
                       ['candidate_id','window_id','frame','event_type','context_status','actor_id'],
                       [dict(candidate_id='x',window_id='e',frame=1,event_type='shot_candidate',
                             context_status='within_scene',actor_id='')])
            row=evaluate(root,'dev')['event_windows'][0]
            self.assertEqual(row['proposed_events'],0)
            self.assertEqual(row['metrics']['typed_at_0.5s']['fp'],0)
            self.assertEqual(row['excluded_uncertain_intervals'],1)

    def test_duplicate_event_needs_existing_distinct_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            store=ReviewStore(fixture(temp))
            with self.assertRaises(ValueError):
                store.save_event(dict(event_id='a',window_id='e',frame=1,event_type='shot',
                    decision='approved',annotation_source='ai_reviewed',reviewer='synthetic_test',duplicate_of='missing'))
    def test_unreviewed_and_ai_labels_do_not_produce_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp)
            self.assertIsNone(evaluate(root,'dev')['per_window'][0]['selected_ball'])
            store=ReviewStore(root);store.save_frame(payload(source='ai_reviewed'))
            report=evaluate(root,'dev')
            self.assertEqual(report['human_reviewed_frame_rows'],0)
            self.assertEqual(report['ai_reviewed_excluded'],1)
            self.assertIsNone(report['per_window'][0]['raw_ball_candidate_recall'])

    def test_candidate_failure_is_distinct_from_selection_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);store=ReviewStore(root)
            store.save_frame(payload(0));store.save_frame(payload(1));store.save_frame(payload(2,ball_state='out_of_frame'))
            row=evaluate(root,'dev')['per_window'][0]
            self.assertEqual(row['raw_ball_candidate_recall'],.5)
            self.assertEqual(row['selected_ball']['tp'],0)
            self.assertEqual(row['selected_ball']['fn'],2)
            self.assertEqual(row['selected_ball']['fp'],2)
            self.assertEqual(row['raw_person']['tp'],1)
            self.assertEqual(row['raw_person']['fp'],1)

    def test_missing_ball_and_incomplete_identity_claim_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            store=ReviewStore(fixture(temp))
            item=payload();item['detections']=[]
            with self.assertRaises(ValueError):store.save_frame(item)
            item=payload();item['identities_complete']=True
            with self.assertRaises(ValueError):store.save_frame(item)

    def test_hidden_ball_coordinates_are_not_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            store=ReviewStore(fixture(temp))
            item=payload();item['ball_state']='fully_occluded'
            with self.assertRaises(ValueError):store.save_frame(item)

    def test_revision_history_preserves_previous_frame_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);store=ReviewStore(root)
            item=payload(source='ai_reviewed');store.save_frame(item)
            item['notes']='Second review';store.save_frame(item)
            history=[json.loads(line) for line in (root/'review_history.jsonl').read_text().splitlines()]
            self.assertEqual(len(history),2)
            self.assertEqual(history[1]['before']['frame_labels.csv'][0]['annotation_source'],'ai_reviewed')

    def test_evaluation_tolerance_cannot_change_silently(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);(root/'evaluation_config.json').write_text('{}')
            with self.assertRaises(ValueError):evaluate(root,'dev')

    def test_reference_export_requires_complete_identity_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);store=ReviewStore(root)
            self.assertEqual(export(root,root/'empty','dev')['exported'],[])
            for frame in range(3):
                item=payload(frame);item['identities_complete']=True
                for row in item['detections']:
                    if row['object_type']=='player':row['identity_id']='person_A'
                store.save_frame(item)
            report=export(root,root/'complete','dev')
            self.assertEqual(report['exported'],['w'])
            self.assertTrue((root/'complete/gt/w/gt/gt.txt').read_text().startswith('1,1,'))
            self.assertIn('seqLength', (root/'complete/gt/w/seqinfo.ini').read_text())
            self.assertIsNone(report['metrics'])

    def test_event_correction_and_random_window_completion_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);store=ReviewStore(root)
            event=dict(event_id='evt',window_id='e',frame=1,event_type='shot',decision='approved',
                       annotation_source='human',reviewer='synthetic_test',notes='test')
            store.save_event(event)
            self.assertIsNone(evaluate(root,'dev')['event_windows'][0]['metrics'])
            store.complete_window(dict(window_id='e',events_complete=True,annotation_source='human',
                                       reviewer='synthetic_test'))
            row=evaluate(root,'dev')['event_windows'][0]
            self.assertEqual(row['metrics']['typed_at_0.5s']['fn'],1)
            event.update(decision='uncertain');store.save_event(event)
            self.assertEqual(evaluate(root,'dev')['human_confirmed_event_counts'],{})

    def test_local_server_rejects_cross_origin_write_and_serves_ui(self):
        with tempfile.TemporaryDirectory() as temp:
            server=make_server(fixture(temp),0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/') as response:self.assertIn(b'<canvas',response.read())
                request=Request(base+'/api/save_frame',data=json.dumps(payload(source='ai_reviewed')).encode(),
                                headers={'Content-Type':'application/json','Origin':'https://external.invalid'})
                with self.assertRaises(HTTPError):urlopen(request)
                self.assertEqual(read_rows(Path(temp)/'frame_labels.csv'),[])
            finally:
                server.shutdown();server.server_close();thread.join()


class EventTests(unittest.TestCase):
    def test_fast_motion_emits_only_unverified_shot_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp)
            rows=read_rows(root/'run/ball_track.csv')
            rows[0].update(state='observed',confidence=.9,center_x=0,center_y=0,ball_id=1)
            rows[1].update(center_x=39,center_y=39)
            from replay_ball import TRACK_FIELDS
            write_rows(root/'run/ball_track.csv',TRACK_FIELDS,rows)
            report=generate(root,dict(motion_diagonals_per_s=.01))
            self.assertGreater(report['candidates'],0)
            candidates=read_rows(root/'event_candidates.csv')
            self.assertEqual(candidates[0]['event_type'],'shot_candidate')
            self.assertEqual(candidates[0]['status'],'human_review_required')
            self.assertEqual(read_rows(root/'event_labels.csv'),[])
    def test_single_frame_and_long_gap_do_not_establish_possession(self):
        tracker=Possession()
        ball=dict(x=10,y=20,confidence=.9)
        people=[dict(box=(0,0,20,20),track_id=1,confidence=.9)]
        self.assertEqual(tracker.update(0,ball,people)['state'],'unknown')
        self.assertEqual(tracker.update(.5,ball,people)['state'],'unknown')
        tracker.update(.6,ball,people)
        self.assertEqual(tracker.update(.7,ball,people)['owner_id'],1)
        self.assertEqual(tracker.update(.73,None,people)['state'],'unknown')
        self.assertEqual(tracker.update(.9,None,people)['owner_id'],'')

    def test_rivals_and_unassigned_players_are_not_forced_owners(self):
        ball=dict(x=10,y=20,confidence=.9)
        people=[dict(box=(0,0,20,20),track_id=1,confidence=.9),
                dict(box=(1,0,21,20),track_id=2,confidence=.9)]
        tracker=Possession()
        self.assertEqual(tracker.update(0,ball,people)['state'],'contested')
        self.assertEqual(Possession().update(0,ball,[dict(people[0],track_id=-1)])['state'],'unknown')

    def test_matching_is_one_to_one_and_maximum_cardinality(self):
        left,right=[0.,.6],[.4,1.]
        pairs=match_pairs(left,right,lambda a,b:abs(a-b)<=.6,lambda a,b:abs(a-b))
        self.assertEqual(len(pairs),2)
        self.assertEqual(len({j for _,j in pairs}),2)

    def test_motion_candidates_do_not_cross_scene_cut(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp)
            # The only motion would cross a cut; no shot should be invented.
            rows=read_rows(root/'run/ball_track.csv')
            rows[0].update(state='observed',confidence=.9,center_x=0,center_y=0,ball_id=1)
            rows[1].update(scene_id=1,state='observed',center_x=39,center_y=39)
            rows[2].update(scene_id=1,center_x=39,center_y=39)
            from replay_ball import TRACK_FIELDS
            write_rows(root/'run/ball_track.csv',TRACK_FIELDS,rows)
            data=json.loads((root/'dataset.json').read_text())
            data['sources'][0]['scenes']=[dict(scene_id=0,start_frame=0,end_frame=1),
                                         dict(scene_id=1,start_frame=1,end_frame=3)]
            for window in data['windows']:window['end_frame']=1
            data['windows'].append(dict(data['windows'][1],window_id='e1',scene_id=1,
                                        start_frame=1,end_frame=3,split_group='s:1'))
            (root/'dataset.json').write_text(json.dumps(data))
            report=generate(root,dict(motion_diagonals_per_s=.01))
            self.assertEqual(report['candidates'],0)

    def test_overlap_audit_does_not_suppress_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);before=(root/'run/detections.csv').read_bytes()
            result=audit(root/'run/detections.csv',root/'audit')
            self.assertEqual(result['weak_rows_with_overlap'],1)
            self.assertEqual((root/'run/detections.csv').read_bytes(),before)

    def test_overlap_audit_uses_only_complete_human_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=fixture(temp);store=ReviewStore(root)
            store.save_frame(payload(source='ai_reviewed'))
            result=audit(root/'run/detections.csv',root/'audit_ai',dataset=root)
            self.assertEqual(result['review_interpretations'],{'unreviewed_overlap_not_proven_duplicate':1})
            store.save_frame(payload(source='human'))
            result=audit(root/'run/detections.csv',root/'audit_human',dataset=root)
            self.assertEqual(result['review_interpretations'],{'both_boxes_match_same_human_box':1})
            empty=payload(ball_state='out_of_frame');empty['detections']=[]
            store.save_frame(empty)
            result=audit(root/'run/detections.csv',root/'audit_empty',dataset=root)
            self.assertEqual(result['review_interpretations'],{'reviewed_but_unmatched':1})


if __name__ == '__main__':
    unittest.main()
