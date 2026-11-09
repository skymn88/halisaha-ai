import copy
import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
from body_ball_evidence import appearance,body_attachment
from reviewed_geometry import validate_geometry,resolve_geometry
from tiered_events import directed_to_mouth,transfer_kind,generate_events
from evaluate_clip_proxy import score
from review_pipeline_v06 import load_config

C=load_config('configs/review_v06.json')

class ReviewV06Tests(unittest.TestCase):
    def test_human_geometry_requires_hash_source_and_name(self):
        r=dict(schema_version=2,scene='s',video_sha256='abc',image_size=[640,360],annotation_source='human',review_status='reviewed',reviewer='tester',goals=[dict(polygon=[[10,50],[10,10],[50,10],[50,50]],mouth_line=[[10,50],[50,50]])],endlines=[])
        self.assertIs(validate_geometry(r,'s','abc'),r)
        for key,value in [('video_sha256','wrong'),('annotation_source','ai_reviewed'),('reviewer',''),('review_status','unreviewed')]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_geometry(dict(r,**{key:value}),'s','abc')
        r['goals'][0]['polygon'][1]=[float('nan'),20]
        with self.assertRaises(ValueError):validate_geometry(r,'s','abc')

    def test_skin_is_penalty_not_rejection_and_feet_retained(self):
        image=np.full((120,120,3),[80,130,180],np.uint8)
        person=dict(x1=10,y1=10,x2=60,y2=110,track_id=1)
        c=dict(x=30,y=100,box=[27,97,33,103],weight=.3)
        result=appearance(c,image,[person],C)
        self.assertTrue(result['inside_person']);self.assertTrue(result['foot_zone'])
        self.assertGreater(result['weight'],0)
        self.assertGreater(result['skin_fraction'],.45)

    def points(self,independent=False):
        return [dict(frame=i,x=10+i*(6 if independent else 2),y=30,person_center=[10+i*2,40],person_height=100,person_id=1,foot_distance=.3) for i in range(4)]

    def test_body_motion_veto_and_independent_release(self):
        self.assertTrue(body_attachment(self.points(),C))
        self.assertFalse(body_attachment(self.points(True),C))

    def test_stationary_person_is_not_attachment_evidence(self):
        ps=self.points()
        for p in ps:p['person_center']=[10,40]
        self.assertFalse(body_attachment(ps,C))

    def test_identity_switch_does_not_prove_attachment(self):
        ps=self.points();ps[-1]['person_id']=2
        self.assertFalse(body_attachment(ps,C))

    def test_goal_direction_forward_only(self):
        self.assertTrue(directed_to_mouth(dict(x=0,y=5),dict(x=5,y=5),[[10,0],[10,10]],0))
        self.assertFalse(directed_to_mouth(dict(x=0,y=5),dict(x=-5,y=5),[[10,0],[10,10]],0))

    def test_unknown_teams_never_completed_pass(self):
        self.assertEqual(transfer_kind(1,2,{}),'control_change_unknown_team')
        self.assertEqual(transfer_kind(1,2,{'1':'a','2':'a'}),'completed_pass_candidate')
        self.assertEqual(transfer_kind(1,2,{'1':'a','2':'b'}),'ball_recovery_candidate')

    def test_proxy_does_not_call_unmatched_false_positive(self):
        s=score([dict(type='shot_candidate',time_s=4),dict(type='fast_release_review',time_s=4)],4,60)
        self.assertTrue(s['ai_window_hit']);self.assertIsNone(s['false_candidates_per_minute']);self.assertIsNone(s['human_recall'])
        self.assertEqual(s['unmatched_review_candidates_per_minute'],1)
        self.assertFalse(score([dict(type='fast_release_review',time_s=4)],4,60)['ai_window_hit'])

    def synthetic(self,gap=False):
        # Sustained control, observed release, then line crossing directed into the goal.
        positions=[0]*8+[10,20,30,40,50,60,70,80,90,100]
        frames=list(range(len(positions)))
        if gap:frames=[f+(2 if f>=9 else 0) for f in frames]
        ps={f:[dict(track_id='1',confidence='1',x1='-5',y1='-10',x2='5',y2='10')] for f in frames}
        obs=[dict(trajectory_id='t',frame=f,x=x,y=10) for f,x in zip(frames,positions)]
        geom=dict(geometry_source='human_reviewed',endlines=[[[85,0],[85,20]]],goals=[dict(polygon=[[85,0],[100,0],[100,20],[85,20]],mouth_line=[[85,0],[85,20]])])
        return generate_events(obs,ps,geom,30,C)

    def test_shot_strong_requires_observed_flight(self):
        shots=[e for e in self.synthetic() if e['type']=='shot_candidate']
        self.assertTrue(shots);self.assertEqual(shots[0]['tier'],'A')
        shots=[e for e in self.synthetic(True) if e['type']=='shot_candidate']
        self.assertTrue(shots);self.assertEqual(shots[0]['tier'],'B')

    def test_no_release_is_not_a_shot(self):
        obs=[dict(trajectory_id='t',frame=i,x=70+10*i,y=10) for i in range(4)]
        geom=dict(geometry_source='human_reviewed',endlines=[[[85,0],[85,20]]],goals=[])
        self.assertFalse(any(e['type']=='shot_candidate' for e in generate_events(obs,{},geom,30,C)))

    def test_separate_tracks_never_form_release_chain(self):
        obs=[dict(trajectory_id=str(i),frame=i,x=0 if i<8 else 100,y=10) for i in range(12)]
        ps={i:[dict(track_id='1',confidence='1',x1='-5',y1='-10',x2='5',y2='10')] for i in range(12)}
        self.assertEqual(generate_events(obs,ps,dict(geometry_source='unavailable',goals=[],endlines=[]),30,C),[])

    def test_recontrol_without_goal_evidence_is_not_shot(self):
        xs=[0]*8+[10,20,30,40,50,60,70,80,90,100]+[100]*8
        frames=[i+(2 if i>=9 else 0) for i in range(len(xs))]
        people={f:[dict(track_id=str(j),confidence='1',x1=str(x-5),y1='-10',x2=str(x+5),y2='10') for j,x in [(1,0),(2,100)]] for f in frames}
        obs=[dict(trajectory_id='t',frame=f,x=x,y=10) for f,x in zip(frames,xs)]
        geometry=dict(geometry_source='ai_reviewed_exploratory',endlines=[],goals=[dict(polygon=[[120,0],[140,0],[140,20],[120,20]],mouth_line=[[120,0],[120,20]])])
        events=generate_events(obs,people,geometry,30,C)
        self.assertTrue(any(e['type']=='release_recontrol_review' for e in events))
        self.assertFalse(any(e['type']=='shot_candidate' for e in events))

    def test_default_geometry_never_silently_uses_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'scene_000.mp4';video.write_bytes(b'synthetic')
            geometry=resolve_geometry(video)
            self.assertEqual(geometry['geometry_source'],'unavailable')
            self.assertEqual(geometry['goals'],[])

if __name__=='__main__':unittest.main()
