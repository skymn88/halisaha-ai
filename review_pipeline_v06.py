"""Cached multi-hypothesis review pipeline. No detector inference and no human truth fabrication."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
from player_ball_filter import read
from pitch_context import PitchBackground
from motion_consistency import MotionConsistentBall
from body_ball_evidence import appearance, body_attachment
from reviewed_geometry import resolve_geometry
from tiered_events import generate_events


def load_config(path):
    data=json.loads(Path(path).read_text())
    reference=json.loads(Path('configs/review_v06.json').read_text())
    if set(data)!=set(reference) or not isinstance(data['version'],str):raise ValueError('Invalid review config keys/version')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for k,v in data.items() if k!='version'):raise ValueError('Invalid review threshold')
    for k in ('direction_consistency','body_motion_correlation','foot_zone_fraction','skin_fraction_penalty'):
        if data[k]>1:raise ValueError('Fraction exceeds one')
    return data


def analyze(video, output, config, geometry_dir=None, allow_ai=False):
    name=video.stem;root=output/name;root.mkdir()
    base=Path('results/measured')/name
    background=cv2.imread(str(base/'pitch_context/scene_000_background.png'))
    if background is None:raise ValueError('Missing cached background')
    reference=cv2.resize(background,(80,45)).astype(float);evidence=PitchBackground(background)
    geometry=resolve_geometry(video,geometry_dir,allow_ai)
    people=defaultdict(list);raw=defaultdict(list)
    for r in read(base/'detections.csv'):
        if r['class_name']=='person':people[int(r['frame'])].append(r)
    for r in read(Path('results/white_context')/name/'candidates.csv'):
        raw[int(r['frame'])].append(dict(x=float(r['center_x']),y=float(r['center_y']),box=[float(r[k]) for k in ('x1','y1','x2','y2')],weight=.15,source='white_motion'))
    for r in read(base/'raw_detections.csv'):
        if r['class_name']=='sports ball' and float(r['confidence'])>=.15:
            raw[int(r['frame'])].append(dict(x=(float(r['x1'])+float(r['x2']))/2,y=(float(r['y1'])+float(r['y2']))/2,box=[float(r[k]) for k in ('x1','y1','x2','y2')],weight=float(r['confidence']),source='yolo'))
    cap=cv2.VideoCapture(str(video));fps=cap.get(5);width,height=int(cap.get(3)),int(cap.get(4))
    writer=cv2.VideoWriter(str(root/'analysis.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise ValueError('Video writer failed')
    def tracker():return MotionConsistentBall(max_gap=max(2,round(config['association_gap_s']*fps)))
    association=tracker();epoch=0;frame=0;counts=Counter();observations={};audit=[];excluded=[]
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            mismatch=float(np.abs(cv2.resize(image,(80,45)).astype(float)-reference).mean())
            if mismatch>config['reference_threshold']:
                excluded.append(frame);association=tracker();epoch+=1
                cv2.putText(image,'REFERENCE MISMATCH / UNKNOWN',(8,20),0,.45,(0,200,255),1)
                writer.write(image);frame+=1;continue
            eligible=[]
            for c in raw[frame]:
                c=appearance(c,image,people[frame],config);reason='eligible'
                if evidence.evidence(image,c['box'])['static_marking']:reason='static_reference'
                elif c['size_ratio']>config['max_size_heights']:reason='relative_size'
                counts[reason]+=1
                audit.append(dict(frame=frame,**c,decision=reason))
                if reason=='eligible':eligible.append(c)
            association.update(frame,eligible)
            for track in association.tracks:
                pts=track['points']
                if pts[-1]['frame']!=frame or len(pts)<3:continue
                tid=f'{epoch}:{track["id"]}'
                if body_attachment(pts,config):
                    counts['body_attached']+=1
                    for p in pts:observations.pop((tid,p['frame']),None)
                    continue
                height_scale=float(np.median([p['person_height'] for p in pts]))
                displacement=math.hypot(pts[-1]['x']-pts[0]['x'],pts[-1]['y']-pts[0]['y'])/height_scale
                if displacement<config['candidate_displacement_heights']:counts['static_track']+=1;continue
                # Retrospective evidence includes the original contact frames, never predicted points.
                for p in pts:
                    observations[(tid,p['frame'])]=dict(p,trajectory_id=tid,time_s=p['frame']/fps,status='review_hypothesis',evidence='observed',body_attachment=False)
                cv2.circle(image,(round(pts[-1]['x']),round(pts[-1]['y'])),5,(0,200,255),1)
            for p in people[frame]:cv2.rectangle(image,(int(float(p['x1'])),int(float(p['y1']))),(int(float(p['x2'])),int(float(p['y2']))),(60,160,60),1)
            for g in geometry['goals']:cv2.polylines(image,[np.array(g['polygon'],np.int32)],True,(255,120,20),1)
            for line in geometry['endlines']:cv2.line(image,tuple(map(int,line[0])),tuple(map(int,line[1])),(200,80,200),1)
            cv2.putText(image,'ALTERNATIVE BALL HYPOTHESES / HUMAN REVIEW REQUIRED',(8,18),0,.35,(0,200,255),1)
            writer.write(image);frame+=1
    finally:cap.release();writer.release()
    obs=sorted(observations.values(),key=lambda p:(p['frame'],p['trajectory_id']))
    events=generate_events(obs,people,geometry,fps,config)
    summary=dict(scene=name,frames=frame,fps=fps,duration_s=frame/fps,evaluable_duration_s=(frame-len(excluded))/fps,
                 config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                 video_sha256=geometry['video_sha256'],geometry_source=geometry['geometry_source'],
                 hypothesis_observations=len(obs),frames_with_hypotheses=len({p['frame'] for p in obs}),
                 human_labels=0,human_precision=None,human_recall=None,diagnostics=dict(counts),unevaluable_frames=excluded,
                 event_candidates=dict(Counter(e['type'] for e in events)),evidence_tiers=dict(Counter(e['tier'] for e in events)))
    for filename,data in [('summary',summary),('events',events),('ball_observations',obs),('candidate_audit',audit),('geometry',geometry)]:
        (root/f'{filename}.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(scene=name,events=summary['event_candidates'],hypotheses=len(obs))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--config',default='configs/review_v06.json')
    parser.add_argument('--geometry-dir');parser.add_argument('--allow-ai-geometry',action='store_true',help='Explicit exploratory use; not human geometry')
    args=parser.parse_args();config=load_config(args.config);out=Path(args.output);out.mkdir(exist_ok=False)
    for video in sorted(Path('results/scene_clips').glob('scene_???.mp4')):analyze(video,out,config,args.geometry_dir,args.allow_ai_geometry)
