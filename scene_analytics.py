"""Integrated image-space analytics prototype with explicit unvalidated event provenance."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from auto_pitch import infer_geometry,draw_geometry
from player_ball_filter import read
from event_candidates import Possession
from pitch_context import PitchBackground
from motion_consistency import MotionConsistentBall
from motion_blur_candidates import MotionBlurCandidates


def inside_person(x,y,people):
    return any(float(p['x1'])<=x<=float(p['x2']) and float(p['y1'])<=y<=float(p['y2']) for p in people)


def crossing(a,b,c,d):
    def cross(v,w):return v[0]*w[1]-v[1]*w[0]
    a,b,c,d=[np.asarray(p,float) for p in (a,b,c,d)]
    r,s=b-a,d-c;den=cross(r,s)
    if abs(den)<1e-8:return False
    t,u=cross(c-a,s)/den,cross(c-a,r)/den
    return 0<t<=1 and 0<=u<=1


def shot_exit(history, fps, origin, line):
    """Unverified shot evidence requires player departure, fast motion and an observed crossing."""
    if len(history)<3 or origin is None:return False
    a,b=history[-2:]
    dt=b['frame']-a['frame']
    if dt<=0 or not 0<=b['frame']-origin[1]<=2*fps:return False
    speed=np.hypot(b['x']-a['x'],b['y']-a['y'])*fps/dt
    return bool(speed>100 and crossing((a['x'],a['y']),(b['x'],b['y']),*line))


def analyze(scene, output_root="results/full_analysis_v052", use_motion_blur=False, reference_threshold=8.):
    if not np.isfinite(reference_threshold) or reference_threshold<=0:raise ValueError("Reference threshold must be finite and positive")
    scene=Path(scene);name=scene.stem;root=Path(output_root)/name;root.mkdir(parents=True,exist_ok=True)
    base=Path('results/measured')/name
    background=cv2.imread(str(base/'pitch_context/scene_000_background.png'))
    background_evidence=PitchBackground(background)
    reference_thumbnail=cv2.resize(background,(80,45)).astype(float)
    unevaluable_frames=[]
    geometry=infer_geometry(background)
    overrides=json.loads(Path('configs/pilot_goal_regions.json').read_text())['scenes'].get(name)
    geometry['automatic_goal_candidates']=list(geometry['goal_candidates'])
    geometry['automatic_endline_candidates']=list(geometry['endline_candidates'])
    if overrides and overrides['video_sha256']==hashlib.sha256(scene.read_bytes()).hexdigest():
        geometry['unresolved_partial_goal_regions']=[g for g in overrides['goals'] if g.get('partial')]
        geometry['goal_candidates']=[g for g in overrides['goals'] if not g.get('partial')]
        geometry['endline_candidates']=overrides['endlines']
        geometry['goal_geometry_source']='ai_reviewed_pilot_override'
    else:geometry['goal_geometry_source']='automatic_proposal'
    (root/'geometry.json').write_text(json.dumps(geometry,indent=2))
    cv2.imwrite(str(root/'geometry.png'),draw_geometry(background.copy(),geometry))
    people=defaultdict(list);candidates=defaultdict(list)
    for r in read(base/'detections.csv'):
        if r['class_name']=='person':people[int(r['frame'])].append(r)
    for r in read(Path('results/white_context')/name/'candidates.csv'):
        candidates[int(r['frame'])].append(dict(x=float(r['center_x']),y=float(r['center_y']),support=int(r['support']),moving=int(r['moving']),source='white_motion',weight=.15,box=[float(r[k]) for k in ('x1','y1','x2','y2')]))
    for r in read(base/'raw_detections.csv'):
        if r['class_name']=='sports ball' and float(r['confidence'])>=.15:
            candidates[int(r['frame'])].append(dict(x=(float(r['x1'])+float(r['x2']))/2,y=(float(r['y1'])+float(r['y2']))/2,support=3,moving=1,source='yolo',weight=float(r['confidence']),box=[float(r[k]) for k in ('x1','y1','x2','y2')]))
    cap=cv2.VideoCapture(str(scene));fps=cap.get(cv2.CAP_PROP_FPS);w,h=int(cap.get(3)),int(cap.get(4))
    writer=cv2.VideoWriter(str(root/'analysis.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise ValueError('Video writer unavailable')
    trajectory=MotionConsistentBall();origins={}
    blur_proposals=MotionBlurCandidates(background)
    previous=None;observations=[];events=[];last_event={};frame=0;counts=Counter();control=Possession();ownership=Counter();last_owner=None;last_owner_frame=None
    heat=np.zeros((h,w),np.float32);track_points=defaultdict(list);audit=[]
    def event(kind,reason,actor=None,recipient=None):
        if frame-last_event.get(kind,-100000)<fps:return
        events.append(dict(frame=frame,time_s=round(frame/fps,3),type=kind,reason=reason,actor=actor,recipient=recipient,status='unreviewed_candidate',source='automatic_proposal'))
        last_event[kind]=frame
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            mismatch=float(np.abs(cv2.resize(image,(80,45)).astype(float)-reference_thumbnail).mean())
            if mismatch>reference_threshold:
                counts['reference_mismatch_frames']+=1;unevaluable_frames.append(frame)
                trajectory=MotionConsistentBall();origins.clear();control=Possession();previous=None
                last_owner=None;last_owner_frame=None
                ownership['unknown']+=1
                cv2.putText(image,'TRANSITION / REFERENCE MISMATCH - UNKNOWN',(8,18),0,.4,(0,200,255),1)
                writer.write(image);frame+=1;continue
            ps=people[frame];eligible=[]
            for p in ps:
                x,y=float(p['foot_x']),float(p['foot_y'])
                if 0<=int(x)<w and 0<=int(y)<h:heat[int(y),int(x)]+=1
                if int(p['track_id'])>=0:track_points[(p['scene_id'],p['track_id'])].append((frame,x,y))
            additions=blur_proposals.update(image) if use_motion_blur else []
            counts['motion_blur_candidates']+=len(additions)
            for c in candidates[frame]+additions:
                counts['raw_candidates']+=1
                if inside_person(c['x'],c['y'],ps):
                    counts['person_suppressed']+=1;audit.append(dict(frame=frame,**c,decision='person_box'));continue
                if background_evidence.evidence(image,c['box'])['static_marking']:
                    counts['static_suppressed']+=1;continue
                point=(c['x'],c['y']);polygon=np.array(geometry['pitch_polygon'],np.float32)
                if len(polygon) and cv2.pointPolygonTest(polygon,point,True)<-20:
                    counts['outside_pitch']+=1;continue
                # Perspective-aware upper size bound from nearby player heights.
                nearby=[]
                for p in ps:
                    ph=float(p['y2'])-float(p['y1'])
                    px=(float(p['x1'])+float(p['x2']))/2;py=float(p['y2'])
                    distance=np.hypot(c['x']-px,c['y']-py)
                    if ph>0 and distance<3*ph:nearby.append((distance,ph))
                limit=min(24.,max(4.,.4*float(np.median([h for _,h in sorted(nearby)[:3]])))) if nearby else 24.
                if max(c['box'][2]-c['box'][0],c['box'][3]-c['box'][1])>limit:
                    counts['perspective_size_rejected']+=1;continue
                eligible.append((c['weight'],c))
            ball=trajectory.update(frame,[c for _,c in eligible])
            counts.update(trajectory.diagnostics)
            if ball:
                history=ball['history'];tid=ball['trajectory_id']
                if tid not in origins:
                    nearest=[]
                    for point in history[:-1]:
                        for p in people[point['frame']]:
                            x1,y1,x2,y2=[float(p[k]) for k in ('x1','y1','x2','y2')]
                            distance=np.hypot(max(x1-point['x'],0,point['x']-x2),max(y1-point['y'],0,point['y']-y2))/max(1,y2-y1)
                            if distance<=.5:nearest.append((distance,p,point['frame']))
                    if nearest:
                        _,p,origin_frame=min(nearest,key=lambda x:x[0]);origins[tid]=(p['track_id'],origin_frame)
                observations.append(dict(frame=frame,time_s=round(frame/fps,4),x=ball['x'],y=ball['y'],source=ball['source'],status='motion_supported_unverified',trajectory_id=tid,step_px=ball['step_px'],size_px=ball['size'],straightness=ball['straightness']))
                a=(history[-2]['x'],history[-2]['y']);b=(ball['x'],ball['y']);speed=ball['step_px']*fps
                origin=origins.get(tid)
                for line in geometry['endline_candidates']:
                    if crossing(a,b,*line):
                        event('endline_crossing','Motion-consistent observed trajectory crosses approximate end line')
                        if shot_exit(history,fps,origin,line):
                            event('shot_candidate','Player departure + sustained fast motion + observed end-line crossing',origin[0])
                for g in geometry['goal_candidates']:
                    poly=np.array(g['polygon'],np.float32)
                    if cv2.pointPolygonTest(poly,a,False)<0 and cv2.pointPolygonTest(poly,b,False)>=0:
                        event('goal_entry_candidate','Observed moving trajectory enters complete visible goal-region proposal; not confirmed goal')
                previous=ball
            pinput=[dict(track_id=int(p['track_id']),confidence=float(p['confidence']),box=[float(p[k]) for k in ('x1','y1','x2','y2')]) for p in ps]
            state=control.update(frame/fps,dict(x=ball['x'],y=ball['y'],confidence=.15) if ball else None,pinput)
            ownership[state['state']]+=1
            if state['state']=='player':
                owner=state['owner_id']
                if last_owner is not None and owner!=last_owner and frame-last_owner_frame<2*fps:event('transfer_candidate','Control hypothesis changes; teams and pass intent unknown',last_owner,owner)
                last_owner,last_owner_frame=owner,frame
            draw_geometry(image,geometry)
            for p in ps:
                a,b=(int(float(p['x1'])),int(float(p['y1']))),(int(float(p['x2'])),int(float(p['y2'])))
                cv2.rectangle(image,a,b,(80,180,80),1)
            if ball:
                for a,b in zip(ball['history'],ball['history'][1:]):
                    if b['frame']-a['frame']==1:cv2.line(image,(int(a['x']),int(a['y'])),(int(b['x']),int(b['y'])),(255,255,0),1)
            if ball:cv2.circle(image,(int(ball['x']),int(ball['y'])),6,(255,255,0),2)
            cv2.putText(image,'MOTION-SUPPORTED BALL / UNVERIFIED EVENTS',(8,18),0,.4,(0,200,255),1)
            writer.write(image);frame+=1
    finally:cap.release();writer.release()
    geometry['unevaluable_frames']=unevaluable_frames
    (root/'geometry.json').write_text(json.dumps(geometry,indent=2))
    movement=[]
    for (sid,tid),points in track_points.items():
        steps=[np.hypot(b[1]-a[1],b[2]-a[2]) for a,b in zip(points,points[1:]) if b[0]-a[0]==1]
        movement.append(dict(track_id=tid,scene_id=sid,observation_s=len(points)/fps,observed_steps=len(steps),movement_px=round(sum(steps),1)))
    heat=cv2.GaussianBlur(heat,(0,0),12)
    hm=cv2.addWeighted(background,.55,cv2.applyColorMap((255*heat/max(1,float(heat.max()))).astype('uint8'),cv2.COLORMAP_TURBO),.45,0)
    cv2.imwrite(str(root/'heatmap.png'),hm)
    summary=dict(scene=name,reference_difference_threshold=reference_threshold,frames=frame,fps=fps,duration_s=frame/fps,**counts,selected_ball_frames=len(observations),ball_coverage_percent=100*len(observations)/max(1,frame),ownership_seconds={k:v/fps for k,v in ownership.items()},event_candidates=dict(Counter(e['type'] for e in events)),confirmed_events=None,running_meters=None,rating=None,goal_geometry_source=geometry['goal_geometry_source'],unresolved_partial_goals=len(geometry.get('unresolved_partial_goal_regions',[])),goal_candidates=len(geometry['goal_candidates']),endline_candidates=len(geometry['endline_candidates']),marking_segments=len(geometry['marking_segments']),pitch_detected=bool(geometry['pitch_polygon']),human_labels=0)
    for name,data in [('summary',summary),('events',events),('ball_observations',observations),('movement',movement),('suppression_audit',audit)]:
        (root/(name+'.json')).write_text(json.dumps(data,indent=2))
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True,help='Fresh output directory for this comparison')
    parser.add_argument('--scene',help='Optional single scene clip; default runs cached pilot clips')
    parser.add_argument('--reference-threshold',type=float,default=8.,help='Mean thumbnail channel difference above which geometry is unevaluable')
    parser.add_argument('--motion-blur',action='store_true',help='Experimental, not enabled in delivered report')
    args=parser.parse_args()
    if Path(args.output_root).exists():raise ValueError('Output directory already exists; use a fresh path')
    scenes=[Path(args.scene)] if args.scene else sorted(Path('results/scene_clips').glob('scene_00[0-8].mp4'))
    for scene in scenes:print(json.dumps(analyze(scene,args.output_root,args.motion_blur,args.reference_threshold)),flush=True)
