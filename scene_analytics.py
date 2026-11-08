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


def inside_person(x,y,people):
    return any(float(p['x1'])<=x<=float(p['x2']) and float(p['y1'])<=y<=float(p['y2']) for p in people)


def crossing(a,b,c,d):
    def cross(v,w):return v[0]*w[1]-v[1]*w[0]
    a,b,c,d=[np.asarray(p,float) for p in (a,b,c,d)]
    r,s=b-a,d-c;den=cross(r,s)
    if abs(den)<1e-8:return False
    t,u=cross(c-a,s)/den,cross(c-a,r)/den
    return 0<t<=1 and 0<=u<=1


def analyze(scene):
    scene=Path(scene);name=scene.stem;root=Path('results/full_analysis')/name;root.mkdir(parents=True,exist_ok=True)
    base=Path('results/measured')/name
    background=cv2.imread(str(base/'pitch_context/scene_000_background.png'))
    background_evidence=PitchBackground(background)
    geometry=infer_geometry(background)
    overrides=json.loads(Path('configs/pilot_goal_regions.json').read_text())['scenes'].get(name)
    geometry['automatic_goal_candidates']=list(geometry['goal_candidates'])
    geometry['automatic_endline_candidates']=list(geometry['endline_candidates'])
    if overrides and overrides['video_sha256']==hashlib.sha256(scene.read_bytes()).hexdigest():
        geometry['goal_candidates']=overrides['goals'];geometry['endline_candidates']=overrides['endlines']
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
            ps=people[frame];eligible=[]
            for p in ps:
                x,y=float(p['foot_x']),float(p['foot_y'])
                if 0<=int(x)<w and 0<=int(y)<h:heat[int(y),int(x)]+=1
                if int(p['track_id'])>=0:track_points[(p['scene_id'],p['track_id'])].append((frame,x,y))
            for c in candidates[frame]:
                counts['raw_candidates']+=1
                if inside_person(c['x'],c['y'],ps):
                    counts['person_suppressed']+=1;audit.append(dict(frame=frame,**c,decision='person_box'));continue
                if c['support']<3:continue
                if background_evidence.evidence(image,c['box'])['static_marking']:
                    counts['static_suppressed']+=1;continue
                point=(c['x'],c['y']);polygon=np.array(geometry['pitch_polygon'],np.float32)
                if len(polygon) and cv2.pointPolygonTest(polygon,point,True)<-20:
                    counts['outside_pitch']+=1;continue
                if previous and frame-previous['frame']<=3:
                    distance=np.hypot(c['x']-previous['x'],c['y']-previous['y'])
                    if distance>35*(frame-previous['frame']):continue
                    score=c['weight']+.35-distance/100
                else:
                    if not c['moving']:continue
                    score=c['weight']
                eligible.append((score,c))
            eligible.sort(key=lambda z:-z[0]);ball=None
            # Ambiguous white components do not become an arbitrary single ball.
            if eligible and (len(eligible)==1 or eligible[0][0]-eligible[1][0]>.10):
                ball=dict(eligible[0][1],frame=frame)
                observations.append(dict(frame=frame,time_s=round(frame/fps,4),x=ball['x'],y=ball['y'],source=ball['source'],status='unverified_selected'))
                if previous and frame-previous['frame']<=3:
                    a=(previous['x'],previous['y']);b=(ball['x'],ball['y']);speed=np.linalg.norm(np.array(b)-a)*fps/(frame-previous['frame'])
                    for line in geometry['endline_candidates']:
                        if crossing(a,b,*line):event('endline_crossing','2D crossing of approximate end-line region; not a shot classification')
                    for g in geometry['goal_candidates']:
                        poly=np.array(g['polygon'],np.float32);center=poly.mean(axis=0)
                        before=np.linalg.norm(np.array(a)-center);after=np.linalg.norm(np.array(b)-center)
                        if speed>100 and after<80 and after<before-2:event('shot_candidate','Fast motion towards approximate goal region; unverified shot hypothesis')
                        if cv2.pointPolygonTest(poly,a,False)<0 and cv2.pointPolygonTest(poly,b,False)>=0:event('goal_entry_candidate','2D goal-region entry; not proof of goal-line crossing')
                previous=ball
            if previous and frame-previous['frame']>3:previous=None
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
            for _,c in eligible[:30]:cv2.circle(image,(int(c['x']),int(c['y'])),3,(0,180,255),1)
            if ball:cv2.circle(image,(int(ball['x']),int(ball['y'])),6,(255,255,0),2)
            cv2.putText(image,'AUTO GEOMETRY / UNVERIFIED BALL AND EVENTS',(8,18),0,.4,(0,200,255),1)
            writer.write(image);frame+=1
    finally:cap.release();writer.release()
    movement=[]
    for (sid,tid),points in track_points.items():
        steps=[np.hypot(b[1]-a[1],b[2]-a[2]) for a,b in zip(points,points[1:]) if b[0]-a[0]==1]
        movement.append(dict(track_id=tid,scene_id=sid,observation_s=len(points)/fps,observed_steps=len(steps),movement_px=round(sum(steps),1)))
    heat=cv2.GaussianBlur(heat,(0,0),12)
    hm=cv2.addWeighted(background,.55,cv2.applyColorMap((255*heat/max(1,float(heat.max()))).astype('uint8'),cv2.COLORMAP_TURBO),.45,0)
    cv2.imwrite(str(root/'heatmap.png'),hm)
    summary=dict(scene=name,frames=frame,fps=fps,duration_s=frame/fps,**counts,selected_ball_frames=len(observations),ball_coverage_percent=100*len(observations)/max(1,frame),ownership_seconds={k:v/fps for k,v in ownership.items()},event_candidates=dict(Counter(e['type'] for e in events)),confirmed_events=None,running_meters=None,rating=None,goal_geometry_source=geometry['goal_geometry_source'],goal_candidates=len(geometry['goal_candidates']),endline_candidates=len(geometry['endline_candidates']),marking_segments=len(geometry['marking_segments']),pitch_detected=bool(geometry['pitch_polygon']),human_labels=0)
    for name,data in [('summary',summary),('events',events),('ball_observations',observations),('movement',movement),('suppression_audit',audit)]:
        (root/(name+'.json')).write_text(json.dumps(data,indent=2))
    return summary


if __name__=='__main__':
    for p in sorted(Path('results/scene_clips').glob('scene_00[0-8].mp4')):print(json.dumps(analyze(p)),flush=True)
