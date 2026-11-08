"""Suppress persistent torso-overlap proposals using cached player boxes, not segmentation."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import cv2


class TorsoFilter:
    def __init__(self, required_frames=3, torso_fraction=.8, relative_tolerance=.15):
        self.required_frames=required_frames
        self.torso_fraction=torso_fraction
        self.relative_tolerance=relative_tolerance
        self.history={}

    def classify(self, candidate, people):
        frame=int(candidate['frame']);key=int(candidate['proposal_id'])
        x,y=float(candidate['center_x']),float(candidate['center_y'])
        self.history={k:v for k,v in self.history.items() if frame-v['frame']<=1}
        matches=[]
        for p in people:
            x1,y1,x2,y2=[float(p[k]) for k in ('x1','y1','x2','y2')]
            if int(p['track_id'])<0 or float(p['confidence'])<.25:continue
            if x1<=x<=x2 and y1<=y<=y1+self.torso_fraction*(y2-y1):
                matches.append(((x2-x1)*(y2-y1),p,((x-x1)/max(1,x2-x1),(y-y1)/max(1,y2-y1))))
        if not matches:
            self.history.pop(key,None)
            return 'retained','no_persistent_torso_overlap'
        _,p,relative=min(matches,key=lambda item:item[0]);previous=self.history.get(key)
        count=1
        if previous and previous['frame']==frame-1 and previous['person']==p['track_id'] and max(abs(a-b) for a,b in zip(previous['relative'],relative))<=self.relative_tolerance:
            count=previous['count']+1
        self.history[key]=dict(frame=frame,person=p['track_id'],relative=relative,count=count)
        return ('suppressed','persistent_torso_overlap') if count>=self.required_frames else ('retained','torso_overlap_pending')


def read(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def process(scene, output_root='results/torso_filter'):
    name=Path(scene).stem
    proposals=read(Path('results/white_context')/name/'candidates.csv')
    people=defaultdict(list)
    for p in read(Path('results/measured')/name/'detections.csv'):
        if p['class_name']=='person':people[int(p['frame'])].append(p)
    filt=TorsoFilter();by_frame=defaultdict(list)
    for r in proposals:
        r['decision'],r['reason']=filt.classify(r,people[int(r['frame'])]);by_frame[int(r['frame'])].append(r)
    out=Path(output_root)/name;out.mkdir(parents=True,exist_ok=False)
    fields=list(proposals[0]) if proposals else []
    with (out/'candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(proposals)
    cap=cv2.VideoCapture(str(scene));fps=cap.get(cv2.CAP_PROP_FPS);w,h=int(cap.get(3)),int(cap.get(4));frame=0
    writer=cv2.VideoWriter(str(out/'review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise ValueError('Cannot open video writer')
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            for p in people[frame]:
                x1,y1,x2,y2=[int(float(p[k])) for k in ('x1','y1','x2','y2')]
                cv2.rectangle(image,(x1,y1),(x2,y2),(100,180,80),1)
            for r in by_frame[frame]:
                if r['decision']=='suppressed':continue
                cv2.rectangle(image,(int(r['x1'])-2,int(r['y1'])-2),(int(r['x2'])+2,int(r['y2'])+2),(0,200,255),1)
            cv2.putText(image,'GREEN: PEOPLE / YELLOW: UNVERIFIED BALL CANDIDATES',(8,18),0,.35,(0,200,255),1)
            writer.write(image);frame+=1
    finally:cap.release();writer.release()
    summary=dict(frames=frame,proposals=len(proposals),suppressed=sum(r['decision']=='suppressed' for r in proposals),retained=sum(r['decision']=='retained' for r in proposals),human_labels=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output-root',default='results/torso_filter');a=p.parse_args();print(process(a.scene,a.output_root))
