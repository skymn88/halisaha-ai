"""Opt-in small bright moving-region proposals; scores are not ball probabilities."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import cv2
import numpy as np


class WhiteMotionCandidates:
    def __init__(self, min_value=150, max_saturation=100, max_area=90, min_change=18, grass_context=False):
        self.min_value, self.max_saturation = min_value, max_saturation
        self.max_area, self.min_change = max_area, min_change
        self.grass_context = grass_context
        self.previous = None
        self.tracks = []
        self.next_id = 1

    def update(self, image, background, frame):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        local = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((11,11),np.uint8))
        changed = cv2.absdiff(gray, background) >= self.min_change
        white = (hsv[:,:,2]>=self.min_value)&(hsv[:,:,1]<=self.max_saturation)
        mask = (white & changed & (local>=12)).astype('uint8')
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask,8)
        difference = cv2.absdiff(gray, self.previous) if self.previous is not None else None
        result=[]
        used=set()
        self.tracks=[t for t in self.tracks if frame-t['frame']<=3]
        for i in range(1,count):
            x,y,w,h,area=map(int,stats[i])
            if not 2<=area<=self.max_area or max(w,h)>20 or max(w,h)/max(1,min(w,h))>4: continue
            if self.grass_context:
                # Context is an experimental cue, not a pitch polygon or unconditional ball rule.
                patch=hsv[max(0,y-8):min(hsv.shape[0],y+h+8),max(0,x-8):min(hsv.shape[1],x+w+8)]
                green=(patch[:,:,0]>=30)&(patch[:,:,0]<=95)&(patch[:,:,1]>=45)&(patch[:,:,2]>=25)
                if float(green.mean())<.35:continue
            cx,cy=map(float,centers[i])
            moving = difference is not None and float(np.mean(difference[y:y+h,x:x+w][labels[y:y+h,x:x+w]==i]))>=self.min_change
            matches=[]
            for t in self.tracks:
                if t['id'] in used:continue
                dt=frame-t['frame'];pred=np.array(t['center'])+np.array(t['velocity'])*dt
                distance=float(np.linalg.norm(np.array([cx,cy])-pred))
                if distance<=12*dt:matches.append((distance,t))
            if matches:
                _,t=min(matches,key=lambda z:z[0]);dt=frame-t['frame']
                velocity=((cx-t['center'][0])/dt,(cy-t['center'][1])/dt)
                t.update(center=(cx,cy),velocity=velocity,frame=frame,hits=t['hits']+1)
            elif moving:
                t=dict(id=self.next_id,center=(cx,cy),velocity=(0,0),frame=frame,hits=1)
                self.next_id+=1;self.tracks.append(t)
            else: continue
            used.add(t['id'])
            result.append(dict(frame=frame,proposal_id=t['id'],x1=x,y1=y,x2=x+w,y2=y+h,center_x=cx,center_y=cy,area=area,moving=int(moving),support=t['hits'],status='temporal_candidate' if t['hits']>=3 else 'new_candidate',source='white_motion',probability=''))
        self.previous=gray
        return result


def run(video, output, grass_context=False):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(str(video));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    samples=[]
    for index in np.linspace(0,n-1,min(41,n)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,im=cap.read()
        if ok:samples.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
    if not samples:raise ValueError('No decoded input frames')
    background=np.median(samples,axis=0).astype('uint8')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    h,w=background.shape
    writer=cv2.VideoWriter(str(output/'proposals.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise ValueError('Cannot write preview')
    detector=WhiteMotionCandidates(grass_context=grass_context);all_rows=[];frame=0
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            proposals=detector.update(im,background,frame);all_rows.extend(proposals)
            for r in proposals:
                color=(0,200,255) if r['support']>=3 else (180,180,180)
                cv2.rectangle(im,(r['x1']-2,r['y1']-2),(r['x2']+2,r['y2']+2),color,1)
            cv2.putText(im,'WHITE MOTION PROPOSALS - NOT VERIFIED BALLS',(8,18),0,.4,(0,200,255),1)
            writer.write(im);frame+=1
    finally:cap.release();writer.release()
    fields=['frame','proposal_id','x1','y1','x2','y2','center_x','center_y','area','moving','support','status','source','probability']
    with (output/'candidates.csv').open('w',newline='') as f:
        cw=csv.DictWriter(f,fields);cw.writeheader();cw.writerows(all_rows)
    summary=dict(grass_context=grass_context,frames=frame,candidates=len(all_rows),candidate_frames=len({r['frame'] for r in all_rows}),temporal_candidate_frames=len({r['frame'] for r in all_rows if r['support']>=3}),human_labels=0,note='Proposal coverage is not ball recall. Offline per-clip median uses future frames. Run on cut-separated clips only.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--output',required=True)
    p.add_argument('--grass-context',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.video,a.output,a.grass_context)))
