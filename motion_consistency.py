"""Multi-observation ball association: static jitter and size/trajectory jumps are rejected."""
import math
import numpy as np


class MotionConsistentBall:
    def __init__(self, min_hits=3, max_gap=2, min_displacement=3., min_step=.7,
                 max_step=65., max_size_ratio=2.2, min_straightness=.65):
        values=[min_displacement,min_step,max_step,max_size_ratio,min_straightness]
        if not all(math.isfinite(v) for v in values) or min_hits<3 or max_gap<1 or min_displacement<=0 or min_step<0 or max_step<=min_step or max_size_ratio<1 or not 0<min_straightness<=1:
            raise ValueError('Invalid motion-consistency thresholds')
        self.min_hits=min_hits;self.max_gap=max_gap;self.min_displacement=min_displacement
        self.min_step=min_step;self.max_step=max_step;self.max_size_ratio=max_size_ratio
        self.min_straightness=min_straightness;self.tracks=[];self.next_id=1
        self.last_frame=-1;self.diagnostics=[];self.active_id=None;self.last_selected_frame=-100

    def update(self, frame, candidates):
        if frame<=self.last_frame:raise ValueError('Frames must increase')
        self.last_frame=frame;self.diagnostics=[]
        self.tracks=[t for t in self.tracks if frame-t['points'][-1]['frame']<=self.max_gap]
        normalized=[]
        for c in candidates:
            x1,y1,x2,y2=c['box'];w,h=x2-x1,y2-y1
            if min(w,h)<1 or max(w,h)>24 or w*h>200 or max(w,h)/min(w,h)>4:
                self.diagnostics.append('size_rejected');continue
            c=dict(c,frame=frame,size=math.sqrt(w*h))
            # Merge duplicate evidence for the same spatial observation before association.
            match=next((p for p in normalized if math.hypot(c['x']-p['x'],c['y']-p['y'])<3),None)
            if match is None:normalized.append(c)
            elif c.get('weight',0)>match.get('weight',0):normalized[normalized.index(match)]=c
        edges=[]
        for ci,c in enumerate(normalized):
            for ti,t in enumerate(self.tracks):
                p=t['points'][-1];dt=frame-p['frame'];ratio=c['size']/p['size']
                if not 1/self.max_size_ratio<=ratio<=self.max_size_ratio:continue
                step=math.hypot(c['x']-p['x'],c['y']-p['y'])/dt
                if step>self.max_step:continue
                prediction=np.array([p['x'],p['y']])+np.array(t['velocity'])*dt
                residual=float(np.linalg.norm(np.array([c['x'],c['y']])-prediction))
                if len(t['points'])>=2 and residual>max(8.,2*c['size']+.35*np.linalg.norm(t['velocity']))*dt:continue
                edges.append((residual,ci,ti))
        used_c=set();used_t=set();updated=[]
        for _,ci,ti in sorted(edges):
            if ci in used_c or ti in used_t:continue
            c=normalized[ci];t=self.tracks[ti];p=t['points'][-1];dt=frame-p['frame']
            t['velocity']=((c['x']-p['x'])/dt,(c['y']-p['y'])/dt)
            t['points'].append(c);t['points']=t['points'][-8:]
            used_c.add(ci);used_t.add(ti);updated.append(t)
        for ci,c in enumerate(normalized):
            if ci not in used_c:
                t=dict(id=self.next_id,points=[c],velocity=(0.,0.));self.next_id+=1;self.tracks.append(t);updated.append(t)
        ranked=[]
        for t in updated:
            points=t['points']
            if len(points)<self.min_hits:self.diagnostics.append('temporal_pending');continue
            recent=points[-4:];p=recent[-1];previous=recent[-2]
            displacement=math.hypot(p['x']-recent[0]['x'],p['y']-recent[0]['y'])
            length=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(recent,recent[1:]))
            step=math.hypot(p['x']-previous['x'],p['y']-previous['y'])/(p['frame']-previous['frame'])
            if displacement<self.min_displacement or step<self.min_step:
                self.diagnostics.append('stationary_or_jitter');continue
            straightness=displacement/max(length,1e-9)
            if straightness<self.min_straightness:self.diagnostics.append('direction_inconsistent');continue
            score=min(len(points),6)*.2+straightness*.5+min(p.get('weight',0),.8)*.3
            ranked.append((score,t,straightness,step))
        ranked.sort(key=lambda item:-item[0])
        if not ranked:return None
        active=next((r for r in ranked if r[1]['id']==self.active_id),None)
        if active is not None:
            ranked=[active]
        elif self.active_id is not None and frame-self.last_selected_frame<=self.max_gap:
            self.diagnostics.append('active_track_gap');return None
        if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.12:
            self.diagnostics.append('competing_trajectories');return None
        _,t,straightness,step=ranked[0]
        self.active_id=t['id'];self.last_selected_frame=frame
        return dict(t['points'][-1],trajectory_id=t['id'],straightness=straightness,
                    step_px=step,history=[dict(p) for p in t['points']])
