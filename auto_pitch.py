"""Automatic image-space pitch/marking/goal hypotheses. Not camera calibration."""
import math
import cv2
import numpy as np


def sample_line(mask, a, b):
    points=np.linspace(a,b,max(2,int(np.linalg.norm(np.array(a)-b))))
    x=np.clip(points[:,0].astype(int),0,mask.shape[1]-1)
    y=np.clip(points[:,1].astype(int),0,mask.shape[0]-1)
    return float((mask[y,x]>0).mean())


def infer_geometry(background):
    h,w=background.shape[:2]
    hsv=cv2.cvtColor(background,cv2.COLOR_BGR2HSV)
    grass=((hsv[:,:,0]>=18)&(hsv[:,:,0]<=100)&(hsv[:,:,1]>=25)&(hsv[:,:,2]>=25)).astype('uint8')*255
    grass=cv2.morphologyEx(grass,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    contours,_=cv2.findContours(grass,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=dict(source='automatic_proposal',review_status='unreviewed',image_size=[w,h],pitch_polygon=[],marking_segments=[],boundary_segments=[],goal_candidates=[],endline_candidates=[],metric_calibration=False)
    if not contours:return result
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<.1*w*h:return result
    hull=cv2.convexHull(contour)
    polygon=cv2.approxPolyDP(hull,.015*cv2.arcLength(hull,True),True).reshape(-1,2)
    result['pitch_polygon']=polygon.tolist()
    pitch=np.zeros((h,w),np.uint8);cv2.fillPoly(pitch,[polygon],255)
    result['boundary_segments']=[[a.tolist(),b.tolist()] for a,b in zip(polygon,np.roll(polygon,-1,axis=0))]
    white=((hsv[:,:,1]<85)&(hsv[:,:,2]>145)).astype('uint8')*255
    local=cv2.morphologyEx(cv2.cvtColor(background,cv2.COLOR_BGR2GRAY),cv2.MORPH_TOPHAT,np.ones((9,9),np.uint8))
    markings=cv2.bitwise_and(white,pitch);markings[local<12]=0
    lines=cv2.HoughLinesP(markings,1,np.pi/180,25,minLineLength=30,maxLineGap=12)
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            line=[[int(x1),int(y1)],[int(x2),int(y2)]]
            if len(result['marking_segments'])<50:result['marking_segments'].append(line)
    # Goal proposal: paired near-vertical white posts plus a supported crossbar.
    goal_white=((hsv[:,:,1]<100)&(hsv[:,:,2]>115)).astype('uint8')*255
    raw=cv2.HoughLinesP(goal_white,1,np.pi/180,6,minLineLength=8,maxLineGap=3)
    posts=[]
    if raw is not None:
        for x1,y1,x2,y2 in raw.reshape(-1,4):
            if 8<=abs(y2-y1)<=130 and abs(x2-x1)<=.4*abs(y2-y1):
                a,b=sorted([[int(x1),int(y1)],[int(x2),int(y2)]],key=lambda p:p[1])
                if cv2.pointPolygonTest(polygon.astype(np.float32),(float(b[0]),float(b[1])),True)>=-12:posts.append((a,b))
    merged=[]
    for a,b in sorted(posts,key=lambda p:p[0][0]):
        match=next((p for p in merged if abs((p[0][0]+p[1][0])/2-(a[0]+b[0])/2)<3 and max(p[0][1],a[1])-min(p[1][1],b[1])<8),None)
        if match is None:merged.append([a,b])
        else:
            match[0]=min(match[0],a,key=lambda q:q[1]);match[1]=max(match[1],b,key=lambda q:q[1])
    posts=[p for p in merged if 14<=p[1][1]-p[0][1]<=130]
    goals=[]
    for i,(a,b) in enumerate(posts):
        for c,d in posts[i+1:]:
            separation=abs(a[0]-c[0]);ha=b[1]-a[1];hb=d[1]-c[1]
            if not (12<=separation<=140 and abs(a[1]-c[1])<=25 and  .65<ha/hb<1.5 and 1.1<separation/max(ha,hb)<5):continue
            crossbar=sample_line(cv2.dilate(goal_white,np.ones((5,5),np.uint8)),a,c)
            if crossbar<.85:continue
            left,right=sorted([a[0],c[0]])
            top,bottom=max(a[1],c[1])+3,min(b[1],d[1])-2
            interior=white[top:bottom,left+3:right-3]
            if interior.size<12 or float((interior>0).mean())>.18:continue
            below=grass[max(b[1],d[1]):min(h,max(b[1],d[1])+8),left:right]
            if below.size==0 or float((below>0).mean())<.5:continue
            # Repeated upright structures around the candidate are more likely fencing.
            nearby=sum(left-20<=p[0][0]<=right+20 and abs(p[0][1]-a[1])<12 for p in posts)
            if nearby>12:continue
            # Keep goal-like structures near an estimated pitch edge, not arbitrary field markings.
            center=((b[0]+d[0])/2,(b[1]+d[1])/2)
            edge_distance=abs(cv2.pointPolygonTest(polygon.astype(np.float32),center,True))
            if edge_distance>80:continue
            score=crossbar+.005*min(ha,hb)-.003*abs(ha-hb)
            goals.append(dict(polygon=[a,c,d,b],mouth_line=[b,d],score=round(score,3),status='goal_structure_candidate'))
    for goal in sorted(goals,key=lambda g:-g['score']):
        center=np.mean(goal['polygon'],axis=0)
        if any(np.linalg.norm(center-np.mean(g['polygon'],axis=0))<45 for g in result['goal_candidates']):continue
        result['goal_candidates'].append(goal)
        if len(result['goal_candidates'])==2:break
    for goal in result['goal_candidates']:
        a,b=np.array(goal['mouth_line'],float);direction=b-a
        if np.linalg.norm(direction)<1:continue
        for line in result['marking_segments']:
            c,d=np.array(line,float);v=d-c
            cosine=abs(float(np.dot(v,direction)))/(np.linalg.norm(v)*np.linalg.norm(direction))
            distance=abs(float(direction[0]*(c-a)[1]-direction[1]*(c-a)[0]))/np.linalg.norm(direction)
            if cosine>.94 and distance<18:result['endline_candidates'].append(line)
    return result


def draw_geometry(image, geometry):
    for key,color in [('boundary_segments',(255,180,0)),('marking_segments',(160,160,160)),('endline_candidates',(255,0,255))]:
        for a,b in geometry[key]:cv2.line(image,tuple(a),tuple(b),color,1)
    for i,g in enumerate(geometry['goal_candidates']):
        cv2.polylines(image,[np.array(g['polygon'],np.int32)],True,(0,80,255),2)
        cv2.putText(image,f'GOAL? {i+1}',tuple(g['polygon'][0]),0,.4,(0,80,255),1)
    return image
