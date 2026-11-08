"""Normalized observation density on an illustrative pitch, not metric calibration."""
from itertools import combinations
import json
from pathlib import Path
import cv2
import numpy as np
from player_ball_filter import read

TEMPLATE_LENGTH=45.
TEMPLATE_WIDTH=25.
SOURCE_URLS=[
 'https://www.thefa.com/football-rules-governance/lawsandrules/laws/football-5-5/law-1---the-playing-area',
 'https://nurteks.com.tr/haberler/blog/hali-saha-yapimi-ve-hali-saha-olculeri-hakkinda-her-sey',
]


def visible_quad(polygon):
    """Largest inscribed convex quadrilateral of the visible grass hull; not full pitch corners."""
    if len(polygon)<4:raise ValueError('At least four hull points required')
    hull=cv2.convexHull(np.asarray(polygon,np.float32)).reshape(-1,2)
    if len(hull)>18:hull=cv2.approxPolyDP(hull,.01*cv2.arcLength(hull,True),True).reshape(-1,2)
    choices=[np.asarray(q,np.float32) for q in combinations(hull,4)]
    q=max(choices,key=lambda x:abs(cv2.contourArea(x)))
    if abs(cv2.contourArea(q))<100:raise ValueError('Degenerate visible-field quadrilateral')
    # Preserve cyclic order and start at the upper-left image corner.
    center=q.mean(axis=0)
    q=q[np.argsort(np.arctan2(q[:,1]-center[1],q[:,0]-center[0]))]
    q=np.roll(q,-int(np.argmin(q.sum(axis=1))),axis=0)
    return q


def density_for_scene(root):
    root=Path(root);geometry=json.loads((root/'geometry.json').read_text());summary=json.loads((root/'summary.json').read_text())
    quad=visible_quad(geometry['pitch_polygon'])
    destination=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
    transform=cv2.getPerspectiveTransform(quad,destination)
    points=[]
    for row in read(Path('results/measured')/root.name/'detections.csv'):
        if row['class_name']=='person':points.append([float(row['foot_x']),float(row['foot_y'])])
    mapped=cv2.perspectiveTransform(np.asarray(points,np.float32).reshape(-1,1,2),transform).reshape(-1,2) if points else np.empty((0,2))
    valid=np.isfinite(mapped).all(axis=1)&(mapped>=0).all(axis=1)&(mapped<=1).all(axis=1)
    mapped=mapped[valid]
    grid=np.zeros((50,90),np.float32)
    for x,y in mapped:
        grid[min(49,int(y*50)),min(89,int(x*90))]+=1/summary['fps']
    smoothed=cv2.GaussianBlur(grid,(0,0),1.8)
    if smoothed.sum()>0:smoothed/=smoothed.sum()
    audit=dict(template_length_m=TEMPLATE_LENGTH,template_width_m=TEMPLATE_WIDTH,dimensions_measured=False,
               method='visible-hull quadrilateral normalized to a schematic; not verified full-pitch homography',
               source_quad=quad.tolist(),transform=transform.tolist(),input_observations=len(points),
               mapped_observations=int(valid.sum()),excluded_observations=int((~valid).sum()),
               observed_person_seconds=float(grid.sum()),metric_calibration_valid=False,
               orientation='image-relative; attacking direction and full-field orientation unknown',
               color_scale='per-scene maximum; colors compare locations within a clip, not activity between clips',sources=SOURCE_URLS)
    return smoothed,audit


def render(grid,maximum):
    margin=55;width,height=900,500
    image=np.full((height+2*margin,width+2*margin,3),(30,45,35),np.uint8)
    pitch=np.zeros((height,width,3),np.uint8)
    for x in range(width):pitch[:,x]=(48,105 if (x//90)%2 else 115,45)
    strength=cv2.resize(grid,(width,height),interpolation=cv2.INTER_CUBIC)/max(maximum,1e-12)
    strength=np.clip(strength,0,1)
    colors=cv2.applyColorMap((strength*255).astype('uint8'),cv2.COLORMAP_JET)
    alpha=(np.sqrt(strength)*.8)[:,:,None]
    pitch=(pitch*(1-alpha)+colors*alpha).astype('uint8')
    image[margin:margin+height,margin:margin+width]=pitch
    white=(230,240,235)
    a,b=margin,margin+width;top,bottom=margin,margin+height;mid=margin+height//2
    cv2.rectangle(image,(a,top),(b,bottom),white,2)
    cv2.line(image,(a+width//2,top),(a+width//2,bottom),white,2)
    cv2.circle(image,(a+width//2,mid),60,white,2)
    cv2.circle(image,(a+width//2,mid),4,white,-1)
    for side in (0,1):
        x=a if side==0 else b;sign=1 if side==0 else -1
        cv2.rectangle(image,(x,mid-120),(x+sign*100,mid+120),white,2)
        cv2.rectangle(image,(x-sign*15,mid-40),(x,mid+40),white,2)
        cv2.circle(image,(x+sign*120,mid),3,white,-1)
    cv2.putText(image,'45 x 25 m ORNEK SEMA - SAHA OLCUMU DEGIL',(margin,28),0,.55,white,1)
    cv2.putText(image,'DUSUK',(margin,height+margin+34),0,.45,white,1)
    gradient=np.tile(np.arange(256,dtype=np.uint8),(12,1));image[height+margin+20:height+margin+32,margin+75:margin+331]=cv2.applyColorMap(gradient,cv2.COLORMAP_JET)
    cv2.putText(image,'YUKSEK GORELI YOGUNLUK',(margin+345,height+margin+33),0,.45,white,1)
    return image


def export():
    scenes=[]
    for root in sorted(p for p in Path('results/full_analysis').glob('scene_*') if p.is_dir()):
        grid,audit=density_for_scene(root);scenes.append((root,grid,audit))
    for root,grid,audit in scenes:
        maximum=float(grid.max())
        audit['maximum_density']=maximum
        cv2.imwrite(str(root/'schematic_heatmap.png'),render(grid,maximum))
        (root/'schematic_heatmap.json').write_text(json.dumps(audit,indent=2))
        # Source transform visible to reviewers; no hidden calibration claim.
        im=cv2.imread(str(root/'geometry.png'))
        cv2.polylines(im,[np.asarray(audit['source_quad'],np.int32)],True,(255,255,0),2)
        for i,p in enumerate(audit['source_quad']):cv2.putText(im,str(i+1),tuple(map(int,p)),0,.7,(255,255,0),2)
        cv2.imwrite(str(root/'heatmap_projection.png'),im)
        print(root.name,audit['mapped_observations'],'/',audit['input_observations'])


if __name__=='__main__':export()
