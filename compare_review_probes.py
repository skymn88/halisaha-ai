"""Selected AI visual probes: diagnostic examples, never human precision/recall."""
import json
import math
from pathlib import Path
import cv2

PROBES=[('scene_000',375,205,207,'ball','Vuruş öncesinde görünen top'),
        ('scene_000',390,180,174,'ball','Oyuncunun ayağı yanındaki top'),
        ('scene_003',113,600,247,'ball','Vuruş sonrası sağ kenarda görünen top'),
        ('scene_001',97,209,55,'not_ball','Uzak oyuncuya bağlı görünen iz; şut uçuşuyla ilişkisiz (AI yorumu)'),
        ('scene_002',329,380,149,'not_ball','Turuncu oyuncunun ayakkabı/çorap bölgesi; şut uçuşu değil (AI yorumu)')]


def compare(root):
    root=Path(root);rows=[]
    for scene,frame,x,y,kind,note in PROBES:
        row=dict(scene=scene,frame=frame,center=[x,y],ai_object=kind,annotation_source='ai_reviewed',reviewer='assistant_visual_review',note=note,tolerance_px=5)
        row['video_sha256']=json.loads((root/scene/'summary.json').read_text())['video_sha256']
        for name,directory in [('v052_selected',Path('results/full_analysis_v052_release')),('v06_hypotheses',root)]:
            points=[p for p in json.loads((directory/scene/'ball_observations.json').read_text()) if p['frame']==frame]
            delta=min((math.hypot(p['x']-x,p['y']-y) for p in points),default=None)
            row[name]=dict(nearest_px=delta,within_tolerance=delta is not None and delta<=5)
        cap=cv2.VideoCapture(str(Path('results/scene_clips')/(scene+'.mp4')));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();cap.release()
        if not ok:raise ValueError('Probe frame unavailable')
        cv2.circle(image,(x,y),9,(0,255,255),1)
        path=root/f'probe_{scene}_{frame}.png';cv2.imwrite(str(path),image);row['image']=str(path)
        rows.append(row)
    (root/'ai_probes.json').write_text(json.dumps(dict(rows=rows,limitations=['Hand-selected AI examples, not a representative evaluation set.','No human labels or accuracy aggregates.','Multi-hypothesis presence is not single-ball selection accuracy.']),indent=2,ensure_ascii=False))
    return rows
