"""Small human pilot: two raw frames and one random event window per clip; all unreviewed."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import cv2
from prepare_review import source_record
from ball_cache import sha256_file
from review_schema import TABLES,write_rows,load_dataset


def prepare(output):
    root=Path(output);root.mkdir(exist_ok=False)
    labels=json.loads(Path('configs/ai_event_windows_v06.json').read_text())['scenes']
    sources=[];windows=[];frames=[];rng=random.Random(42)
    for label in labels:
        name=label['scene'];video=Path('results/scene_clips')/(name+'.mp4')
        source=source_record(name,video,Path('results/measured')/name);sources.append(source)
        cap=cv2.VideoCapture(str(video));fps=source['fps']
        random_start=rng.randrange(max(1,source['frame_count']-60))
        tasks=[('detections',max(0,label['start_frame']-12),1,'before_ai_event'),('detections',label['anchor_frame'],1,'ai_event_context'),('events',random_start,min(60,source['frame_count']-random_start),'random_audit_seed42')]
        for kind,start,length,selection in tasks:
            wid=f'w{len(windows):03d}';clip=f'clips/{wid}.mp4';(root/'clips').mkdir(exist_ok=True)
            windows.append(dict(window_id=wid,source_id=name,scene_id=0,start_frame=start,end_frame=start+length,kind=kind,selection=selection,split='dev',split_group=name,angle='unverified_'+name,angle_reviewed=False,clip=clip))
            subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(start/fps),'-i',str(video),'-frames:v',str(length),'-c:v','libx264','-pix_fmt','yuv420p','-an',str(root/clip)],check=True)
            if kind=='detections':
                cap.set(cv2.CAP_PROP_POS_FRAMES,start);ok,image=cap.read()
                if not ok:raise ValueError('Missing pilot frame')
                directory=root/'frames'/wid;directory.mkdir(parents=True);cv2.imwrite(str(directory/f'{start:06d}.png'),image)
                frames.append(dict(window_id=wid,source_id=name,scene_id=0,frame=start,ball_state='unreviewed',people_complete=0,identities_complete=0,review_status='unreviewed',annotation_source='proposal',reviewer='',notes=''))
        cap.release()
    config=dict(person_iou=.5,ball_center_tolerance_px=5.,event_tolerances_s=[.5,1.],provenance='preset_before_human_review',source_coordinates=True)
    (root/'evaluation_config.json').write_text(json.dumps(config,indent=2))
    dataset=dict(schema_version=1,sources=sources,windows=windows,selection_seed=42,evaluation_config_sha256=sha256_file(root/'evaluation_config.json'),labeling_status='unreviewed',limitations=['Development set, not held-out accuracy.','No human labels. AI localization only selects tasks; it is not imported as truth.','Existing evaluate_review evaluates original detector raw/selected cache; v0.6 alternative hypotheses are a separate output.'])
    (root/'dataset.json').write_text(json.dumps(dataset,indent=2))
    for filename,fields in TABLES.items():write_rows(root/filename,fields,frames if filename=='frame_labels.csv' else [])
    load_dataset(root)
    print(f'{len(frames)} unreviewed frames, 9 random event windows: {root}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/human_pilot_v06');a=p.parse_args();prepare(a.output)
