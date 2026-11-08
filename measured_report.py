"""Export a standalone evidence report; operational measurements are not accuracy scores."""
import argparse
import base64
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from event_candidates import Possession


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def export(root, manifest, output):
    manifest = json.loads(Path(manifest).read_text())
    fps = manifest['source_fps']
    cards, measurements = [], []
    for clip in manifest['clips']:
        run = Path(root)/Path(clip['file']).stem
        summary = json.loads((run/'summary.json').read_text())
        detections = rows(run/'detections.csv')
        balls = rows(run/'ball_track.csv')
        people = defaultdict(list)
        for row in detections:
            if row['class_name'] == 'person':
                people[int(row['frame'])].append(row)
        cap = cv2.VideoCapture(str(run/'tracked.mp4'))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if decoded == 0: background = frame.copy()
            decoded += 1
        cap.release()
        if decoded != clip['decoded_frames'] or len(balls) != decoded:
            raise ValueError('Frame mismatch: '+str(run))
        heat = np.zeros(background.shape[:2], np.float32)
        ownership = Counter()
        tracker, scene = None, None
        for b in balls:
            if b['scene_id'] != scene:
                tracker, scene = Possession(), b['scene_id']
            ps = people[int(b['frame'])]
            candidates = []
            for p in ps:
                x, y = int(float(p['foot_x'])), int(float(p['foot_y']))
                if 0 <= y < heat.shape[0] and 0 <= x < heat.shape[1]: heat[y,x] += 1
                candidates.append(dict(track_id=int(p['track_id']),confidence=float(p['confidence']),box=[float(p[k]) for k in ('x1','y1','x2','y2')]))
            ball = dict(x=float(b['center_x']),y=float(b['center_y']),confidence=float(b['confidence'])) if b['state']=='observed' else None
            state = tracker.update(float(b['time_s']),ball,candidates)
            ownership[state['state']] += 1
        heat = cv2.GaussianBlur(heat,(0,0),12)
        colors = cv2.applyColorMap((heat/max(float(heat.max()),1)*255).astype('uint8'),cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(background,.55,colors,.45,0)
        _, png = cv2.imencode('.png',overlay)
        observed = sum(b['state']=='observed' for b in balls)
        record = dict(scene=clip['parent_scene_id'],source_start_frame=clip['source_start_frame'],frames=decoded,duration_s=decoded/fps,player_observations=sum(map(len,people.values())),mean_person_boxes_per_frame=sum(map(len,people.values()))/decoded,track_ids=summary['unique_player_ids'],possible_id_transitions=summary['possible_id_transitions'],raw_ball_candidates=summary['raw_ball_candidates'],observed_ball_frames=observed,ball_coverage_percent=100*observed/decoded,ball_observation_s=observed/fps,ownership_hypothesis_seconds={k:v/fps for k,v in ownership.items()},human_reviewed_events=0,confirmed_event_statistics=None,rating=None)
        measurements.append(record)
        web = run/'tracked_web.mp4'
        if not web.exists():
            subprocess.run(['ffmpeg','-loglevel','error','-i',str(run/'tracked.mp4'),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(web)],check=True)
        video = base64.b64encode(web.read_bytes()).decode()
        picture = base64.b64encode(png).decode()
        state_names = {'unknown':'Bilinmiyor','player':'Oyuncu kontrolü adayı','contested':'Mücadele adayı','free_ball':'Serbest top adayı'}
        possession_text = ' · '.join(f'{state_names[k]}: {v:.2f} s' for k,v in record['ownership_hypothesis_seconds'].items())
        cards.append(f'''<article><h2>Kesit {record['scene']+1} · {record['duration_s']:.1f} s</h2><video controls playsinline preload="none" src="data:video/mp4;base64,{video}"></video><dl><dt>İşlenen / doğrulanan kare</dt><dd>{decoded}</dd><dt>Kişi kutusu gözlemi</dt><dd>{record['player_observations']} · kare başına {record['mean_person_boxes_per_frame']:.1f}</dd><dt>Geçici takip ID sayısı (oyuncu sayısı değildir)</dt><dd>{record['track_ids']}</dd><dt>Olası ID geçişi (kanıtlanmış hata değildir)</dt><dd>{record['possible_id_transitions']}</dd><dt>Ham top adayı</dt><dd>{record['raw_ball_candidates']}</dd><dt>Seçilmiş top gözlemi</dt><dd>{observed} kare · {record['ball_observation_s']:.2f} s · %{record['ball_coverage_percent']:.1f} kapsama</dd></dl><h3>Top kontrolü hipotezi · doğrulanmadı</h3><p>{html.escape(possession_text)}</p><h3>Görüntü düzleminde kişi gözlem yoğunluğu</h3><img src="data:image/png;base64,{picture}"><p>Her kesitte ayrı normalize edilir; saha mesafesi, oyuncu performansı veya kesitler arası yoğunluk karşılaştırması değildir. Çift kutular yoğunluğu etkileyebilir.</p><p>Pas / şut / top kazanma / reyting: doğrulanmış olay ve kimlik verisi yetersiz; hesaplanmadı.</p></article>''')
    output = Path(output)
    output.write_text('''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Halı Saha · Ölçüm raporu</title><style>body{background:#101820;color:#edf2f7;font:16px system-ui;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:20px}article{background:#203340;border-radius:14px;padding:18px;min-width:0}video,img{width:100%}dt{color:#abc;margin-top:12px}dd{margin:4px 0;font-weight:bold}p{line-height:1.5}</style><h1>Halı Saha · Ölçülmüş takip çıktıları</h1><p>YOLO11n · imgsz 640 · stride 1. Kişi/top gözlemleri ve kontrol hipotezleri otomatik; insan onayı yok. Kapsama doğruluk değildir. Video kutularını izleyerek hataları inceleyebilirsin. Olaylar bilinmiyor; sıfır olay iddiası yok.</p><main>'''+''.join(cards)+'</main></html>',encoding='utf-8')
    output.with_suffix('.json').write_text(json.dumps(measurements,indent=2),encoding='utf-8')
    return measurements


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',default='results/measured')
    p.add_argument('--manifest',default='results/scene_clips/manifest.json')
    p.add_argument('--output',default='results/olcum-raporu.html')
    a=p.parse_args()
    print(json.dumps(export(a.runs,a.manifest,a.output),indent=2))
