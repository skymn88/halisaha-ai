"""Combined per-clip diagnostics, player pixel movement and local geometry sketches."""
import base64
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import subprocess
import cv2
import numpy as np
from player_ball_filter import read


def export(output='results/gelisim-raporu.html'):
    cards=[];records=[]
    for run in sorted(Path('results/torso_filter').glob('scene_*')):
        name=run.name;s=json.loads((run/'summary.json').read_text());base=Path('results/measured')/name
        tracks=defaultdict(list)
        for r in read(base/'detections.csv'):
            if r['class_name']=='person' and int(r['track_id'])>=0:
                tracks[(r['scene_id'],r['track_id'])].append(r)
        movements=[]
        for key,rs in tracks.items():
            distance=0;steps=0
            for a,b in zip(rs,rs[1:]):
                if int(b['frame'])-int(a['frame'])!=1:continue
                distance+=float(np.hypot(float(b['foot_x'])-float(a['foot_x']),float(b['foot_y'])-float(a['foot_y'])))
                steps+=1
            movements.append(dict(track_id=key[1],observations=len(rs),consecutive_steps=steps,observed_pixel_displacement=round(distance,1)))
        movements.sort(key=lambda r:-r['observations'])
        cap=cv2.VideoCapture('results/scene_clips/'+name+'.mp4');ok,image=cap.read();fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if not ok:raise ValueError('No scene image')
        heat=np.zeros(image.shape[:2],np.float32)
        for rs in tracks.values():
            for r in rs:
                x,y=int(float(r['foot_x'])),int(float(r['foot_y']))
                if 0<=y<heat.shape[0] and 0<=x<heat.shape[1]:heat[y,x]+=1
        heat=cv2.GaussianBlur(heat,(0,0),12)
        overlay=cv2.addWeighted(image,.55,cv2.applyColorMap((255*heat/max(1,float(heat.max()))).astype('uint8'),cv2.COLORMAP_TURBO),.45,0)
        def uri(im):
            _,p=cv2.imencode('.png',im);return 'data:image/png;base64,'+base64.b64encode(p).decode()
        web=run/'web.mp4'
        if not web.exists():subprocess.run(['ffmpeg','-loglevel','error','-i',str(run/'review.mp4'),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(web)],check=True)
        video='data:video/mp4;base64,'+base64.b64encode(web.read_bytes()).decode()
        lines=''.join(f"<tr><td>P{r['track_id']}</td><td>{r['observations']}</td><td>{r['observations']/fps:.2f}</td><td>{r['observed_pixel_displacement']}</td></tr>" for r in movements[:12])
        records.append(dict(scene=name,filter=s,tracks=movements,goal_geometry='unmarked',shots=None,passes=None,running_meters=None))
        cards.append(f'''<article data-scene="{name}"><h2>{name}</h2><video controls playsinline preload="none" src="{video}"></video><p>Yeşil: kişi kutuları. Sarı: doğrulanmamış top adayları.</p><div class="stats"><b>{s['proposals']}</b> önce · <b>{s['suppressed']}</b> gövde örtüşmesi nedeniyle bastırılan · <b>{s['retained']}</b> kalan aday</div><p>Bu azalma doğruluk değildir. Ayak çevresini koruyoruz; baş/gövdeyle gerçek top teması da yanlış bastırılabilir.</p><h3>Kişi gözlem yoğunluğu</h3><img src="{uri(overlay)}"><h3>En uzun gözlenen geçici izler</h3><table><tr><th>ID</th><th>Kare</th><th>Gözlem süresi (s)</th><th>Ardışık hareket (px)</th></tr>{lines}</table><p>Piksel hareketi kutu titremesi ve kimlik hatası içerir; koşu mesafesi/hız değildir. Kamera perspektifi nedeniyle oyuncu sıralaması yapılmaz. Kimlikler kesitler arasında birleştirilmez.</p><details><summary>Kale / aut çizgisi işaretleme taslağı</summary><p>İki noktaya tıklayarak görüntü üzerinde çizgi oluştur. Bunlar yerel taslaklardır; saha kalibrasyonu veya onaylanmış gol çizgisi değildir.</p><select class="kind"><option value="goal_line">Kale çizgisi</option><option value="end_line">Aut çizgisi</option></select><canvas width="640" height="360" data-image="{uri(image)}"></canvas><button class="clear">Çizgileri temizle</button><button class="download">Çizgi taslağını indir</button><p class="geometry-status">İşaretleme yok. Şut / gol sayacı üretilmedi.</p></details><p>Pas, gerçek koşu mesafesi, şut ve gol: doğrulanmış veri yetersiz. Beyaz adayın çizgiden geçmesi tek başına şut/gol sayılmaz.</p></article>''')
    script=Path('geometry_sketch.js').read_text()
    Path(output).write_text('''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Halı Saha · Gelişim</title><style>body{background:#101820;color:#edf;font:16px system-ui;margin:16px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:20px}article{background:#203340;border-radius:12px;padding:16px;min-width:0}video,img,canvas{width:100%}canvas{touch-action:none}button,select{padding:12px;margin:8px 0}table{width:100%;font-size:13px}td,th{padding:5px;text-align:left}p{line-height:1.5}.stats{padding:12px;background:#304653}</style><h1>Oyuncu örtüşmesi ve hareket raporu</h1><p>Üç ardışık kare boyunca aynı oyuncunun üst %80'lik kutu alanında, benzer göreli konumda kalan aday bastırılır. Tam piksel segmentasyonu değildir. Ayak çevresi korunur ve aday gövdeden çıkınca yeniden değerlendirilir. İnsan etiketi yok; artan sayılar veya daha az kutu tek başına başarı kanıtı değildir.</p><main>'''+''.join(cards)+'</main><script>'+script+'</script></html>',encoding='utf-8')
    Path(output).with_suffix('.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':export()
