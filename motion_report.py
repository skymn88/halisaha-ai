"""Standalone comparison of white-motion proposal experiments; no event claims."""
import base64
import json
from pathlib import Path
import subprocess


def export(output='results/top-deneyi.html'):
    cards=[];records=[]
    for run in sorted(Path('results/white_context').glob('scene_*')):
        new=json.loads((run/'summary.json').read_text())
        old=json.loads((Path('results/white_motion')/run.name/'summary.json').read_text())
        baseline=json.loads((Path('results/measured')/run.name/'summary.json').read_text())
        web=run/'web.mp4'
        if not web.exists():subprocess.run(['ffmpeg','-loglevel','error','-i',str(run/'proposals.mp4'),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(web)],check=True)
        video=base64.b64encode(web.read_bytes()).decode()
        records.append(dict(scene=run.name,baseline_yolo_proposals=baseline['raw_ball_candidates'],white_motion=old,white_context=new))
        cards.append(f'''<article><h2>{run.name}</h2><video controls playsinline preload="none" src="data:video/mp4;base64,{video}"></video><p>YOLO ham top adayı: {baseline['raw_ball_candidates']}</p><p>Hareketli beyaz bölge: {old['candidates']} aday</p><p>Çim çevresi koşuluyla: {new['candidates']} aday · kare başına {new['candidates']/new['frames']:.1f}</p><p>Ardışık destekli aday bulunan kare: {new['temporal_candidate_frames']}/{new['frames']}</p><p>Sarı kutular doğrulanmış top değildir. Forma ve ayakkabı yanlış adayları sürebilir. Gol / şut / pas: bu deney tarafından doğrulanmadı.</p></article>''')
    Path(output).write_text('''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top adayı deneyi</title><style>body{background:#101820;color:#eef;font:16px system-ui;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}article{background:#203340;padding:16px;border-radius:12px;min-width:0}video{width:100%}p{line-height:1.5}</style><h1>Küçük hareketli beyaz bölge deneyi</h1><p>Bu deney YOLO'yu tekrar çalıştırmaz. Beyazlık, arka plan farkı, küçük bölge boyutu ve zamansal eşleşme kullanılır. Çim çevresi koşulu aday yükünü azaltmak için denenmiştir; havadaki veya oyuncuyla örtüşen gerçek topu kaçırabilir. Aday sayısı doğruluk veya top geri çağırımı değildir. Ana takip çıktısı değişmedi.</p><main>'''+''.join(cards)+'</main></html>',encoding='utf-8')
    Path(output).with_suffix('.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':export()
