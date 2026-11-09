for (const article of document.querySelectorAll('article:not(.geometry-editor)')) {
  const canvas=article.querySelector('canvas'), ctx=canvas.getContext('2d'), image=new Image();
  const key='halisaha-geometry-v1-'+article.dataset.scene;
  let lines=[], pending=null;
  try { lines=JSON.parse(localStorage.getItem(key)||'[]'); } catch (_) {}
  function draw(){ctx.drawImage(image,0,0,canvas.width,canvas.height);for(const l of lines){ctx.beginPath();ctx.moveTo(...l.a);ctx.lineTo(...l.b);ctx.strokeStyle=l.kind==='goal_line'?'#ffda44':'#65dfff';ctx.lineWidth=3;ctx.stroke()}if(pending){ctx.fillStyle='#fff';ctx.fillRect(pending[0]-3,pending[1]-3,6,6)}article.querySelector('.geometry-status').textContent=lines.length+' yerel çizgi taslağı · henüz olay hesabında kullanılmıyor';}
  image.onload=draw;image.src=canvas.dataset.image;
  canvas.addEventListener('pointerdown',e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height];if(!pending)pending=p;else{if(Math.hypot(p[0]-pending[0],p[1]-pending[1])>=5)lines.push({kind:article.querySelector('.kind').value,a:pending,b:p,source:'manual_draft',review_status:'unreviewed'});pending=null;try{localStorage.setItem(key,JSON.stringify(lines))}catch(_){}}draw()});
  article.querySelector('.clear').onclick=()=>{lines=[];pending=null;try{localStorage.removeItem(key)}catch(_){}draw()};
  article.querySelector('.download').onclick=()=>{const blob=new Blob([JSON.stringify({scene:article.dataset.scene,image_size:[640,360],lines},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=article.dataset.scene+'-geometry-draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
}

// Version 2: explicit human action, clip hash binding, editable proposals, downloadable inference input.
for(const article of document.querySelectorAll('.geometry-editor')) {
  const canvas=article.querySelector('canvas'),ctx=canvas.getContext('2d'),image=new Image();
  const key='halisaha-geometry-v2-'+article.dataset.hash;
  let goals=[],endlines=[],pending=[],drag=null;
  const status=article.querySelector('.geometry-status');
  const confirmed=article.querySelector('.confirmed');
  function changed(){confirmed.checked=false;try{localStorage.setItem(key,JSON.stringify({goals,endlines}))}catch(_){}draw()}
  try{const stored=JSON.parse(localStorage.getItem(key)||'null');if(stored){goals=stored.goals;endlines=stored.endlines}}catch(_){}
  function points(){return [...goals.flatMap(g=>g.polygon),...endlines.flat(),...pending]}
  function draw(){
    ctx.drawImage(image,0,0,canvas.width,canvas.height);ctx.lineWidth=2;
    for(const [shape,color,closed] of [...goals.map(g=>[g.polygon,'#ffda44',true]),...endlines.map(l=>[l,'#65dfff',false])]){
      ctx.strokeStyle=color;ctx.beginPath();shape.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));if(closed)ctx.closePath();ctx.stroke();
    }
    points().forEach((p,i)=>{ctx.fillStyle='#fff';ctx.fillRect(p[0]-3,p[1]-3,6,6)});
    status.textContent=`${goals.length} kale ağzı · ${endlines.length} çizgi · ${pending.length} bekleyen nokta. Kaydedilene kadar taslak.`;
  }
  const position=e=>{const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(640,(e.clientX-r.left)*640/r.width)),Math.max(0,Math.min(360,(e.clientY-r.top)*360/r.height))]};
  image.onload=draw;image.src=canvas.dataset.image;
  canvas.addEventListener('pointerdown',e=>{
    e.preventDefault();const p=position(e);canvas.setPointerCapture(e.pointerId);
    drag=points().find(q=>Math.hypot(q[0]-p[0],q[1]-p[1])<8)||null;
    if(drag)return;
    pending.push(p);const mode=article.querySelector('.kind').value;
    if(mode==='goal'&&pending.length===4){goals.push({polygon:pending,mouth_line:[pending[0],pending[3]]});pending=[]}
    if(mode==='endline'&&pending.length===2){endlines.push(pending);pending=[]}changed();
  });
  canvas.addEventListener('pointermove',e=>{if(drag){const p=position(e);drag[0]=p[0];drag[1]=p[1];changed()}});
  canvas.addEventListener('pointerup',()=>drag=null);
  article.querySelector('.kind').onchange=()=>{pending=[];changed()};
  article.querySelector('.clear').onclick=()=>{goals=[];endlines=[];pending=[];changed()};
  article.querySelector('.undo').onclick=()=>{if(pending.length)pending.pop();else if(article.querySelector('.kind').value==='goal')goals.pop();else endlines.pop();changed()};
  article.querySelector('.proposal').onclick=()=>{
    const p=JSON.parse(article.querySelector('.geometry-proposals').textContent);
    goals=p.goals.filter(g=>g.polygon.length===4).map(g=>({polygon:g.polygon,mouth_line:[g.polygon[0],g.polygon[3]]}));endlines=p.endlines;pending=[];changed();
    status.textContent+=' Otomatik öneriler yüklendi; direk ayaklarını ve köşe sırasını düzelt.';
  };
  article.querySelector('.download').onclick=()=>{
    const reviewer=article.querySelector('.reviewer').value.trim();
    if(!reviewer||!confirmed.checked||pending.length||(!goals.length&&!endlines.length)){status.textContent='Kaydedilmedi: inceleyen adı, açık onay ve tamamlanmış çizim gerekiyor.';return}
    const validGoal=g=>{
      const p=g.polygon,cross=p.map((a,i)=>{const b=p[(i+1)%4],c=p[(i+2)%4];return (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])});
      const area=Math.abs(p.reduce((s,a,i)=>{const b=p[(i+1)%4];return s+a[0]*b[1]-a[1]*b[0]},0))/2;
      return area>=5&&(cross.every(v=>v>0)||cross.every(v=>v<0));
    };
    if(goals.some(g=>!validGoal(g))||endlines.some(l=>Math.hypot(l[0][0]-l[1][0],l[0][1]-l[1][1])<5)){
      status.textContent='Kaydedilmedi: kale köşeleri çevre sırasıyla, kesişmeden çizilmeli; çizgi en az 5 kaynak pikseli olmalı.';return;
    }
    const record={schema_version:2,scene:article.dataset.scene,video_sha256:article.dataset.hash,image_size:[640,360],annotation_source:'human',review_status:'reviewed',reviewer,reviewed_at:new Date().toISOString(),goals:goals.map(g=>({...g,mouth_line:[g.polygon[0],g.polygon[3]]})),endlines};
    const blob=new Blob([JSON.stringify(record,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=article.dataset.scene+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='İnsan geometri kaydı indirildi. Çıkarımda --geometry-dir ile kullanılır; bu sayfadaki sonuçlar kendiliğinden değişmez.';
  };
}
