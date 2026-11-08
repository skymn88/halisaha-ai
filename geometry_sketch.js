for (const article of document.querySelectorAll('article')) {
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
