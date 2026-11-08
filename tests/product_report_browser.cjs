const {chromium}=require('playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 for(const javascript of [false,true]){
  const context=await browser.newContext({javaScriptEnabled:javascript,viewport:{width:390,height:844}});
  const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('http://localhost:8888/**',r=>r.fulfill({contentType:'text/html',body:fs.readFileSync(path.resolve('results/halisaha-analiz-v051.html'))}));
  await page.goto('http://localhost:8888/');
  if(await page.locator('section:visible').count()!==9)throw Error('Not all scenes accessible');
  if(await page.locator('.heatmap:visible').count()!==9)throw Error('Heatmaps hidden');
  for(let i=0;i<9;i++){
   await page.locator('#contents a').nth(i).click();
   if(!page.url().endsWith('#scene-'+i))throw Error('Scene anchor failed');
   const section=page.locator('#scene-'+i);
   await section.locator('a[href="#heat-'+i+'"]').click();
   if(!page.url().endsWith('#heat-'+i))throw Error('Heat anchor failed');
   const img=section.locator('.heatmap');
   if(!await img.evaluate(img=>img.complete&&img.naturalWidth>0))throw Error('Heatmap image failed');
   const bounds=await img.boundingBox();if(!bounds||bounds.width>390)throw Error('Mobile heatmap overflow');
   if(javascript){
    const v=section.locator('video');await v.evaluate(async v=>{v.muted=true;await v.play()});
    await page.waitForFunction(i=>document.querySelectorAll('video')[i].currentTime>0,i);
    await v.evaluate(v=>v.pause());
    if(await section.locator('.seek').count()){
     const link=section.locator('.seek').first(),t=Number(await link.getAttribute('data-time'));await link.click();
     if(Math.abs(await v.evaluate(v=>v.currentTime)-t)>.1)throw Error('Event seek failed');
    }
   }
  }
  await page.locator('#contents a').nth(1).click();
  await page.screenshot({path:path.resolve('results/browser/v051-mobile-'+javascript+'.png')});
  if(javascript){await page.setViewportSize({width:1440,height:1000});await page.locator('#scene-1').screenshot({path:path.resolve('results/browser/v051-desktop.png')})}
  if(errors.length)throw Error(errors.join('\n'));
  console.log(`JS ${javascript}: 9 scene links, 9 heatmap links, visible images passed`);
  await context.close();
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
