"""Actual local coverage page test with network namespace disabled."""
import html
import json
import re
import subprocess

from tools.m8j5b_build_coverage import ROOT,save


def main():
    web=ROOT/'web';page=(web/'index.html').read_text()
    test='''<script>
const checks=[];function ok(x,name){if(!x)throw Error(name);checks.push(name);}
try{ok(data.length===5,'five_frames');ok(document.querySelectorAll('svg[data-camera]').length===6,'six_cameras');ok(data[index].ordinal==='00124','missing_walker_default');ok(document.querySelector('[data-layer="missing"]'),'missing_regions');ok(document.querySelector('[data-layer="local-ignore"]'),'local_ignore');ok(document.querySelectorAll('[data-layer="frame-ignore"]').length===6,'frame_fallback_explicit');$('raw').checked=false;$('converted').checked=true;render();ok(document.querySelector('polygon[data-label$=" C"]'),'converted_switch');$('mode').value='instance';render();ok(document.querySelector('image').getAttribute('href').includes('instance'),'instance_switch');ok(!document.getElementById('approve'),'no_human_signoff');const p=document.createElement('pre');p.id='test-result';p.textContent=JSON.stringify({passed:true,checks});document.body.append(p);}catch(e){const p=document.createElement('pre');p.id='test-result';p.textContent=JSON.stringify({passed:false,checks,error:String(e)});document.body.append(p);}
</script>'''
    target=web/'ui_test_private.html';target.write_text(page.replace('</body>',test+'</body>'))
    cmd=['unshare','--net','/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking',
        '--no-first-run','--user-data-dir='+str(ROOT/'chrome_ui_test'),'--dump-dom','--virtual-time-budget=1500',target.as_uri()]
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
    (ROOT/'ui_chrome_stderr.log').write_text(proc.stderr)
    m=re.search(r'<pre id="test-result">(.*?)</pre>',proc.stdout,re.S)
    if not m:raise RuntimeError('Browser result missing')
    result=json.loads(html.unescape(m[1]));assert result['passed'] and proc.returncode==0,result
    result.update(evidence='ActualPrivatePage_HeadlessChrome',network='Disabled',signoffs_created=0)
    save('reports/m8j5b_ui_validation.json',result);print(json.dumps(result))


if __name__=='__main__':main()
