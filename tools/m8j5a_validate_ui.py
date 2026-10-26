"""Headless actual private-page read-only interaction regression; no signoff."""
import html
import json
import re
import subprocess
from pathlib import Path

from tools.m8j5a_audit import OUT,save,report


def main():
    web=OUT/'web';page=(web/'index.html').read_text()
    test=r'''<script>
const checks=[];function ok(value,name){if(!value)throw Error(name);checks.push(name);}
try {
 ok(index===17,'default_00017');ok(document.querySelectorAll('#frame option').length===214,'all214frames');
 ok(document.querySelector('[data-camera="CAM_BACK"] polygon[data-label="393 R"]'),'rear393_visible_while_focus394');
 ok(document.querySelector('[data-camera="CAM_FRONT"] polygon[data-label="394 C"]'),'front394_converted_visible');
 $('focus').value='393';render();ok(document.querySelector('polygon[data-label="394 R"]'),'focus_does_not_hide_other');
 $('raw').checked=false;$('converted').checked=false;$('final').checked=true;render();ok(document.querySelectorAll('polygon').length===0,'final_empty_explicit');
 $('converted').checked=true;render();ok(document.querySelectorAll('polygon').length>1,'all_converted_switch');
 index=124;render(true);ok(!data.frames[index].objects.some(o=>o.source_class==='walker'),'raw_walker_absent124');
 $('regions').checked=true;render();ok(document.querySelector('[data-camera="CAM_FRONT_LEFT"] rect'),'unexplained_region_overlay');
 $('modality').value='semantic';render();ok(document.querySelector('image').getAttribute('href').includes('semantic'),'semantic_switch');
 $('modality').value='instance';render();ok(document.querySelector('image').getAttribute('href').includes('instance'),'instance_switch');
 ok(!document.getElementById('approve')&&!document.getElementById('export'),'signoff_controls_removed');
 const p=document.createElement('pre');p.id='test-result';p.textContent=JSON.stringify({passed:true,checks});document.body.append(p);
}catch(e){const p=document.createElement('pre');p.id='test-result';p.textContent=JSON.stringify({passed:false,checks,error:String(e)});document.body.append(p);}
</script>'''
    target=web/'ui_test_private.html';target.write_text(page.replace('</body>',test+'</body>'))
    command=['unshare','--net','/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu',
        '--disable-background-networking','--no-first-run','--user-data-dir='+str(OUT/'chrome_ui_test'),
        '--dump-dom','--virtual-time-budget=1500',target.as_uri()]
    proc=subprocess.run(command,capture_output=True,text=True,timeout=45)
    (OUT/'ui_chrome_stderr.log').write_text(proc.stderr)
    m=re.search(r'<pre id="test-result">(.*?)</pre>',proc.stdout,re.S)
    if not m:raise RuntimeError('Browser result missing')
    result=json.loads(html.unescape(m[1]));assert result['passed'],result
    result.update(evidence='ActualPrivate214FramePage_HeadlessChrome_ReadOnly',network='Disabled',actual_signoffs=0,
        layers=['Raw','Converted','Final'],golden_requirement='Suspended',chrome_exit=proc.returncode)
    save(report('m8j5a_ui_validation.json'),result);print(json.dumps(result))


if __name__=='__main__':main()
