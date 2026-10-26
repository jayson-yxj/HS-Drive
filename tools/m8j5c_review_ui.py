"""Add exact review questions, verify the private 8x6 viewer in an offline browser."""
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from tools.m8j5c_candidates import ROOT,REPORT


def main():
    web=ROOT/'web';path=web/'index.html';page=path.read_text()
    qs=json.loads((ROOT/'targeted_review_questions.json').read_text())
    rows=''.join('<tr><td>'+html.escape(str(q['ordinal']))+'</td><td>'+html.escape(q['camera'])+'</td><td>'+
        html.escape(str(q['bbox_xyxy']))+'</td><td>'+html.escape(str(q['semantic']))+'</td><td>'+
        html.escape(str(q['pixels']))+'</td><td>'+html.escape(q['question'])+'</td><td>Pending</td></tr>' for q in qs)
    panel='<section id="review-questions"><h2>27个具体疑点（未人工签核）</h2><p>任何结论都需要逐帧六摄全部任务对象复核；以下只列已知疑点，不是审核一框即全帧通过。</p><table><thead><tr><th>帧</th><th>相机</th><th>像素框</th><th>语义类</th><th>像素数</th><th>待核实问题</th><th>状态</th></tr></thead><tbody>'+rows+'</tbody></table></section>'
    page=page.replace('</main>',panel+'</main>');path.write_text(page)
    test="""<script>
const checks={};checks.eight_frames=data.length===8;checks.question_count=document.querySelectorAll('#review-questions tbody tr').length===27;
checks.all_pending=Array.from(document.querySelectorAll('#review-questions tbody tr')).every(r=>r.lastElementChild.textContent==='Pending');
checks.no_training=data.every(f=>f.negative_enabled===false&&f.qualification.qualified===false);
checks.each_six_cameras=data.every(f=>Object.keys(f.images).length===6);
checks.all_original_objects=data.every(f=>f.objects.length>0);
checks.no_review_write_controls=document.querySelectorAll('button,input[type=file]').length===0;
for(let i=0;i<8;i++){index=i;render();checks['frame_'+data[i].ordinal]=document.querySelectorAll('svg[data-camera]').length===6;}
$('mode').value='semantic_color';render();checks.semantic_switch=document.querySelector('svg image').getAttribute('href').includes('semantic_color');
$('converted').checked=true;render();checks.converted_layer=Boolean(document.querySelector('[data-label$=" C"]'));
const pre=document.createElement('pre');pre.id='test-result';pre.textContent=JSON.stringify({passed:Object.values(checks).every(Boolean),checks});document.body.append(pre);
</script>"""
    (web/'test.html').write_text(page.replace('</body>',test+'</body>'))
    cmd=['unshare','--net','/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu',
         '--disable-background-networking','--no-first-run','--user-data-dir='+str(ROOT/'chrome'),
         '--dump-dom','--virtual-time-budget=2000',(web/'test.html').as_uri()]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
    (ROOT/'browser_stderr.log').write_text(p.stderr)
    match=re.search(r'<pre id="test-result">(.*?)</pre>',p.stdout,re.S)
    assert match and p.returncode==0
    result=json.loads(html.unescape(match[1]));assert result['passed']
    result.update(evidence='ActualOfflineBrowser',private_page=str(path),page_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  human_signoff='Pending',signoffs_created=0,network='Disabled')
    with (REPORT/'m8j5c_ui_validation.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
