"""Headless browser checks using 36 SYNTHETIC candidates, never real signoff."""
import copy
import argparse
import html
import json
import re
import subprocess
from pathlib import Path

from datasets.m8j5_supervision import digest,validate_signoff


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suffix',default='run1');args=ap.parse_args()
    root=Path(json.loads(Path('configs/m8j5_protocol.json').read_text())['private_root'])
    out=root/('ui_synthetic_test_'+args.suffix);out.mkdir(exist_ok=False)
    rows=[]
    for i in range(36):
        obj=dict(model_class='car',source_class='vehicle',geometry_status='Valid',reason='SyntheticOnly',
            yaw_only_comparable=i!=0,center_hs_m=[10,0,1],dimensions_lwh_m=[4,2,2],yaw_ccw_rad=0,
            corners_hs_m=[],distance_xy_m=10,tilt_deg=0,views={})
        trust=dict(geometry_validity='Valid',class_validity='Valid',identity_validity='Unknown',visibility='Synthetic',
            PETR_common_region=True,supervision_eligibility=dict(reasons=['SyntheticOnly']))
        c=dict(candidate_id='SYNTHETIC_%02d'%i,ordinal='SYNTHETIC',object=obj,trust=trust,selection_tags=['SYNTHETIC_NOT_REAL_REVIEW'])
        c['candidate_sha256']=digest(c);rows.append(c)
    payload=dict(candidates=[dict(c,rgb_image='',instance_image='') for c in rows])
    page=Path('tools/m8j5_review_ui.html').read_text().replace('__REVIEW_DATA__',json.dumps(payload)).replace('__DESTINATION__','SYNTHETIC_NO_SERVER_WRITE')
    harness=r'''<script>
const checks=[];function ok(value,name){if(!value)throw Error(name);checks.push(name);}
try {
 ok($('export').disabled,'export_disabled_initially');ok($('approve').disabled,'ineligible_approval_disabled');
 choose('NeedsReview');ok(responses.size===0,'no_anonymous_review');
 $('reviewer').value='SYNTHETIC_BROWSER_TEST';$('source').value='SYNTHETIC_UNIT_TEST_NOT_HUMAN_SIGNOFF';
 choose('ApprovePositive');ok(responses.size===0,'ineligible_approval_rejected');choose('Reject');
 current=1;render();choose('ApprovePositive');ok(responses.size===1,'four_confirmations_required');
 for(const id of checkIds)$(id).checked=true;choose('ApprovePositive');ok(responses.size===2,'eligible_synthetic_approval');
 current=2;render();choose('NeedsReview');ok(responses.size===3,'unknown_retained');
 for(let i=3;i<36;i++){current=i;render();choose('NeedsReview');}
 ok(!$('export').disabled,'export_enabled_after_all36');let exported=buildExport();
 ok(exported.length===36,'36_exports');ok(exported[0].decision==='Reject'&&exported[1].decision==='ApprovePositive'&&exported[2].decision==='NeedsReview','all_three_decisions');
 ok(exported.every(r=>r.reviewed_at.endsWith('Z')&&r.review_source.includes('client_clock_not_platform_verified')),'timestamps_and_sources');
 let blobMade=false;URL.createObjectURL=blob=>{blobMade=blob instanceof Blob;return 'blob:synthetic';};URL.revokeObjectURL=()=>{};HTMLAnchorElement.prototype.click=function(){ok(this.download==='m8j5_review_responses.json','download_filename');};$('export').click();ok(blobMade,'actual_export_blob');
 current=1;render();$('notes').value='edited';$('notes').dispatchEvent(new Event('input'));ok($('export').disabled&&responses.size===35,'editing_requires_new_decision');
 const result=document.createElement('pre');result.id='test-result';result.textContent=JSON.stringify({passed:true,checks,exported});document.body.append(result);
} catch(error){const result=document.createElement('pre');result.id='test-result';result.textContent=JSON.stringify({passed:false,error:String(error),checks});document.body.append(result);}
</script>'''
    fixture=out/'fixture.html';fixture.write_text(page.replace('</body>',harness+'</body>'))
    cmd=['unshare','--net','/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking',
         '--no-first-run','--no-default-browser-check','--user-data-dir='+str(out/'chrome'),
         '--dump-dom','--virtual-time-budget=1000',fixture.as_uri()]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
    (out/'chrome_stderr.log').write_text(p.stderr)
    match=re.search(r'<pre id="test-result">(.*?)</pre>',p.stdout,re.S)
    if not match:raise RuntimeError('No browser test output; '+p.stderr[-1000:])
    result=json.loads(html.unescape(match[1]));assert result['passed'],result
    exported=result.pop('exported')
    for c,e in zip(rows,exported):validate_signoff(c,e)
    result.update(evidence='ActualHeadlessChrome_36SyntheticCandidates_NoHumanReviews',
        export_schema_validated=36,actual_golden_created=0,network='DisabledByNamespace',
        chrome_exit=p.returncode,checks_count=len(result['checks']),private_fixture=str(fixture))
    with Path('reports/m8j5_review_ui_validation.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
