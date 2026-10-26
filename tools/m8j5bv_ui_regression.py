"""Repeat the original nine UI checks; additional actual pixel/crop equality."""
import ast
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np

from tools.m8j5bv_run import ROOT,PUBLIC


def main():
    old=Path('/root/gpufree-data/hs-drive-private/m8j5b-coverage-supervision')
    ROOT.mkdir(exist_ok=True,parents=True);PUBLIC.mkdir(exist_ok=True,parents=True)
    web=ROOT/'web';web.mkdir(exist_ok=False);(web/'images').symlink_to(old/'web/images',target_is_directory=True)
    page=(old/'web/index.html').read_text()
    # Execute exactly the old nine assertions without writing old results/page.
    tree=ast.parse(Path('tools/m8j5b_validate_ui.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='test' for t in n.targets))
    test=ast.literal_eval(assignment.value)
    target=web/'ui_test.html';target.write_text(page.replace('</body>',test+'</body>'))
    (web/'index.html').write_text(page)
    cmd=['unshare','--net','/usr/bin/google-chrome','--headless','--no-sandbox','--disable-gpu',
        '--disable-background-networking','--no-first-run','--user-data-dir='+str(ROOT/'chrome_ui'),
        '--dump-dom','--virtual-time-budget=1500',target.as_uri()]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
    (ROOT/'chrome_stderr.log').write_text(p.stderr)
    match=re.search(r'<pre id="test-result">(.*?)</pre>',p.stdout,re.S)
    assert match and p.returncode==0
    result=json.loads(html.unescape(match[1]));assert result['passed'] and len(result['checks'])==9
    payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',page,re.S)[1])
    checks=[]
    for frame in payload:
        n=frame['ordinal']
        with np.load(old/'masks_v2'/(n+'.npz'),allow_pickle=False) as z:train=z['ignore'].copy()
        for ci,(cam,paths) in enumerate(frame['images'].items()):
            image_path=old/'web'/paths['mask'];overlay=cv2.imread(str(image_path),-1)
            alpha=overlay[:,:,3]>0
            original=cv2.imread(str(old/'views'/(n+'_'+cam+'_ignore.png')),0)>0
            assert np.array_equal(alpha,original)
            crop=(cv2.resize(alpha.astype(np.float32),(1408,792),interpolation=cv2.INTER_AREA)>0)[280:792]
            assert np.array_equal(crop,train[ci])
            checks.append(dict(ordinal=n,camera=cam,mask_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                original_and_overlay_equal=True,display_crop_and_loss_mask_equal=True))
    result.update(evidence='ActualPrivatePage_BrowserAndPixelArrays',network='Disabled',old_page_sha256=hashlib.sha256(page.encode()).hexdigest(),
        pixel_checks=checks,pixel_checks_passed=len(checks),human_signoff='Pending',signoffs_created=0)
    with (PUBLIC/'m8j5bv_ui_validation.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(browser_assertions=9,pixel_views=len(checks),passed=True)))


if __name__=='__main__':main()
