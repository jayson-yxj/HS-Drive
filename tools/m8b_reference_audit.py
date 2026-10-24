"""Freeze every evaluator reference actually used, without modifying historical data."""
import json
from pathlib import Path
from tools.m8b_common import *
def main():
    index=json.loads((ROOT/'evaluation_index.json').read_text());assets={};pixel_checks=[]
    for x in index:
        r=x['row'];pixel=asset(r['pixel_label']);assert pixel['sha256']==r['pixel_sha256'];pixel_checks.append(dict(token=x['token'],pixel_SHA_matches=True))
        rawpath=Path(r['raw_frame']);raw=json.loads(rawpath.read_text());paths=[Path(r['input']),rawpath,Path(r['pixel_label'])]
        for c in raw['cameras'].values():paths.append(rawpath.parent.parent/c['instance'])
        for p in paths:
            if str(p) not in assets:assets[str(p)]=asset(p)
    for split in ['train','val','calibration','sealed']:
        p=Path('runs/m5o_cache_'+split+'/supervision.json');assets[str(p)]=asset(p)
        if split!='train':
            p=Path('runs/m5o_eval_'+split+'/predictions.json');assets[str(p)]=asset(p)
    write('reports/m8b_evaluation_reference_audit.json',dict(**provenance('OfflineReadOnlyReferenceHashAudit'),files=list(assets.values()),pixel_checks=pixel_checks,reference='Historical CARLA rendered instances and depth for visible support; Actor OBB center/dimensions/yaw for localization. Not independent physical mesh.',future_truth_inference=False))
if __name__=='__main__':main()
