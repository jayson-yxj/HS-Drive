"""Run the real frozen inference under a data-file read audit, then evaluate."""
import json,runpy,sys,time
from pathlib import Path
import numpy as np
from tools.m7a_common import read,write,asset,provenance


def main():
    manifest=read('reports/m7a_shadow_inference_manifest.json')
    allowed={str(Path(manifest[k]['checkpoint']).resolve()) for k in ['mobile','direct']}
    for r in manifest['packets']:
        p=read(r['path']);allowed.update(str(Path(c['path']).resolve()) for c in p['cameras'])
    opened=[]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=str(Path(args[0]).resolve())
        sensitive=('/hs-drive-data/' in p or '/runs/m5o_' in p or '/runs/m5i_' in p)
        if sensitive:
            opened.append(p)
            if p not in allowed:raise PermissionError('GT_or_unapproved_data_read:'+p)
    # Hook applies only to this process and exits before evaluator GT reads.
    sys.addaudithook(audit)
    sys.argv=['m7a_shadow_infer','--manifest','reports/m7a_shadow_inference_manifest.json',
      '--output','reports/m7a_shadow_rgb_forward_audited.json']
    runpy.run_module('tools.m7a_shadow_infer',run_name='__main__')
    write('reports/m7a_execution_logs/inference_file_access.json',dict(kind='ActualFrozenGPUForwardWithDataReadAllowlist',
      allowed_data_files=sorted(allowed),opened_sensitive_data_files=opened,GT_data_access=False,
      limits='Python open audit in this process; not OS sandbox or authentication; imports/framework resources unrestricted',provenance=provenance()))

if __name__=='__main__':main()
