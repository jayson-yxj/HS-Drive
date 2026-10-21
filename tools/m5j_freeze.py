"""No scientific changes after new test observation."""
import json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from tools.m5j_common import write,source
from simulator.recorder import file_hash

def verify():
    f=json.loads(Path('reports/m5j_pretest_freeze.json').read_text())
    for key in ['sources','reports','checkpoints']:
        for p,h in f[key].items():
            if file_hash(p)!=h:raise ValueError('Frozen asset changed '+p)
    return f

def main():
    cfg=json.loads(Path('configs/m5j_models.json').read_text());sel=json.loads(Path('reports/m5j_selection.json').read_text());
    if sel['conditional_D']:assert Path('reports/m5j_calibration_val.json').exists()
    ck={r['checkpoint']:r['sha256']for r in cfg['pixel']};ck[cfg['depth']['checkpoint']]=cfg['depth']['sha256'];reports={str(p):file_hash(p)for p in Path('reports').glob('m5j*.json')}
    write('reports/m5j_pretest_freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),sources=source(),reports=reports,checkpoints=ck,selection=sel,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),new_test_opened=False))
if __name__=='__main__':main()
