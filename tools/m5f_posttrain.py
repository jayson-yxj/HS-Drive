"""Sequential scientific freeze before any new test images are collected."""
import json,subprocess,time
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def main():
    while not Path('reports/m5f_factorization_validation.json').exists():time.sleep(2)
    commands=[('bev_features',['tools.m5f_bev_features']),('fusion_training',['tools.m5f_fusion'])]
    for name,args in commands:
        with open(f'reports/m5f_logs/m5f_{name}.log','w') as log:subprocess.run(['.venv-gpu/bin/python','-m',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    checkpoints={str(p):file_hash(p) for d in ['runs/m5f_learning','runs/m5f_fusion'] for p in Path(d).glob('*.pt')}
    atomic_json('reports/m5f_pretest_decision.json',dict(checkpoints=checkpoints,protocol_sha256=file_hash('configs/m5f_protocol.json'),geometry_selection_sha256=file_hash('reports/m5f_geometry_selection.json'),sources={str(p):file_hash(p) for d in ['models','training','datasets','diagnostics','tools'] for p in Path(d).glob('*m5f*.py')},decision='All6models reported. No moretraining/thresholdtuning afternewtest. FPN remains auxiliarycopy, olddepthfrozen. Allvalidationjointconstraintsfailed; factorizationandshadow diagnostic, notformalquality.'))
    print('frozen',flush=True)

if __name__=='__main__':main()
