"""Correct invalid stationary injection experiment, preserve original results."""
import json,subprocess
from pathlib import Path
from tools.m6d_common import write,provenance,asset

def main():
    cfg=json.loads(Path('configs/m6d/moving_fault_correction.json').read_text());write('reports/m6d_moving_fault_plan.json',dict(**provenance('predeclared_fault_validity_correction'),config=cfg));runs=[]
    for arm in cfg['arms']:
        for fault in cfg['faults']:
            name=f'dev_moving_fault_{fault}_{arm}';cmd=['.venv/bin/python','-m','tools.m6d_drive','--name',name,'--arm',arm,'--scene','straight','--spawn','170','--seed','201','--speed','2.5','--ticks','500','--fault',fault,'--fault-on-motion']
            p=Path('reports/m6d_execution_logs')/(name+'.log.txt')
            with p.open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            runs.append(dict(name=name,command=cmd,exit_code=r.returncode,log=asset(p)));print(name,r.returncode,flush=True)
            if r.returncode:break
    write('reports/m6d_moving_fault_execution.json',dict(**provenance('actual_CARLA_motion_triggered_faults'),runs=runs))
if __name__=='__main__':main()
