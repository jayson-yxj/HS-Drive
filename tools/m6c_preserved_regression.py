"""Frozen C bypass/recovery regressions separate from failing D2 candidate."""
import subprocess
from pathlib import Path
from tools.m6c_common import write,provenance


def main():
    tasks=[]
    for spawn in [126,170]:
        for seed in [101,102]:tasks.append(dict(name=f'final_bypass_p{spawn}_s{seed}_C',scene='bypass',spawn=spawn,seed=seed))
        tasks.append(dict(name=f'final_removed_p{spawn}_C',scene='removed',spawn=spawn,seed=101))
    write('reports/m6c_preserved_regression_plan.json',dict(**provenance('frozen_C_capability_regression'),tasks=tasks,reason='Original4/4 bypass and removed recovery must be independently retained; not just D candidate run. No tuning.'))
    records=[]
    for t in tasks:
        cmd=['.venv/bin/python','-m','tools.m6c_drive','--arm','C','--length','100','--speed','2.5','--ticks','1800']
        for k,v in t.items():cmd.extend(['--'+k,str(v)])
        path=Path('runs/m6c_logs')/(t['name']+'.log')
        with path.open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        records.append(dict(task=t,command=cmd,exit_code=p.returncode,log=str(path)));print(t['name'],p.returncode,path.read_text()[-400:],flush=True)
    write('reports/m6c_preserved_regression_execution.json',dict(**provenance('actual_CARLA_frozen_C_regression'),runs=records))

if __name__=='__main__':main()
