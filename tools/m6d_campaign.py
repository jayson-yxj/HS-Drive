"""Finite development-only F0 interface/safety comparison, never F1 qualification."""
import json,subprocess
from pathlib import Path
from tools.m6d_common import write,provenance,asset


def main():
    protocol=json.loads(Path('configs/m6d/protocol.json').read_text());p=protocol['f0_development'];tasks=[]
    for pos in p['stop_positions']:
        for seed in p['seeds']:
            for arm in p['arms']:
                tasks.append(dict(name=f'dev_stop_p{pos}_s{seed}_{arm}',arm=arm,spawn=pos,seed=seed,scene='stop',length=p['length'],speed=p['speed'],ticks=1000))
    f=protocol['f0_fault_diagnostics']
    for arm in ['C','F0']:
        for fault in f['faults']:
            tasks.append(dict(name=f'dev_fault_{fault}_{arm}',arm=arm,spawn=f['position'],seed=201,scene='straight',length=60,speed=f['speed'],ticks=f['ticks'],fault=fault,fault_tick=65))
    write('reports/m6d_development_matrix_plan.json',dict(**provenance('predeclared_limited_development_only'),tasks=tasks,
        limit='12 matched short-route trials plus paired fault checks. Not the gated F1 25m/blocked/bypass/sealed matrix.'))
    results=[]
    for i,task in enumerate(tasks):
        cmd=['.venv/bin/python','-m','tools.m6d_drive']
        for k,v in task.items():cmd+=['--'+k.replace('_','-'),str(v)]
        log=Path('reports/m6d_execution_logs')/(task['name']+'.log.txt')
        with log.open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        results.append(dict(task=task,command=cmd,exit_code=r.returncode,log=asset(log)))
        print(i+1,len(tasks),task['name'],r.returncode,flush=True)
        if r.returncode:break
    write('reports/m6d_development_matrix_execution.json',dict(**provenance('actual_CARLA_development_commands'),runs=results))
if __name__=='__main__':main()
