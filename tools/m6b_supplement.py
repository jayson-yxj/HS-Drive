"""Append-only retries of fixture initialization and bypass fault injections.

No controller/planner tuning. Preserves failed original attempt directories.
"""
import subprocess
from pathlib import Path
from tools.m6b_common import write,provenance


def main():
    tasks=[]
    for spawn in [126,170]:
        tasks.append(dict(name=f'final_blocked_p{spawn}_s101_fixture_retry',scene='blocked',spawn=spawn,ticks=1800))
    for spawn,fault in [(126,'timeout'),(170,'crash')]:
        tasks.append(dict(name=f'final_bypass_{fault}_p{spawn}',scene='bypass',spawn=spawn,ticks=600,fault=fault,fault_tick=350))
    for fault in ['emergency','invalid']:
        tasks.append(dict(name=f'final_moving_{fault}',scene='stop',spawn=27,ticks=600,fault=fault,fault_tick=350))
    write('reports/m6b_supplement_plan.json',dict(**provenance('predeclared_fixture_retry_and_fault_injection'),tasks=tasks,
        rationale='Retry only snapshot initialization failures; add mid-bypass faults. Frozen controller/Safety and thresholds.'))
    records=[]
    for task in tasks:
        cmd=['.venv/bin/python','-m','tools.m6b_drive','--arm','C','--length','100','--speed','2.5','--seed','101']
        for k,v in task.items():cmd.extend(['--'+k.replace('_','-'),str(v)])
        path=Path('runs/m6b_logs')/(task['name']+'.log')
        with path.open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        records.append(dict(command=cmd,exit_code=p.returncode,log=str(path)))
        print(task['name'],p.returncode,path.read_text()[-400:],flush=True)
    write('reports/m6b_supplement_execution.json',dict(**provenance('actual_CARLA_supplement'),runs=records))

if __name__=='__main__':main()
