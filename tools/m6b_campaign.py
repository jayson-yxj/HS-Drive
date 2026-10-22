"""Fixed-budget, serial CARLA experiments: no shared tick owners or GPU training."""
import argparse,subprocess,json
from pathlib import Path
from tools.m6b_common import write,provenance


def tasks(kind):
    if kind=='stop':
        for seed in [101,102]:
            for spawn in [27,170]:
                for length,speed in [(9,1),(25,2.5)]:
                    # Reverse arm order on repeat; no claim seeds create new roads.
                    for arm in ('ABC' if seed==101 else 'CBA'):
                        yield dict(name=f'final_stop_p{spawn}_l{length}_s{seed}_{arm}',arm=arm,scene='stop',spawn=spawn,length=length,speed=speed,seed=seed,ticks=1500)
    elif kind=='maneuver':
        for seed in [101,102]:
            for spawn in [126,170]:
                for scene in ['bypass','blocked','removed']:
                    yield dict(name=f'final_{scene}_p{spawn}_s{seed}',arm='C',scene=scene,spawn=spawn,length=100,speed=2.5,seed=seed,ticks=2200)
            for scene in ['straight','mild_curve','junction_turn','lead','emergency','crash','timeout','invalid']:
                yield dict(name=f'final_{scene}_s{seed}',arm='C',scene=scene,spawn=27,length=60,speed=2.5,seed=seed,ticks=1600 if scene in ['straight','mild_curve','junction_turn','lead'] else 200)
    elif kind=='runtime':
        for repeat in [0,1]:
            names=['A','B','C','D'] if repeat==0 else ['D','C','B','A']
            for arm in names:
                yield dict(name=f'runtime_{arm}_r{repeat}',arm='tick_only' if arm=='A' else 'A',scene='straight',speed=2.5,ticks=600,
                    **({'tick_cpus':'2,3'} if arm in ['C','D'] else {}),
                    **({'worker_cpus':'4,5'} if arm=='D' else {}),fixed_budget=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['stop','maneuver','runtime']);a=p.parse_args()
    plan=list(tasks(a.kind));write(f'reports/m6b_{a.kind}_campaign_plan.json',dict(**provenance('predeclared_CARLA_campaign'),tasks=plan))
    results=[]
    for task in plan:
        cmd=['.venv/bin/python','-m','tools.m6b_drive']
        for k,v in task.items():
            cmd.append('--'+k.replace('_','-'))
            if not isinstance(v,bool):cmd.append(str(v))
        path=Path('runs/m6b_logs')/(task['name']+'.log')
        with path.open('x') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(task=task,command=cmd,exit_code=proc.returncode,log=str(path)))
        print(task['name'],proc.returncode,path.read_text()[-400:],flush=True)
    write(f'reports/m6b_{a.kind}_campaign_execution.json',dict(**provenance('actual_serial_CARLA_commands'),runs=results))

if __name__=='__main__':main()
