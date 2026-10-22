"""Fixed evaluation budgets and append-only serial CARLA commands."""
import argparse,subprocess
from pathlib import Path
from tools.m6c_common import write,provenance


def tasks(kind):
    if kind=='stop':
        for seed in [101,102]:
            for spawn in [27,126,170]:
                for length,speed in [(9,1),(25,2.5)]:
                    arms=['C','D'] if seed==101 else ['D','C']
                    if length==9 and spawn in [27,170]:arms=['A']+arms
                    for arm in arms:yield dict(name=f'final_stop_p{spawn}_l{length}_s{seed}_{arm}',scene='stop',arm=arm,spawn=spawn,length=length,speed=speed,seed=seed,ticks=1000)
    elif kind=='maneuver':
        for seed in [101,102]:
            for spawn in [126,170]:
                for speed in [1,2.5]:
                    for arm in ['C','D'] if seed==101 else ['D','C']:
                        yield dict(name=f'final_blocked_p{spawn}_v{speed}_s{seed}_{arm}',scene='blocked',arm=arm,spawn=spawn,length=100,speed=speed,seed=seed,ticks=1400)
                yield dict(name=f'final_bypass_p{spawn}_s{seed}_D',scene='bypass',arm='D',spawn=spawn,length=100,speed=2.5,seed=seed,ticks=1800)
                # Exactly one sudden-appearance velocity per location/repeat;
                # severe margin shortfall is retained, never dropped as outlier.
                for arm in ['C','D']:
                    yield dict(name=f'final_sudden_p{spawn}_s{seed}_{arm}',scene='sudden',arm=arm,spawn=spawn,length=100,speed=2.5,seed=seed,ticks=500)
            for scene in ['straight','mild_curve','junction_turn','lead']:
                for arm in ['C','D']:
                    yield dict(name=f'final_{scene}_s{seed}_{arm}',scene=scene,arm=arm,spawn=27,length=60,speed=2.5,seed=seed,ticks=1400)
        for spawn in [126,170]:
            yield dict(name=f'final_removed_p{spawn}_D',scene='removed',arm='D',spawn=spawn,length=100,speed=2.5,seed=101,ticks=1800)
        for fault in ['emergency','crash','timeout','invalid']:
            yield dict(name=f'final_fault_{fault}_D',scene='stop',arm='D',spawn=170,length=100,speed=2.5,seed=101,ticks=550,fault=fault,fault_tick=300)
    elif kind=='runtime':
        for seed in [101,102]:
            for arm in ['C','D'] if seed==101 else ['D','C']:
                yield dict(name=f'runtime_driving_{arm}_s{seed}',scene='straight',arm=arm,spawn=27,length=60,speed=2.5,seed=seed,ticks=600,fixed_budget=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['stop','maneuver','runtime']);a=p.parse_args();plan=list(tasks(a.kind))
    write(f'reports/m6c_{a.kind}_plan.json',dict(**provenance('predeclared_final_CARLA_matrix'),tasks=plan,E='NotRun_no_comfort_support_from_startup_probes; no execution-policy_search'))
    records=[]
    for task in plan:
        cmd=['.venv/bin/python','-m','tools.m6c_drive']
        for k,v in task.items():
            cmd.append('--'+k.replace('_','-'))
            if not isinstance(v,bool):cmd.append(str(v))
        logfile=Path('runs/m6c_logs')/(task['name']+'.log')
        with logfile.open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        records.append(dict(task=task,command=cmd,exit_code=r.returncode,log=str(logfile)))
        print(task['name'],r.returncode,logfile.read_text()[-500:],flush=True)
    write(f'reports/m6c_{a.kind}_execution.json',dict(**provenance('actual_CARLA_serial_matrix'),runs=records))

if __name__=='__main__':main()
