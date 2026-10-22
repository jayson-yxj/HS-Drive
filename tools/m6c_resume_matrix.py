"""Resume exact predeclared matrix; invalid D kept, corrected D suffix D2."""
import subprocess
from pathlib import Path
from tools.m6c_campaign import tasks
from tools.m6c_common import write,provenance


def main():
    plan=[]
    for kind in ['stop','maneuver','runtime']:
        for task in tasks(kind):
            if task['arm']=='D':task['name']+='2'
            if not (Path('reports/m6c_trials')/task['name']/'result.json').exists():plan.append(dict(kind=kind,task=task))
    write('reports/m6c_corrected_matrix_plan.json',dict(**provenance('predeclared_same_matrix_contract_fix_only'),tasks=plan))
    records=[];observer=None
    for item in plan:
        task=item['task'];kind=item['kind']
        if kind=='runtime' and observer is None:
            observer=subprocess.Popen(['.venv/bin/python','-m','tools.m6c_resource_monitor','--until','reports/m6c_corrected_matrix_execution.json','--output','runs/m6c_logs/resources.jsonl'])
        cmd=['.venv/bin/python','-m','tools.m6c_drive']
        for k,v in task.items():
            cmd.append('--'+k.replace('_','-'))
            if not isinstance(v,bool):cmd.append(str(v))
        logfile=Path('runs/m6c_logs')/(task['name']+'.log')
        with logfile.open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        records.append(dict(kind=kind,task=task,command=cmd,exit_code=p.returncode,log=str(logfile)))
        print(task['name'],p.returncode,logfile.read_text()[-450:],flush=True)
    write('reports/m6c_corrected_matrix_execution.json',dict(**provenance('actual_corrected_CARLA_matrix'),runs=records))
    if observer:observer.wait()
    write('reports/m6c_runtime_execution.json',dict(**provenance('actual_runtime_completion_marker'),runs=[v for v in records if v['kind']=='runtime']))

if __name__=='__main__':main()
