"""Two predeclared evidence-motivated actual rollback protection attempts."""
import json
from pathlib import Path
from tools.m6d_probe import run
from tools.m6d_common import write,provenance,asset

def main():
    plan=json.loads(Path('configs/m6d/guarded_warm_followup.json').read_text());protocol=json.loads(Path('configs/m6d/protocol.json').read_text())
    write('reports/m6d_guard_probe_plan.json',dict(**provenance('predeclared_unchanged_guard_followup'),plan=plan))
    records=[]
    for task in plan['tasks']:
        p=run(task,protocol);records.append(asset(p));print(p,flush=True)
    write('reports/m6d_guard_probe_execution.json',dict(**provenance('actual_CARLA_guard_probe'),results=records))
if __name__=='__main__':main()
