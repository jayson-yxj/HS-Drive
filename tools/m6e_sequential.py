"""Sequential single tick-owner batches; no background CARLA clients."""
import json,subprocess
from pathlib import Path
from tools.m6e_common import write,asset,provenance

def main():
    records=[]
    commands=[('P0_development_analysis',['.venv/bin/python','-m','tools.m6e_analyze','development']),
              ('P0_validation',['.venv/bin/python','-m','tools.m6e_probe','validation']),
              ('P0_validation_analysis',['.venv/bin/python','-m','tools.m6e_analyze','validation']),
              ('guard_live',['.venv/bin/python','-m','tools.m6e_probe','guard']),
              ('guard_live_analysis',['.venv/bin/python','-m','tools.m6e_analyze','guard'])]
    for name,cmd in commands:
        log=Path('reports/m6e_execution_logs')/(name+'.log.txt')
        with log.open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        records.append(dict(command=cmd,exit_code=r.returncode,log=asset(log)));print(name,r.returncode,flush=True)
        if r.returncode:break
    write('reports/m6e_P0_execution.json',dict(**provenance('actual_single_tick_owner_sequence'),commands=records))
if __name__=='__main__':main()
