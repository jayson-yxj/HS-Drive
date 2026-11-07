"""Three explicitly authorized one-start faults. Journal never resumed/reset."""
import fcntl
import json
from pathlib import Path
import time
from tools.m8j13_common import digest,write
from tools.m8j13_logic_runtime import supervise
from tools.m8j13_logic_fault_report import audit
from simulator.m8j13_logic_faults import SCENARIOS,reserve_scenario


def main():
    paths=[Path('configs/m8j13_logic_fault_'+n+'.json') for n in SCENARIOS]
    cfgs=[json.loads(p.read_text()) for p in paths];base=Path(cfgs[0]['suite_private_root']);base.mkdir(parents=True,exist_ok=True)
    with (base/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (base/'suite_ledger.jsonl').open('x',buffering=1) as journal:
            records=[]
            def record(r):
                import os
                r['wall_s']=time.monotonic();journal.write(json.dumps(r)+'\n');journal.flush();os.fsync(journal.fileno());records.append(r)
            for name,path,cfg in zip(SCENARIOS,paths,cfgs):
                reservation=reserve_scenario(records,name,cfg['run_id'],digest(path));record(reservation)
                try:
                    root=supervise(path,reservation);summary=audit(root);write(root/'scenario_audit.json',summary)
                    passed=summary['status']=='Passed_FiniteInjectedFaultOnly';charged=summary['budget']['charged_ticks']
                    record(dict(event='finished',scenario=name,run_id=cfg['run_id'],passed=passed,charged_ticks=charged,first_fault=summary['first_fault'],failed_checks=summary['failed_checks']))
                except BaseException as exc:
                    record(dict(event='finished',scenario=name,run_id=cfg['run_id'],passed=False,charged_ticks=800,error=repr(exc)));raise
                if not passed:break
            write(base/'suite_summary.json',dict(records=records,starts=sum(r['event']=='reserved' for r in records),charged_ticks=sum(r.get('charged_ticks',0) for r in records),model_forwards=0,model_controls=0,pid_executions=0,training=0))

if __name__=='__main__':main()
