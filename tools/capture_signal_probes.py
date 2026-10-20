"""Capture small controlled RGB/rules diagnostics on an already running dedicated server."""
import argparse,subprocess,sys,json
from pathlib import Path
from simulator.recorder import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',default='/root/gpufree-data/hs-drive-data/raw')
    p.add_argument('--prefix',default='m4signal');p.add_argument('--report');a=p.parse_args();runs=[]
    report=Path(a.report or f'reports/{a.prefix}_capture_complete.json')
    if report.exists():raise FileExistsError(report)
    for town in ('Town01','Town04','Town02'):
        run=a.prefix+'_'+town.lower();runs.append(run)
        with Path('logs/'+run+'.log').open('x') as log:
            subprocess.run([sys.executable,'-m','tools.record_dataset','--capture-config','configs/data_capture.yaml',
                '--map',town,'--load-map','--signal-probe','--sun-altitude','50','--frames','41','--tick-wall-delay','.7',
                '--output',a.raw_root,'--run-id',run],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Captured',run,flush=True)
    atomic_json(report,{'runs':runs})

if __name__=='__main__':main()
