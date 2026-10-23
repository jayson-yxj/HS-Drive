"""Clean sequential paired runs; no future evaluator runs during timings."""
import argparse,subprocess,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',default='dev_constant_r3');a=p.parse_args()
    env={**os.environ,'PYTHONPATH':'runs/m7a_geometry_deps:.'}
    for mode in ['baseline','cold','stream']:
        # Redirect only append-only output; imported planning implementation frozen.
        script='from tools import m7d_replay as r\nfrom tools.m7d_common import write\nr.write=lambda p,d:write(p.replace("_'+mode+'.json","_'+mode+'_clean.json"),d)\nr.run('+repr(a.episode)+','+repr(mode)+')\n'
        with Path(f'reports/m7d_execution_logs/paired_{a.episode}_{mode}_clean.txt').open('x') as f:
            subprocess.run(['.venv/bin/python','-c',script],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(mode,'done',flush=True)
if __name__=='__main__':main()
