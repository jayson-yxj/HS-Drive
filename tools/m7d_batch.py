"""Sequential subprocess launcher: isolates future evaluator and clean timings."""
import argparse,subprocess,os
from pathlib import Path
from tools.m7d_common import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--mode',default='stream');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    for manifest in sorted(Path('reports/m7d_inputs').glob('*/manifest.json')):
        m=read(manifest)
        if m['split']!=a.split:continue
        episode=m['episode_id'];tool='tools.m7d_evaluate' if a.evaluate else 'tools.m7d_replay'
        cmd=['.venv/bin/python','-m',tool,'--episode',episode]
        if not a.evaluate:cmd+=['--mode',a.mode]
        env={**os.environ,'PYTHONPATH':'runs/m7a_geometry_deps:.'}
        with Path(f'reports/m7d_execution_logs/batch_{episode}_{"eval" if a.evaluate else a.mode}.txt').open('x') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(episode,'complete',flush=True)
if __name__=='__main__':main()
