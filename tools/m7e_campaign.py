"""Sequential isolated planner/evaluator launcher, current input roots explicit."""
import argparse,subprocess,os
from pathlib import Path
from tools.m7e_common import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',nargs='+',required=True);p.add_argument('--arms',nargs='+',default=['A_uncached','A','B']);p.add_argument('--suffix',default='');p.add_argument('--input-root',default='reports/m7d_inputs');p.add_argument('--evaluate',action='store_true');p.add_argument('--future-root',default='reports/m7d_evaluator');a=p.parse_args()
    env={**os.environ,'PYTHONPATH':'runs/m7a_geometry_deps:.'}
    for ep in a.episodes:
        if a.evaluate:
            cmd=['.venv/bin/python','-m','tools.m7e_evaluate','--episode',ep,'--input-root',a.input_root,'--future-root',a.future_root]
            with Path(f'reports/m7e_execution_logs/{ep}_eval.txt').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        else:
            for arm in a.arms:
                cmd=['.venv/bin/python','-m','tools.m7e_replay','--episode',ep,'--arm',arm,'--input-root',a.input_root,'--suffix',a.suffix]
                with Path(f'reports/m7e_execution_logs/{ep}_{arm}{a.suffix}.txt').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(ep,'done',flush=True)
if __name__=='__main__':main()
