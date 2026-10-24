"""Clean continuous-input pairing with counterbalanced arm order across episodes."""
import subprocess,os
from pathlib import Path

def main():
    for ep,arms in [('dev_constant_r3',['A_uncached','A','B']),('dev_braking_r20',['B','A','A_uncached'])]:
        for arm in arms:
            with Path(f'reports/m7e_execution_logs/{ep}_{arm}_clean.txt').open('x') as f:
                subprocess.run(['.venv/bin/python','-m','tools.m7e_replay','--episode',ep,'--arm',arm,'--suffix','_clean'],env={**os.environ,'PYTHONPATH':'runs/m7a_geometry_deps:.'},stdout=f,stderr=subprocess.STDOUT,check=True)
            print(ep,arm,'complete',flush=True)
if __name__=='__main__':main()
