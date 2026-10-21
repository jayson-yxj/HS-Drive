"""Complete Val diagnostics, freeze, then open only new sealed scenes once."""
import json,time,subprocess
from pathlib import Path
from tools.m5k_dev_pipeline import run

def main():
 deadline=time.monotonic()+3600
 while not Path('reports/m5k_sampling_val.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Val not complete')
  time.sleep(5)
 if json.loads(Path('reports/m5k_training_decision.json').read_text())['run_residual']:run(['tools.m5k_residual_eval','--split','val'],'residual_val.log')
 run(['tools.m5k_details','--split','val'],'details_val.log')
 run(['tools.m5k_freeze'],'freeze.log')
 log=open('runs/m5k_logs/carla_test.log','x');proc=subprocess.Popen(['runuser','-u','ubuntu','--','bash','scripts/start_carla.sh','/root/gpufree-data/simulators/CARLA_0.9.15','2100','Epic'],stdout=log,stderr=subprocess.STDOUT)
 time.sleep(20)
 try:run(['tools.m5k_test_pipeline'],'test_pipeline.log')
 finally:proc.terminate()
if __name__=='__main__':main()
