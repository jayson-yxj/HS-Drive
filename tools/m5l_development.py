"""Sequential remaining GPU study before test freeze."""
import subprocess

def run(args,log,python='.venv-gpu/bin/python'):
 with open('runs/m5l_logs/'+log,'x')as f:subprocess.run([python,'-m',*args],stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
 for split in ['train','regression']:
  run(['tools.m5l_object_eval','--split',split],'object_'+split+'.log')
 run(['tools.m5l_vehicle_eval','--split','regression'],'vehicle_eval_regression.log')
 for split in ['val','regression']:run(['tools.m5l_paths','--split',split],'paths_'+split+'.log')
 run(['tools.m5l_adaptive','--split','regression'],'adaptive_regression.log')
if __name__=='__main__':main()
