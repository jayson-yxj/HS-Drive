"""Wait for complete capture, then immutable adaptation and support evaluation."""
import subprocess,time,json
from pathlib import Path

def run(args,log,python='.venv-gpu/bin/python'):
 with open('runs/m5k_logs/'+log,'x')as f:subprocess.run([python,'-m',*args],stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
 protocol=json.loads(Path('configs/m5k_capture_dev.json').read_text());root=Path('/root/gpufree-data/hs-drive-data/raw/m5k_dev');deadline=time.monotonic()+1800
 while not all((root/g['group']/'manifest.json').exists()for g in protocol['groups']):
  if time.monotonic()>deadline:raise TimeoutError('Capture not completed')
  time.sleep(5)
 run(['tools.m5a_dataset','--raw',str(root),'--output','/root/gpufree-data/hs-drive-data/processed/m5k_dev_index','--report','runs/m5k_logs/v2_dev_quality.json'],'v2_dev.log','.venv/bin/python')
 run(['tools.m5g_dataset','--indices','/root/gpufree-data/hs-drive-data/processed/m5k_dev_index','--output','/root/gpufree-data/hs-drive-data/processed/m5k_dev_pixels','--splits','train','val'],'pixels_dev.log','.venv/bin/python')
 for split in ['train','val']:
  run(['tools.m5k_support','--split',split],'support_'+split+'.log')
  run(['tools.m5k_alignment','--split',split],'alignment_'+split+'.log')
 run(['tools.m5k_decision'],'decision.log')
 if json.loads(Path('reports/m5k_training_decision.json').read_text())['run_residual']:run(['tools.m5k_train'],'residual_train.log')
 run(['tools.m5k_sampling','--split','val'],'sampling_val.log')
if __name__=='__main__':main()
