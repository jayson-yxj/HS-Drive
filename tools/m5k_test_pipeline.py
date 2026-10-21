"""One sealed evaluation; selected checkpoints and masks frozen first."""
import json
from pathlib import Path
from tools.m5k_dev_pipeline import run
from simulator.recorder import file_hash

def main():
 f=json.loads(Path('reports/m5k_pretest_freeze.json').read_text())
 for p,h in f['sources'].items():assert file_hash(p)==h,p
 run(['tools.m5f_capture','--protocol','configs/m5k_capture_test.json','--output','/root/gpufree-data/hs-drive-data/raw/m5k_test'],'capture_test.log','.venv/bin/python')
 run(['tools.m5a_dataset','--raw','/root/gpufree-data/hs-drive-data/raw/m5k_test','--output','/root/gpufree-data/hs-drive-data/processed/m5k_test_index','--report','runs/m5k_logs/v2_test_quality.json'],'v2_test.log','.venv/bin/python')
 run(['tools.m5g_dataset','--indices','/root/gpufree-data/hs-drive-data/processed/m5k_test_index','--output','/root/gpufree-data/hs-drive-data/processed/m5k_test_pixels','--splits','test'],'pixels_test.log','.venv/bin/python')
 for tool in ['support','alignment','sampling']:
  run(['tools.m5k_'+tool,'--split','test'],tool+'_test.log')
 if f['decisions']['run_residual']:run(['tools.m5k_residual_eval','--split','test'],'residual_test.log')
if __name__=='__main__':main()
