"""Sealed capture and evaluation once; historical M5K assets untouched."""
import json
from pathlib import Path
from tools.m5l_development import run
from simulator.recorder import file_hash

def main():
 freeze=json.loads(Path('reports/m5l_pretest_freeze.json').read_text())
 for p,h in freeze['sources'].items():assert file_hash(p)==h,p
 run(['tools.m5f_capture','--protocol','configs/m5l_capture_test.json','--output','/root/gpufree-data/hs-drive-data/raw/m5l_test'],'capture_test.log','.venv/bin/python')
 run(['tools.m5a_dataset','--raw','/root/gpufree-data/hs-drive-data/raw/m5l_test','--output','/root/gpufree-data/hs-drive-data/processed/m5l_test_index','--report','runs/m5l_logs/v2_test_quality.json'],'v2_test.log','.venv/bin/python')
 run(['tools.m5g_dataset','--indices','/root/gpufree-data/hs-drive-data/processed/m5l_test_index','--output','/root/gpufree-data/hs-drive-data/processed/m5l_test_pixels','--splits','test'],'pixels_test.log','.venv/bin/python')
 for name in ['baseline','vehicle_eval','object_eval','paths','adaptive']:run(['tools.m5l_'+name,'--split','test'],name+'_test.log')
if __name__=='__main__':main()
