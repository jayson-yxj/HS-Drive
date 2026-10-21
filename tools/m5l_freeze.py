"""Pretest commit of methods, validation workpoints and conditional research decisions."""
import json,subprocess
from pathlib import Path
from tools.m5l_common import write,asset
from simulator.recorder import file_hash

def main():
 required=['sampling_failure_analysis','p4_feature_quality','residual_val','residual_regression','objects_train','objects_val','objects_regression','paths_val','paths_regression','adaptive_val','adaptive_regression','adaptive_lock']
 for name in required:assert Path('reports/m5l_'+name+'.json').exists(),name
 protocol=json.loads(Path('configs/m5l_protocol.json').read_text());assets={r['selected']['path']:r['selected']['sha256']for r in json.loads(Path('reports/m5l_vehicle_training.json').read_text())['records']};cfg=json.loads(Path('configs/m5j_models.json').read_text());assets.update({r['checkpoint']:r['sha256']for r in [cfg['depth'],*cfg['pixel']]});paths=[p for root in ['models','diagnostics','simulator','tools','training','configs']for p in Path(root).glob('*m5l*')if p.is_file()]
 write('reports/m5l_pretest_freeze.json',dict(kind='before_new_testcapture',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):file_hash(p)for p in paths},models=assets,protocol=asset('configs/m5l_protocol.json'),validation_workpoints=asset('reports/m5l_adaptive_lock.json'),p4_training=False,limits='All312 existingframes diagnosed before newcapture; oldTest regressiononly. Selectedcheckpoint/thresholds fromVal only, no newtesttuning.'))
if __name__=='__main__':main()
