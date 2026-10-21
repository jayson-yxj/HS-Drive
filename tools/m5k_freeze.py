"""Lock scientific code/configuration and selected models before new test capture."""
import json,subprocess
from pathlib import Path
from tools.m5k_common import write,asset
from simulator.recorder import file_hash

def main():
 assert Path('reports/m5k_sampling_val.json').exists()
 decision=json.loads(Path('reports/m5k_training_decision.json').read_text())
 if decision['run_residual']:assert Path('reports/m5k_residual_val.json').exists()
 paths=[p for root in ['tools','models','training','simulator','configs']for p in Path(root).glob('*m5k*')if p.is_file()]
 checkpoints=json.loads(Path('configs/m5j_models.json').read_text());write('reports/m5k_pretest_freeze.json',dict(kind='before_new_sealed_capture',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(p):file_hash(p)for p in paths},protocol=asset('configs/m5k_protocol.json'),models=checkpoints,residual=asset('reports/m5k_depth_training.json')if decision['run_residual']else None,decisions=decision,threshold=.2,limits='No new testRGB previously read; test once frozen pipeline. Supplementary postfreeze reporting permitted but no model/mask/threshold changes.'))
if __name__=='__main__':main()
