"""Conditional decisions made on Train/Validation only, never sealed outcomes."""
import json
from pathlib import Path
from tools.m5k_common import write,asset

def main():
 support={s:json.loads(Path(f'reports/m5k_support_{s}.json').read_text())['eligibility']for s in ['train','val']};alignment={}
 for split in ['historical_val','val']:
  data=json.loads(Path(f'reports/m5k_alignment_{split}.json').read_text());checks=[]
  for seed in [101,102]:
   a=next(r for r in data['bev']if r['method']=='original_p8'and r['seed']==seed);b=next(r for r in data['bev']if r['method']=='native16_to_p8'and r['seed']==seed);checks.append(dict(seed=seed,precision_gain=b['precision']-a['precision'],recall1_gain=b['recall1']-a['recall1'],fpr_delta=b['fpr']-a['fpr'],actor_delta=b['actor_detection']-a['actor_detection']))
  alignment[split]=dict(records=checks,clear_gain=all(r['precision_gain']>=.01 and r['recall1_gain']>=.05 and r['fpr_delta']<=.001 and r['actor_delta']>=0 for r in checks))
 improved=all(v['clear_gain']for v in alignment.values());eligible=all(v['passed']for v in support.values());write('reports/m5k_training_decision.json',dict(kind='pretest_train_validation_gate',support=support,alignment=alignment,run_residual=eligible and not improved,status='RunBounded600Steps'if eligible and not improved else'NoTraining_AlignmentPreferred'if improved else'NoTraining_SupportInsufficient',protocol=asset('configs/m5k_protocol.json'),limits='No test evaluated; support mask and minimum counts frozen before capture. Neither failed support nor MAE-only gain authorizes training.'))
if __name__=='__main__':main()
