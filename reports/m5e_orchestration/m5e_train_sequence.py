import subprocess,time
from pathlib import Path
root='/root/gpufree-data/hs-drive-data/processed/m5e_index'
while not Path(root,'m5e_pairs.json').exists():time.sleep(2)
commands=[('pretrain_quality',['tools.m5e_quality','--indices',root,'--output','reports/m5e_pretrain_quality.json']),('learning',['tools.m5e_learning']),('depth_after',['tools.m5e_depth_diagnostics','--models']),('stride_factorization_retry',['tools.m5e_stride_factorization']),('validation',['tools.m5e_evaluate','--index',root,'--split','val'])]
for name,args in commands:
 with open(f'reports/m5e_logs/m5e_{name}.log','w') as log:subprocess.run(['.venv-gpu/bin/python','-m',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
