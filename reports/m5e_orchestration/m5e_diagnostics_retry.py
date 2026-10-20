import subprocess
for name,args in [('depth_after_retry',['tools.m5e_depth_diagnostics','--models']),('stride_factorization_retry',['tools.m5e_stride_factorization']),('validation',['tools.m5e_evaluate','--index','/root/gpufree-data/hs-drive-data/processed/m5e_index','--split','val'])]:
 with open(f'reports/m5e_logs/m5e_{name}.log','w') as log:subprocess.run(['.venv-gpu/bin/python','-m',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
