import time,subprocess
from pathlib import Path
while not Path('reports/m5e_evaluation_digest.json').exists():time.sleep(2)
for name,args in [('localization',['tools.m5e_localization'])]:
 with open(f'reports/m5e_logs/m5e_{name}.log','w') as log:subprocess.run(['.venv-gpu/bin/python','-m',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
while not Path('reports/m5e_runtime_faults.json').exists():time.sleep(2)
with open('reports/m5e_logs/m5e_runtime_analysis.log','w') as log:subprocess.run(['.venv-gpu/bin/python','-m','tools.m5e_runtime_analysis'],stdout=log,stderr=subprocess.STDOUT,check=True)
