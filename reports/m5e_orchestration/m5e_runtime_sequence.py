import time,subprocess
from pathlib import Path
while not Path('reports/m5e_evaluation_digest.json').exists():time.sleep(3)
# Wait visual plots and evaluate completion before claiming isolated runtime.
while not Path('reports/m5e_figures/m5e_singleframe_maps.png').exists():time.sleep(3)
for name,args in [('runtime_campaign',['tools.m5e_runtime_campaign']),('runtime_faults',['tools.m5e_async_drive','--output','reports/m5e_runtime_faults.json','--artifacts','reports/m5e_runtime_faults','--scenarios','dynamic_lead','slow_worker','worker_crash','emergency','static_obstacle','stalled_worker','invalid_input','--load-description','isolated M5E fault and moving actor regression'])]:
 python='.venv/bin/python' if name=='runtime_faults' else '.venv-gpu/bin/python'
 with open(f'reports/m5e_logs/m5e_{name}.log','w') as log:subprocess.run([python,'-m',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
