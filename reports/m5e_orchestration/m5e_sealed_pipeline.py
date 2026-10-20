from pathlib import Path
import time,subprocess
while not Path('reports/m5e_pretest_decision.json').exists():time.sleep(3)
commands=[('capture_sealed','.venv/bin/python',['tools.m5e_capture','--output','/root/gpufree-data/hs-drive-data/raw/m5e_sealed','--groups','Town03_e3','Town04_e3','Town05_e3']),('dataset_test','.venv-gpu/bin/python',['tools.m5e_dataset','--raw','/root/gpufree-data/hs-drive-data/raw/m5e_sealed','--output','/root/gpufree-data/hs-drive-data/processed/m5e_test_index','--report','reports/m5e_test_generation.json','--splits','test'])]
for name,python,args in commands:
 with open(f'reports/m5e_logs/m5e_{name}.log','w') as log:subprocess.run([python,'-m',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
procs=[]
for name in ['surface_index','supervision']:
 log=open(f'reports/m5e_logs/m5e_test_{name}.log','w');procs.append((subprocess.Popen(['.venv-gpu/bin/python','-m','tools.m5e_'+name,'--index','/root/gpufree-data/hs-drive-data/processed/m5e_test_index'],stdout=log,stderr=subprocess.STDOUT),log))
for p,l in procs:
 if p.wait():raise RuntimeError('testsidecar failure')
 l.close()
for name,args in [('test_evaluation',['tools.m5e_evaluate','--index','/root/gpufree-data/hs-drive-data/processed/m5e_test_index','--split','test']),('final_quality',['tools.m5e_quality','--indices','/root/gpufree-data/hs-drive-data/processed/m5e_index','/root/gpufree-data/hs-drive-data/processed/m5e_test_index']),('consolidate',['tools.m5e_consolidate']),('visualize',['tools.m5e_visualize'])]:
 with open(f'reports/m5e_logs/m5e_{name}.log','w') as log:subprocess.run(['.venv-gpu/bin/python','-m',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
