import time,subprocess,json,shutil
from pathlib import Path
while not Path('reports/m5e_runtime_attribution.json').exists():time.sleep(2)
commands=[('supervision_isolation','.venv-gpu/bin/python',['-m','tools.m5e_data_analysis','--indices','/root/gpufree-data/hs-drive-data/processed/m5e_index','/root/gpufree-data/hs-drive-data/processed/m5e_test_index','--report','reports/m5e_supervision_isolation.json']),('scene_figure','.venv-gpu/bin/python',['-m','tools.m5e_scene_figure']),('final_runtime_tests','.venv-gpu/bin/python',['-m','pytest','-q','tests/test_m5e_runtime.py','--junitxml=reports/m5e_runtime_unit_tests.xml'])]
for name,py,args in commands:
 with open(f'reports/m5e_logs/m5e_{name}.log','w') as log:subprocess.run([py,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
for index,name in [('m5e_index','trainval'),('m5e_test_index','test')]:
 root=Path('/root/gpufree-data/hs-drive-data/processed',index)
 for file in ['index.json','manifest.json','m5e_pairs.json','m5e_supervision.json','m5e_surface_index.json']:
  dest=Path('reports/m5e_indices')/f'{name}_{file}';dest.parent.mkdir(exist_ok=True)
  if dest.exists():raise FileExistsError(dest)
  shutil.copyfile(root/file,dest)
Path('reports/m5e_checks_completed.json').write_text(json.dumps({'kind':'actual_local_command_completion','commands':commands,'exit_codes':[0]*len(commands)},indent=2)+'\n')
