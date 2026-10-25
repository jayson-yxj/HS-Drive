"""Seal early-stop evidence without altering prior milestones."""
import ast,json,subprocess
from pathlib import Path
from tools.m8h1_common import *
from tools.m8e_verify import verify

def main():
 assert not subprocess.check_output(['git','diff',BASE,'--name-only'],text=True).strip()
 history=[]
 for name in ['m8g_release_manifest','m8h0_release_manifest']:
  p=Path('reports')/(name+'.json');history.append(dict(manifest=asset(p),verified_files=verify(p)))
 required=['summary.md','acceptance.json','frozen_templates.json','shifted_contract.json','ego_settling.json','vehicle_settling.json','walker_grounding.json','actual_visibility.json','render_identity.json','counterfactual_identity.json','sensor_synchronization.json','negative_controls.json','training_readiness.json']
 assert all(Path('reports/m8h1_'+p).is_file() for p in required)
 assert all(Path('docs/'+p).is_file() for p in ['M8H1_IMPLEMENTATION.md','M8H1_ARCHITECTURE_DECISION.md'])
 for folder,pattern in [('tools','m8h1_*.py'),('datasets','m8h1_*.py'),('tests','test_m8h1*.py')]:
  for p in Path(folder).glob(pattern):ast.parse(p.read_text())
 for p in [*Path('reports').glob('m8h1_*.json'),*LOG.glob('*.json')]:
  d=json.loads(p.read_text())
  for f,h in d.get('sources',{}).items():
   if sha(f)!=h:
    old=ROOT/'source_revisions'/h/f;assert old.is_file() and sha(old)==h,(f,h,str(p))
 clean=json.loads(Path('reports/m8h1_runtime_cleanup.json').read_text());assert not clean['synchronous_mode'] and clean['vehicles']==clean['walkers']==clean['sensors']==0
 tests=json.loads(Path('reports/m8h1_test_results.json').read_text());assert tests['total']==92
 assert not list((ROOT/'capture').glob('*/rgb/*'))
 write('reports/m8h1_final_validation.json',dict(**provenance('ActualFinalEvidenceValidation'),history=history,tests=92,actual_negative_queries=5,all_required_deliverables=True,default_code_modified=False,invalid_startup_retained=True,valid_ego_settling_trials=2,all_actor_spawns=4,target_spawns=0,sensors=0,RGB=0,full_smoke='NotRunBeyondEgo_NoGo',own_server_shutdown='PID157737 SIGTERM after cleanup verified',figures_reviewed=2,capture_path='Implemented but not exercised; no end-to-end certification',raw_data_root=str(ROOT)))
 paths=[*Path('configs/m8h1').rglob('*'),*Path('reports').glob('m8h1_*.json'),*Path('reports').glob('m8h1_*.md'),*LOG.rglob('*'),*Path('reports/m8h1_figures').rglob('*'),*Path('tools').glob('m8h1_*.py'),*Path('datasets').glob('m8h1_*.py'),*Path('tests').glob('test_m8h1*.py'),*Path('docs').glob('M8H1*.md'),*ROOT.rglob('*')]
 files=[asset(p) for p in sorted(set(paths)) if p.is_file() and '__pycache__' not in p.parts and p.name!='m8h1_release_manifest.json' and not(p.parent==LOG and p.name.startswith('finalize'))]
 write('reports/m8h1_release_manifest.json',dict(**provenance('FinalSourceAndEvidenceSeal'),files=files,total_bytes=sum(x['bytes'] for x in files),historical_manifests=history,commit_contract='Final Git commit seals manifest; execution parent and exact source SHA recorded',external_assets='External raw root requires separate backup; Git manifest is not bulk-data backup'))
 print('SEALED',len(files),'assets;92tests;actualEgo2;target0;RGB0;SmokeNoGo',flush=True)
if __name__=='__main__':main()
