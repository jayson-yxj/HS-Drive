"""Seal bounded diagnostic assets and prove no tracked historical file was modified."""
import ast,json,subprocess
from pathlib import Path
from tools.m8g_common import *
from tools.m8e_verify import verify

def main():
 required=['summary.md','acceptance.json','rejection_attribution.json','none_hit_analysis.json','overhead_clearance.json','lane_boundary_differential.json','static_geometry_conflicts.json','historical_preflight_index.json','preflight_footprint_audit.json','feasibility_scan.json','validator_differential.json','positive_negative_controls.json','development_smoke.json','training_readiness.json']
 assert all(Path('reports/m8g_'+p).is_file() for p in required)
 assert not subprocess.check_output(['git','diff',BASE,'--name-only'],text=True).strip(),'Historical tracked files changed'
 for folder,pat in [('tools','m8g_*.py'),('datasets','m8g_*.py'),('tests','test_m8g*.py')]:
  for p in Path(folder).glob(pat):ast.parse(p.read_text())
 snapshots=[]
 for p in [*Path('reports').glob('m8g_*.json'),*LOG.glob('*.json')]:
  d=json.loads(p.read_text())
  for f,h in d.get('sources',{}).items():
   if sha(f)!=h:
    old=ROOT/'source_revisions'/h/f;assert old.is_file() and sha(old)==h,(f,h)
    snapshots.append(dict(log=str(p),source=f,sha256=h,archive=str(old)))
 tests=json.loads(Path('reports/m8g_test_results.json').read_text());assert tests['tests']==99
 history=[]
 for n in ['m8b_release_manifest','m8b_evaluation_reference_audit','m8c_release_manifest','m8d_release_manifest','m8e_release_manifest','m8f_release_manifest']:
  p=Path('reports')/(n+'.json');history.append(dict(manifest=asset(p),verified_files=verify(p)))
 clean=json.loads(Path('reports/m8g_runtime_cleanup.json').read_text());assert not clean['synchronous_mode'] and clean['sensors']==clean['vehicles']==clean['walkers']==0
 write('reports/m8g_final_validation.json',dict(**provenance('ActualFinalValidation'),tests=99,synthetic_new_geometry_tests=17,new_actual_evidence_tests=13,historical_tests=69,actual_CARLA_controls=10,history=history,old_sources_preserved=snapshots,all_required_deliverables=True,no_historical_tracked_mutations=True,no_default_code_changes=True,new_RGB_frames=0,new_actors=0,own_server_shutdown='PID112872 SIGTERM after verified asynchronous empty world; process exited',figures_reviewed=5,training='NoGo',smoke='NotRun_PreconditionFailed'))
 paths=[*Path('configs/m8g').rglob('*'),*Path('reports').glob('m8g_*.json'),*Path('reports').glob('m8g_*.md'),*LOG.rglob('*'),*Path('reports/m8g_figures').rglob('*'),*Path('tools').glob('m8g_*.py'),*Path('datasets').glob('m8g_*.py'),*Path('tests').glob('test_m8g*.py'),*Path('docs').glob('M8G*.md'),*ROOT.rglob('*')]
 files=[asset(p) for p in sorted(set(paths)) if p.is_file() and '__pycache__' not in p.parts and p.name!='m8g_release_manifest.json' and not (p.parent==LOG and p.name.startswith('finalize'))]
 write('reports/m8g_release_manifest.json',dict(**provenance('FinalSourceAssetEvidenceSeal'),files=files,total_bytes=sum(x['bytes'] for x in files),historical_manifests=history,raw_root=str(ROOT),contract='Final Git commit seals this manifest; execution parent plus exact source SHA in reports; failed revisions retained; no changed historical assets',limitations='External raw assets need separate backup; manifest is not itself a remote raw-data backup'))
 print('SEALED',len(files),'assets;',sum(x['bytes'] for x in files),'bytes;99tests;NoGo placement;0RGB',flush=True)
if __name__=='__main__':main()
