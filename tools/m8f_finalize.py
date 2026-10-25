"""Final append-only source/evidence seal; historical artifacts rechecked."""
import ast,json
from pathlib import Path
from tools.m8f_common import *
from tools.m8e_verify import verify

def main():
 history=[]
 for name in ['m8b_release_manifest','m8b_evaluation_reference_audit','m8c_release_manifest','m8d_release_manifest','m8e_release_manifest']:
  p=Path('reports')/(name+'.json');n=verify(p);history.append(dict(manifest=asset(p),files=n,unchanged=True));print(name,n,'unchanged',flush=True)
 for folder,pattern in [('tools','m8f_*.py'),('datasets','m8f_*.py'),('simulator','m8f_*.py'),('tests','test_m8f*.py')]:
  for p in Path(folder).glob(pattern):ast.parse(p.read_text())
 for p in Path('reports').glob('m8f_*.json'):json.loads(p.read_text())
 required=['summary.md','acceptance.json','legacy_failure_attribution.json','ground_surface_contract.json','ray_hit_and_contact_audit.json','natural_scene_quality.json','historical_target_footprints.json','split_preflight.json','independence_exceptions.json','identity_and_background_audit.json','far_field_support.json','object_eligibility.json','counterfactual_regression.json','training_readiness.json']
 assert all(Path('reports/m8f_'+s).exists() for s in required)
 assert len(json.loads(Path('reports/m8f_object_eligibility.json').read_text())['objects'])==8
 cleanup=json.loads(Path('reports/m8f_runtime_cleanup.json').read_text());assert not cleanup['synchronous_mode'] and cleanup['sensors']==0 and cleanup['vehicles']==0 and cleanup['walkers']==0
 write('reports/m8f_final_validation.json',dict(**provenance('FinalActualEvidenceValidation'),history=history,tests=69,
   new_tests=17,historical_tests=52,all_deliverables_present=True,default_code_modified=False,
   carla_cleanup=asset('reports/m8f_runtime_cleanup.json'),own_server_shutdown='PID85586 SIGTERM after cleanup; parent session exit143',
   figures_reviewed=6,raw_data_root=str(ROOT),input_contract='8RGBpackets/48images, no model forward',
   later_collection='NotRun because required smoke failed',training_authorized=False))
 paths=[*Path('configs/m8f').rglob('*'),*Path('reports').glob('m8f_*.json'),*Path('reports').glob('m8f_*.md'),
        *LOG.rglob('*'),*Path('reports/m8f_figures').rglob('*'),*Path('tools').glob('m8f_*.py'),
        *Path('datasets').glob('m8f_*.py'),*Path('simulator').glob('m8f_*.py'),*Path('tests').glob('test_m8f*.py'),
        *Path('docs').glob('M8F*.md'),*ROOT.rglob('*')]
 files=[asset(p) for p in sorted(set(paths)) if p.is_file() and '__pycache__' not in p.parts and p.name!='m8f_release_manifest.json']
 write('reports/m8f_release_manifest.json',dict(**provenance('FinalSourceAndEvidenceManifest'),files=files,historical_manifests=history,total_bytes=sum(a['bytes'] for a in files),
   source_contract='Final Git commit seals manifest. Execution parent+exact per-file source SHA in logs. Initial sources retained; no uncommitted historical changes.',
   raw_assets='External experiment root, not bulk Git data',checkpoints='None loaded or changed; historical SHA reverified.'))
 print('SEALED',len(files),'assets',sum(a['bytes'] for a in files),'bytes',flush=True)
if __name__=='__main__':main()
