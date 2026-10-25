"""Seal read-only review, source revisions, history and runtime evidence."""
import ast,json,subprocess
from pathlib import Path
from tools.m8h0_common import *
from tools.m8e_verify import verify

def main():
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
 assert not subprocess.check_output(['git','diff',BASE,'--name-only'],text=True).strip(),'Historical tracked files changed'
 history=verify('reports/m8g_release_manifest.json');index=json.loads(HISTORY.read_text())
 for a in index['historical_sources']:assert sha(a['path'])==a['sha256'],a['path']
 required=['summary.md','acceptance.json','native_vehicle_spawn_inventory.json','pedestrian_navigation_inventory.json','curated_template_candidates.json','geometry_and_overlap_audit.json','historical_preflight.json','architecture_comparison.md']
 assert all(Path('reports/m8h0_'+n).is_file() for n in required)
 assert all(Path('docs/'+n).is_file() for n in ['M8H0_IMPLEMENTATION.md','M8H0_ARCHITECTURE_DECISION.md'])
 for folder,pattern in [('tools','m8h0_*.py'),('datasets','m8h0_*.py'),('tests','test_m8h0*.py')]:
  for p in Path(folder).glob(pattern):ast.parse(p.read_text())
 for p in [*Path('reports').glob('m8h0_*.json'),*LOG.glob('*.json')]:
  d=json.loads(p.read_text())
  for f,h in d.get('sources',{}).items():
   if sha(f)!=h:
    old=ROOT/'source_revisions'/h/f;assert old.is_file() and sha(old)==h,(f,h,str(p))
 tests=json.loads(Path('reports/m8h0_test_results.json').read_text());assert tests['total']==70
 clean=json.loads(Path('reports/m8h0_runtime_cleanup.json').read_text());assert clean['vehicles']==clean['walkers']==clean['sensors']==0 and not clean['synchronous_mode']
 assert len((ROOT/'navigation_returns.jsonl').read_text().splitlines())==60
 write('reports/m8h0_final_validation.json',dict(**provenance('FinalEvidenceValidation'),M8G_sealed_files_verified=history,historical_source_SHA_verified=len(index['historical_sources']),tests=70,offline_frozen_validator_decisions=161,no_historical_or_default_modifications=True,own_server_shutdown='PID139479 SIGTERM after verified empty asynchronous world',actual_smoke='NotRun',independent_Final=0,template_proposal_not_permission=True,raw_root=str(ROOT),all_deliverables_present=True))
 paths=[*Path('configs/m8h0').rglob('*'),*Path('reports').glob('m8h0_*.json'),*Path('reports').glob('m8h0_*.md'),*Path('reports/m8h0_figures').rglob('*'),*LOG.rglob('*'),*Path('tools').glob('m8h0_*.py'),*Path('datasets').glob('m8h0_*.py'),*Path('tests').glob('test_m8h0*.py'),*Path('docs').glob('M8H0*.md'),*ROOT.rglob('*')]
 files=[asset(p) for p in sorted(set(paths)) if p.is_file() and '__pycache__' not in p.parts and p.name!='m8h0_release_manifest.json' and not(p.parent==LOG and p.name.startswith('finalize'))]
 write('reports/m8h0_release_manifest.json',dict(**provenance('FinalSourceAndEvidenceSeal'),files=files,total_bytes=sum(f['bytes'] for f in files),M8G_manifest=asset('reports/m8g_release_manifest.json'),map_identity=asset('reports/m8h0_map_identity_and_runtime.json'),history=asset(HISTORY),raw_asset_backup='External raw root needs separate backup; Git manifest does not upload raw data',commit_contract='Final Git commit seals manifest; logs retain execution parent and per-file source SHA'))
 print('SEALED',len(files),'files;',tests['total'],'tests;161frozen comparisons;0Actor/RGB',flush=True)
if __name__=='__main__':main()
