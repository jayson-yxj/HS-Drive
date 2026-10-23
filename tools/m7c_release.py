"""Verify immutable assets, delivered report links and rules before sealing."""
import subprocess,re,xml.etree.ElementTree as ET
from pathlib import Path
from tools.m7c_common import *
from tools.m7a_release import verify

REQUIRED=['reports/m7c_'+s for s in ['summary.md','acceptance.json','geometry_cache_equivalence.json','planning_latency.json','connected_route_support.json','prediction_support_audit.json','stop_coverage_attribution.json','slow_follow_feasibility.json','candidate_coverage.json','dynamic_false_accepts.json','fresh_final_roads.json','shadow_regression.json']]+['docs/M7C_IMPLEMENTATION.md','docs/M7C_ARCHITECTURE_DECISION.md']

def main():
 for p in REQUIRED:assert Path(p).is_file(),p
 checks=dict(base_tracked=verify(read('reports/m7c_execution_logs/base_freeze.json')['files']),inherited_external=verify(read('runs/m6b_logs/freeze.json')['assets']),M7B=verify({x['path']:x['sha256'] for x in read('reports/m7b_release_manifest.json')['files']}),CARLA0916=verify(read('reports/m6f_execution_logs/server_binary_manifest.json')['files']))
 freeze=read('reports/m7c_execution_logs/rule_freeze.json');verify(freeze['sources'])
 eq=read('reports/m7c_geometry_cache_equivalence_prepared.json');assert eq['all_equal'] and eq['valid_scenes']==205
 assert read('reports/m7c_historical_seven.json')['detected']==7
 audit=read('reports/m7c_independent_final_audit.json');assert all(audit[k]==0 for k in ['actor_failures','accepted_road_failures','stop_gap_failures','future_conflicts'])
 tests={}
 for name in ['full_tests_v3','sdk_frozen_tests']:
  p='reports/m7c_execution_logs/'+name+'.xml';a=ET.parse(p).getroot().find('testsuite').attrib;assert a['errors']==a['failures']=='0';tests[name]=dict(asset=asset(p),counts=a)
 links=[]
 for p in [Path(s) for s in REQUIRED if s.endswith('.md')]:
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' not in target:assert (p.parent/target).is_file(),(p,target);links.append(dict(document=str(p),target=target))
 refs={}
 def check(d):
  if isinstance(d,dict):
   if {'path','sha256'}<=d.keys():
    if sha(d['path'])!=d['sha256']:
     assert d['path']=='configs/m7c/protocol.json' and sha('configs/m7c/development_protocol_snapshot.json')==d['sha256'],d['path']
     refs['configs/m7c/development_protocol_snapshot.json']=d['sha256']
    else:refs[d['path']]=d['sha256']
   for v in d.values():check(v)
  elif isinstance(d,list):
   for v in d:check(v)
 for p in Path('reports').glob('m7c*.json'):check(read(p))
 assert read('reports/m7c_acceptance.json')['status']=='Partial'
 write('reports/m7c_delivery_validation.json',dict(kind='ActualServerPreservationAndDeliveryAudit',provenance=provenance(),status='Passed_DeliveryOnly',required=[asset(p) for p in REQUIRED],preservation=checks,frozen_rules=freeze,tests=tests,reference_assets_checked=len(refs),markdown_links=links,independent_audit=asset('reports/m7c_independent_final_audit.json'),scope='Does not promote runtime, physical execution, vision, occupancy or arbitrary-future safety',qualified_states=dict(cache_equivalence='Passed',latency='NoGo',continuation='Partial',slow_follow='LimitedResearchGain',vehicle_execution='NotRun',vision='CandidateNoGo',nonrealtime_Pilot='ProposalOnly')))
 files=[asset(p) for root in ['reports','docs','tools','diagnostics','tests','configs/m7c'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'm7c' in str(p).lower() and p.name!='m7c_release_manifest.json']
 write('reports/m7c_release_manifest.json',dict(kind='AppendOnlyStageSHAInventory',artifact_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=files,preservation=checks,self_reference='Excluded; final commit seals inventory. Actual per-run SHA preserved.'))
 print('preserved',checks,'assets',len(files),'refs',len(refs))
if __name__=='__main__':main()
