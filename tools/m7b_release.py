"""Final read-only acceptance consistency and preservation seal."""
from pathlib import Path
import re,subprocess,json
import xml.etree.ElementTree as ET
from tools.m7b_common import *
from tools.m7a_release import verify

REQUIRED=['reports/m7b_'+x for x in ['summary.md','acceptance.json','fixture_consistency.json','future_motion_validity.json','dynamic_hypotheses.json','robust_candidate_evaluation.json','dynamic_false_accepts.json','candidate_coverage.json','fresh_final_roads.json','planning_latency.json','shadow_regression.json']]+['docs/M7B_IMPLEMENTATION.md','docs/M7B_ARCHITECTURE_DECISION.md']

def main():
    for p in REQUIRED:assert Path(p).is_file(),p
    checks=dict(base_tracked=verify(read('reports/m7b_execution_logs/base_freeze.json')['tracked_files']),inherited_external=verify(read('runs/m6b_logs/freeze.json')['assets']),M7A_manifest=verify({v['path']:v['sha256'] for v in read('reports/m7a_release_manifest.json')['files']}),CARLA0916_identity=verify(read('reports/m6f_execution_logs/server_binary_manifest.json')['files']))
    frozen=['diagnostics/m7b_motion.py','tools/m7b_experiment.py','tools/m7b_fixtures.py','configs/m7b/protocol.json']
    for path in frozen:assert Path(path).read_bytes()==subprocess.check_output(['git','show','a18f754:'+path]),path
    refs={}
    def reference(obj):
        if isinstance(obj,dict):
            if {'path','sha256'}<=obj.keys():
                assert sha(obj['path'])==obj['sha256'],obj['path'];refs[obj['path']]=obj['sha256']
            for v in obj.values():reference(v)
        elif isinstance(obj,list):
            for v in obj:reference(v)
    for p in Path('reports').glob('m7b*.json'):reference(read(p))
    for p in [Path(x) for x in REQUIRED if x.endswith('.md')]:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in link:assert (p.parent/link).is_file(),(p,link)
    tests={}
    for name in ['full_tests','sdk_tests']:
        p='reports/m7b_execution_logs/'+name+'.xml';a=ET.parse(p).getroot().find('testsuite').attrib
        assert a['errors']=='0' and a['failures']=='0';tests[name]=dict(asset=asset(p),counts=a)
    audit=read('reports/m7b_independent_map_audit.json');assert audit['heading_failures']==audit['accepted_road_failures']==audit['stop_reference_gap_failures']==0
    acceptance=read('reports/m7b_acceptance.json');assert acceptance['status']=='Partial' and not acceptance['new_vehicle_runs']
    # Explicitly expose failures which are not represented by the headline finite-set gates.
    limits=read('reports/m7b_coverage_limits.json')
    write('reports/m7b_delivery_validation.json',dict(kind='ActualServerSourceAndAssetSHA_DeliveryAudit',provenance=provenance(),status='Passed_DeliveryOnly',required=[asset(p) for p in REQUIRED],preservation=checks,reference_assets_verified=len(refs),frozen_rules_unchanged=frozen,tests=tests,independent_map=asset('reports/m7b_independent_map_audit.json'),qualifications=dict(generator_global='Partial_two_reference_route_end_failures',new_final='46/46_limited_two_roads',arbitrary_future='NotValidated',finite_hypotheses='Passed_within_evaluated_references',task_conservatism_fresh_final=limits['per_split']['fresh_final']['count'],online5Hz='NoGo',C_generation='NotRun_NoDevelopmentGap',vehicle_execution='NotRun',perception='CandidateNoGo'),failure_ledger=asset('reports/m7b_execution_logs/failure_ledger.json')))
    files=[asset(p) for root in ['reports','docs','tools','diagnostics','tests','configs/m7b'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'm7b' in str(p).lower() and p.name!='m7b_release_manifest.json']
    write('reports/m7b_release_manifest.json',dict(kind='AppendOnlyStageAssetInventory',artifact_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=files,preservation=checks,self_reference='Excluded; final Git commit seals manifest. Actual execution source hashes remain in raw reports.'))
    print('M7-B delivery and historical SHA verified',checks,'new assets',len(files),'references',len(refs))
if __name__=='__main__':main()
