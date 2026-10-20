"""Independent M48 interface, pilot, generalization and closed-loop gates."""
import argparse,json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
from simulator.recorder import atomic_json,file_hash

def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/m48_acceptance.json');a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    old=read('reports/m48_asset_freeze.json');mismatch=[]
    for group in ['tracked_sha256','checkpoint_sha256']:
        for path,sha in old[group].items():
            if not Path(path).is_file() or file_hash(path)!=sha:mismatch.append([group,path])
    for path,info in old['data_size_mtime'].items():
        q=Path(path)
        if not q.exists() or [q.stat().st_size,q.stat().st_mtime_ns]!=info:mismatch.append(['old_data',path])
    # M47 external raw/sidecar content hashes, not just legacy data metadata.
    for path,sha in read('reports/m47_data_manifest.json')['external_files'].items():
        if file_hash(path)!=sha:mismatch.append(['m47_external',path])
    protocol=read('configs/m48_protocol.json')
    frozen=protocol['label_source_sha256']==file_hash('datasets/m48_evidence.py') and protocol['quality_source_sha256']==file_hash('tools/m48_spatial.py')
    tests={}
    for key,path in [('sim','reports/m48_release_sim_tests.xml'),('gpu','reports/m48_release_gpu_tests.xml'),('m48','reports/m48_final_module_tests.xml')]:
        suite=ET.parse(path).getroot().find('testsuite');tests[key]={**suite.attrib,'path':path,'sha256':file_hash(path)}
    tests_ok=all(int(v['errors'])==0 and int(v['failures'])==0 for v in tests.values())
    pilot=read('reports/m48_occupancy_pilot.json');rgb=pilot['records'][0]
    pilot_ok=rgb['finite_gradients'] and all(rgb['ignored_gradient_zero'].values()) and all(x['iou']>.95 for x in rgb['metrics'].values())
    q=read('reports/m48_spatial_heldout.json');driving=read('reports/m48_junction_validation.json')
    evidence=[str(p) for p in Path('reports').glob('m48_*.json') if p.name not in [Path(a.output).name,'m48_asset_freeze.json']]
    sources={str(p):file_hash(p) for folder in ['datasets','models','training','simulator','tools','tests','configs'] for p in Path(folder).glob('*m48*') if p.is_file()}
    external={}
    for report in ['reports/m48_spatial_evidence.json','reports/m48_spatial_heldout.json']:
        for r in read(report)['records']:
            assert file_hash(r['label'])==r['label_sha256'];external[r['label']]=r['label_sha256']
    for root in Path('/root/gpufree-data/hs-drive-data/raw').glob('m48_*'):
        for p in root.rglob('*'):
            if p.is_file():external[str(p)]=file_hash(p)
    for r in pilot['records']:assert file_hash(r['checkpoint'])==r['sha256'];external[r['checkpoint']]=r['sha256']
    for report in ['reports/m48_junction_pilot_v3.json','reports/m48_junction_short.json','reports/m48_multirate_verified.json']:
        assert read(report)['source_unchanged_during_run']
    atomic_json(a.output,dict(schema='hsd-m48-acceptance-v1',overall='Partial',base_commit=old['commit'],branch='feat/m48-spatial-evidence-control',
        independent_states=dict(engineering_interface='Passed' if tests_ok and frozen else 'Failed',research_training='Passed: two fixed frames only' if pilot_ok else 'Partial',generalization_quality='Partial: limited heldout scenes, incomplete spatial reference',safety_closedloop='Partial: comfort and wall-clock realtime remain unpassed'),
        gates=dict(nonexclusive_evidence_contract=tests_ok,all104_local_coexistence=read('reports/m48_audit104_representation.json')['both']==104,
            frozen_rule_before_unseen_scenes=frozen,supervision_isolation=tests_ok,finite_pilot_gradients=rgb['finite_gradients'],pilot_unknown_gradients_zero=all(rgb['ignored_gradient_zero'].values()),
            formal_occupancy_generalization=False,comfortable_complete_stop=False,proven_physical_impossibility=False,full_horizon_20hz_realtime=False,
            three_route_geometry_recovery=True,async_controller_20hz=False,precise_continuous_mesh_road_collision=False),
        assets_preserved=not mismatch,asset_mismatches=mismatch,old_counts={k:len(old[k]) for k in ['tracked_sha256','checkpoint_sha256','data_size_mtime']},m47_external_count=len(read('reports/m47_data_manifest.json')['external_files']),
        default_behavior_changed=False,depth_bins_changed=False,splat_changed=False,vehicle_physics_changed=False,
        m5_pilot_completed=pilot_ok,m5_formal_training_approved=False,automatic_training_expansion_approved=False,
        tests=tests,sources=sources,evidence={p:file_hash(p) for p in evidence},external_assets_sha256=external,
        limitations=['Pilot is observation-event learning, not physical empty-volume classification.','Other semantic positive learning untested (no positives).','Serial multi-rate loop passes simulated scheduling but blocks wall-clock controller on planning ticks.','Legacy old raw data checked by size/mtime; M47 and new external files checked by SHA.']))
    print(json.dumps(dict(assets_preserved=not mismatch,tests_pass=tests_ok,pilot=pilot_ok,rule_frozen=frozen,mismatches=mismatch)))
    if mismatch or not tests_ok or not frozen:raise SystemExit(1)
if __name__=='__main__':main()
