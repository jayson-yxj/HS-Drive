"""Final read-only review after immutable schedule snapshot correction."""
import ast,json
from pathlib import Path
import xml.etree.ElementTree as ET
from simulator.recorder import atomic_json,file_hash

def read(p):return json.loads(Path(p).read_text())
def main():
    out=Path('reports/m48_delivery.json')
    if out.exists():raise FileExistsError(out)
    acceptance=read('reports/m48_acceptance.json');changes={}
    for p,sha in acceptance['sources'].items():
        if file_hash(p)!=sha:changes[p]={'before':sha,'after':file_hash(p)}
    tests={}
    for group,name in [('sim','reports/m48_delivery_sim_tests.xml'),('gpu','reports/m48_delivery_gpu_tests.xml'),('module','reports/m48_delivery_module_tests.xml')]:
        suite=ET.parse(name).getroot().find('testsuite');assert int(suite.attrib['errors'])==0 and int(suite.attrib['failures'])==0
        tests[group]=dict(**suite.attrib,path=name,sha256=file_hash(name))
    faults=read('reports/m48_multirate_faults_final.json');assert faults['source_unchanged_during_run']
    assert all(r['status']=='Passed' and r['collision_count']==0 for r in faults['results'])
    for r in faults['results']:assert file_hash(r['log'])==r['log_sha256']
    frozen=read('reports/m48_asset_freeze.json');mismatches=[]
    for key in ['tracked_sha256','checkpoint_sha256']:
        for p,sha in frozen[key].items():
            if file_hash(p)!=sha:mismatches.append(p)
    for p,info in frozen['data_size_mtime'].items():
        stat=Path(p).stat()
        if [stat.st_size,stat.st_mtime_ns]!=info:mismatches.append(p)
    assert not mismatches
    atomic_json(out,dict(kind='offline_delivery_source_and_test_review',assets_preserved=True,changes_since_acceptance=changes,
        explanation='Schedule now deep-copies plan/state so fault injection cannot mutate authoritative age. Controller/label behavior unchanged. Whitespace cleanup in runners/safety; runtime snapshots preserved. Added regression test and reran actual CARLA four fault cases.',
        tests=tests,fault_validation_sha256=file_hash('reports/m48_multirate_faults_final.json'),
        source_sha256={str(p):file_hash(p) for folder in ['tools','simulator','datasets','models','training','tests'] for p in Path(folder).glob('*m48*.py')},
        overall='Partial',formal_m5_approved=False,async_controller_realtime_pass=False))
    print('delivery checks passed; quality remains Partial')
if __name__=='__main__':main()
