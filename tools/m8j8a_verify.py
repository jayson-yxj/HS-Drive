"""Final cache-only release checks. No model/weights, no forward, no new cases."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)


def main():
    cfg=read('configs/m8j8a_protocol.json');execution=read('reports/m8j8a_execution.json')
    release=read('reports/m8j8_release_manifest.json')
    for entry in release['files']:assert sha(entry['path'])==entry['sha256'],entry['path']
    for p,h in execution['inputs_sha256'].items():assert sha(p)==h,p
    private=Path(cfg['private_output'])
    for p,h in execution['private_outputs_sha256'].items():assert sha(private/p)==h,p
    selected=read(private/'case_manifest.json')
    assert len(selected)==len({r['instance_token'] for r in selected})==12
    assert [sum(r['bucket']==x for r in selected) for x in cfg['cases']['bucket_order']]==[6,3,3]
    trans=read('reports/m8j8a_GT_transitions.json');tot=trans['total']['states']
    for k,count in cfg['expected_transitions'].items():
        assert tot[k]['records']==count
        assert sum(x['states'][k]['records'] for x in trans['classes'].values())==count
        assert sum(x['states'][k]['records'] for x in trans['scenes'].values())==count
    for script in ['tools/m8j8a_closeout.py','training/m8j8a_cache_diagnostics.py','configs/m8j8a_protocol.json']:
        source=subprocess.check_output(['git','show',execution['code_commit']+':'+script])
        assert hashlib.sha256(source).hexdigest()==sha(script),script
    changes=subprocess.check_output(['git','diff','9380732','--name-only','--diff-filter=MD'],text=True).splitlines()
    assert all('m8j8a' in p.lower() for p in changes),changes
    assert execution['model_forward_calls']==execution['optimizer_steps']==execution['backward_calls']==0
    assert '24 passed' in Path('reports/m8j8a_sources/tests_pre_execution.txt').read_text()
    public=list(Path('reports').glob('m8j8a*.json'))
    for p in public:
        data=p.read_text()
        for token in ['"GT_index"','"annotation_token"','"raw_global_center"','"raw_full_wlh"','"prediction_index"']:
            assert token not in data,(str(p),token)
    write('reports/m8j8a_preservation.json',dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        historical_release_files_checked=len(release['files']),historical_release_unchanged=True,
        input_cache_and_source_hashes_checked=len(execution['inputs_sha256']),all_inputs_unchanged=True,
        private_transition_case_and_html_files_unchanged=True,case_count=12,case_instances=12,
        budgets=[6,3,3],transition_class_and_scene_totals_consistent=True,
        executed_sources_match_commit=True,existing_files_modified=[],model_forward_calls=0,optimizer_steps=0,
        private_data_labels_images_models_published=False))
    print('Verified historical release, cached inputs, 12 unique cases and exact transition totals.')


if __name__=='__main__':main()
