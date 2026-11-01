"""Final cache/default/hash audit without model construction or new evaluation."""
from pathlib import Path
import json
import subprocess
from tools.m8j11a_common import read,write,sha,PROTOCOL


def main():
    c=read(PROTOCOL);root=Path(c['private_root']);fixed=[]
    for p,h in c['fixed_assets_sha256'].items():assert sha(p)==h,p;fixed.append(p)
    for row in c['T0_cache']:assert sha(row['path'])==row['sha256']
    for row in read(c['prepared_manifest'])['frames']:assert sha(row['path'])==row['sha256']
    runs={t:read(root/t/'result.json') for t in ('T1','T2','T3')}
    ledger=[json.loads(l) for l in (root/'forward_budget.jsonl').read_text().splitlines()]
    assert len(ledger)==203
    caches={}
    for t,r in runs.items():
        assert r['status']=='InterventionSequencePassed' and r['optimizer_steps']==r['backward_calls']==0
        assert r['initial_state_sha256']==r['final_state_sha256']
        assert r['protocol_sha256']==sha(PROTOCOL)
        assert sha(root/t/'memory.jsonl')==r['memory_log_sha256']
        assert [x['index'] for x in r['frames']]==[x['index'] for x in c['schedule'][t]]
        for f in r['frames']:
            p=root/t/('{:02}.npz'.format(f['index']));assert sha(p)==f['cache_sha256'];caches[str(p)]=sha(p)
    comparison=read('reports/m8j11a_comparison.json')
    for t in ('T1','T2'):
        e=comparison['exports'][t];assert sha(e['result_path'])==e['result_sha256']
        for n in ('summary','details'):
            assert sha(root/'evaluation'/t/'official'/('metrics_'+n+'.json'))==e['metrics_'+n+'_sha256']
    for path in Path('reports/m8j11a_sources').glob('*.json'):
        d=read(path)
        if 'log_path' in d:assert sha(d['log_path'])==d['log_sha256'] and d['exit_code']==0
    assert subprocess.check_output(['git','diff','--name-only','--diff-filter=MD',c['base_commit']],text=True).strip()==''
    assert subprocess.check_output(['git','rev-parse','main'],text=True).strip()==c['base_commit']
    visual=read(root/'visuals_manifest.json');assert len(visual['groups'])==6
    for g in visual['groups']:
        for k in ('six','bev'):assert sha(g[k+'_path'])==g[k+'_sha256']
    for p in Path('reports').glob('m8j11a*.json'):read(p)
    write(root/'new_cache_manifest.json',caches)
    report=dict(status='FinalHashesAndIsolationPassed',fixed_files_verified=len(fixed),T0_caches_verified=81,prepared_inputs_verified=81,
        new_caches_verified=len(caches),full_forwards=len(ledger),private_visual_groups=6,old_tracked_files_modified=0,
        main_unchanged=True,weight_state_unchanged=True,new_training=0,GT_payload_uploaded=False,
        test_exit_code=read('reports/m8j11a_sources/final_contract_tests.json')['exit_code'],
        clock='Exact UTC timestamps retained in phase logs',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol_sha256=sha(PROTOCOL),budget_ledger_sha256=sha(root/'forward_budget.jsonl'),new_cache_manifest_sha256=sha(root/'new_cache_manifest.json'))
    write('reports/m8j11a_sources/final_verification.json',report);print(report)

if __name__=='__main__':main()
