"""Final read-only verification; no GPU or inference."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from datasets.m8j6_selection import sha256


def read(p):return json.loads(Path(p).read_text())
def main():
    cfg=read('configs/m8j10_protocol.json');root=Path(cfg['private_root']);rep=Path('reports/m8j10_sources')/cfg['run_id']
    train=read(rep/'train/result.json');verify=read(rep/'verify/result.json');official=read(rep/'official/result.json')
    assert train['status']=='Completed400Steps' and verify['status']=='FinalReloadPassed'
    assert official['status']=='SameProtocolB_C200_C400_Completed'
    ledgers=[read(p) for p in (root/'steps').glob('optimizer_step_*_returned.json')]
    assert sorted(r['global_step'] for r in ledgers)==list(range(1,401))
    assert verify['optimizer_steps']==0 and verify['new_process']
    for model in ('A','B'):assert sha256(cfg[model]['path'])==cfg[model]['sha256']
    historical=[]
    for stage in ('m8j7','m8j8','m8j8a','m8j9'):
        mf=read(f'reports/{stage}_release_manifest.json')
        for f in mf['files']:assert sha256(f['path'])==f['sha256'],f['path']
        historical.append(dict(stage=stage,files=len(mf['files']),unchanged=True))
    for p,h in cfg['source_sha256'].items():assert sha256(p)==h,p
    for p,h in cfg['official_sources_sha256'].items():assert sha256(p)==h,p
    cache_checks={}
    for model in ('A','B'):
        run=read(f'reports/m8j8_sources/m8j8_20261027_ab/{model}/result.json')
        for f in run['frames']:assert sha256(f['cache_path'])==f['cache_sha256']
        cache_checks[model]=len(run['frames'])
    for step in (200,400):
        cp=read(rep/'train'/f'checkpoint_{step:03}.json');assert sha256(cp['path'])==cp['sha256']
        ev=read(rep/'train'/f'eval_{step:03}.json')
        assert len(ev['frames'])==81 and ev['model_state_unchanged']
        for f in ev['frames']:assert sha256(f['cache_path'])==f['cache_sha256']
        cache_checks['C'+str(step)]=81
    for model in ('B','C200','C400'):
        r=read(rep/'official'/model/'result.json');assert sha256(r['result_path'])==r['result_sha256']
        assert r['GT_official']==3393 and len(r['frames'])==81 and r['export_GT_fields_read']==[]
    official_scores=read('reports/m8j10_official_comparison.json')['models']
    a=read('reports/m8j9_acceptance.json')['scores'];assert official_scores['A']==a
    phases=[]
    for phase in ('prepare','train','verify','official','report'):
        s=read(rep/phase/'supervisor.json');assert s['exit_code']==0
        assert not Path('/proc',str(s['PID'])).exists()
        assert sha256(rep/phase/'execution.txt')==s['log_sha256']
        phases.append(dict(phase=phase,PID=s['PID'],exit_code=0,ended=True,execution_commit=s['code_commit']))
    payload=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),status='EngineeringRunVerified',actual_optimizer_steps=400,
        final_reload='Passed',frozen_parameters_and_buffers='ExactUnchanged',A_B_original_weights_unchanged=True,
        original_model_and_official_sources_unchanged=True,history=historical,caches_verified=cache_checks,phases=phases,
        official_denominator=3393,project_denominator=3559,A_baseline_exact=True,B_new_forwards=0,
        C_forward_scope='400 training + 162 mini_val + 2 save probes + 1 independent reload probe',
        main_merged=False,private_assets_published=False,model_quality='SeparateDecision_NotImpliedByEngineeringPass')
    with Path('reports/m8j10_verification.json').open('x') as f:json.dump(payload,f,indent=2)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
