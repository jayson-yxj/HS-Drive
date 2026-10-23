"""One conditional experiment declaration using development only, before validation."""
import json
from pathlib import Path
from tools.m6e_common import write,provenance,asset

def main():
    records=json.loads(Path('reports/m6e_development_analysis.json').read_text())['trials']
    assert len(records)==12
    assert all(not r['normal_comfort_passed'] for r in records)
    raw=[p['raw']['max_decel_raw'] for r in records if r['config']['interface']=='A' for phase,p in r['phases'].items() if phase in ['normal_stop','final_stop']]
    assert min(raw)>1.8
    tasks=[]
    for split,pos in [('development',27),('validation',66)]:
        for seed in [301,302]:tasks.append(dict(name=f'P1_{split}_p{pos}_s{seed}_A',physics='P1',split=split,spawn=pos,seed=seed,interface='A',mode='cycle'))
    for split,pos in [('development',126),('validation',170)]:
        for seed in [301,302]:
            for mode in ['emergency','coast','obstacle']:
                for version in ['P0','P1']:tasks.append(dict(name=f'{version}_{split}_p{pos}_s{seed}_{mode}',physics=version,split=split,spawn=pos,seed=seed,interface='A',mode=mode))
    write('configs/m6e/physics_P1.json',dict(version='Physics-P1-Experimental',family='wheel_max_brake_torque',scale=.5,
        changed_fields=[f'wheels[{i}].max_brake_torque' for i in range(4)],old_values=[700]*4,new_values=[350]*4,
        rationale='P0 A=.01 brake repeatedly produces >1.8m/s2; fixed half-torque intervention tests wheel-brake gain dependence, not asserted root cause. Startup unchanged is expected. Moderate single factor, no tuning to pass.',
        caveat='Simple torque/mass/radius calculation at .01 predicts ~.041m/s2, far below observed peaks; tire/transmission/solver transients may dominate. P1 can falsify proportional brake hypothesis. Not a physically validated Tesla.',
        development_evidence=asset('reports/m6e_development_analysis.json'),validation_accessed_for_selection=False,tasks=tasks,
        emergency_gate='No >0.25m longer stopping distance or >0.25s longer stopping time against matched P0; real gap>=3m at initial5m; no collision/boundary/hold degradation. Any violation blocks promotion.',
        success='All original raw comfort, hold/restart, guard and safety gates. No second variant if fails.'))
    write('reports/m6e_physics_variant_declaration.json',dict(**provenance('development_only_conditional_single_family_declaration'),
        development_raw_brake_decel_range=[min(raw),max(raw)],variant=asset('configs/m6e/physics_P1.json'),
        conclusion='P0 interfaces failed; test causal sensitivity of a single wheel-brake parameter family. No uniquely identified internal physics cause.'))
    print('declared',len(tasks),'fixed P0/P1 experiments')
if __name__=='__main__':main()
