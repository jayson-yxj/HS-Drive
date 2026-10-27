import random
import pytest
from datasets.m8j10_selection import select_all,schedule


def fixtures():
    scenes=[dict(name=n,token=n,nbr_samples=3) for n in ('a','b','v')]
    samples=[dict(token=n+str(i),scene_token=n,timestamp=i+1) for n in ('a','b','v') for i in range(3)]
    splits=dict(mini_train=['b','a','excluded'],train=['a','b'],mini_val=['v'],val=['v'])
    return scenes,samples,splits


def test_official_intersection_all_frames():
    frames=select_all(*fixtures())
    assert len(frames)==6 and [f['sample_token'] for f in frames]==['a0','a1','a2','b0','b1','b2']


def test_no_validation():
    sc,sa,sp=fixtures();sp['train'].append('v');sp['mini_train'].append('v')
    with pytest.raises(AssertionError):select_all(sc,sa,sp)


def test_missing_frame_not_silent():
    sc,sa,sp=fixtures()
    with pytest.raises(AssertionError):select_all(sc,sa[1:],sp)


def test_reproducible_balanced_and_rng_isolated():
    frames=select_all(*fixtures());state=random.getstate()
    a=schedule(frames,875606,400)
    assert random.getstate()==state and a==schedule(frames,875606,400)
    assert len(a)==400 and [x['step'] for x in a]==list(range(1,401))
    assert [x['scene_name'] for x in a[:6]]==['a','b']*3
    for start in (0,6,12):assert len({x['index'] for x in a[start:start+6]})==6


def test_different_seed_changes_sample_order_only():
    f=select_all(*fixtures());a=schedule(f,1);b=schedule(f,2)
    assert [r['scene_name'] for r in a]==[r['scene_name'] for r in b]
    assert [r['index'] for r in a]!=[r['index'] for r in b]


def test_frozen_training_settings_equal_j7_and_smoke_clip():
    import json
    from pathlib import Path
    c=json.loads(Path('configs/m8j10_protocol.json').read_text())
    j7=json.loads(Path('configs/m8j7_protocol.json').read_text())
    j6=json.loads(Path('configs/m8j6_protocol.json').read_text())
    for key,value in j7['training'].items():
        if key!='order':assert c['training'][key]==value,key
    assert c['training']['gradient_clip_norm']==j6['smoke']['gradient_clip_norm']
    assert c['training']['maximum_optimizer_steps']==400 and c['training']['scheduler'] is None
    assert c['A']['sha256']!=c['B']['sha256']


def test_actual_fixed_schedule_matches_declared_generator():
    import json
    from pathlib import Path
    m=json.loads(Path('reports/m8j10_training_manifest.json').read_text())
    assert m['order']==schedule(m['frames'],875606,400)
    assert len({r['index'] for r in m['order']})==len(m['frames'])==242
    counts={s:sum(r['scene_name']==s for r in m['order']) for s in {f['scene_name'] for f in m['frames']}}
    assert max(counts.values())-min(counts.values())==1
    for name in counts:
        first=[r['index'] for r in m['order'] if r['scene_name']==name and r['scene_cycle']==0]
        assert len(first)==len(set(first))==sum(f['scene_name']==name for f in m['frames'])
