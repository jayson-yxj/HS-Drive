"""Fixed official-split intersection and independent scene-round-robin schedule."""
from collections import Counter
import random


def select_all(scenes, samples, splits):
    eligible = sorted(set(splits['mini_train']) & set(splits['train']))
    forbidden = set(splits['mini_val']) | set(splits['val'])
    assert eligible and not (set(eligible) & forbidden)
    lookup = {s['name']: s for s in scenes}
    assert len({x['token'] for x in samples}) == len(samples)
    frames = []
    for name in eligible:
        scene = lookup[name]
        seq = sorted((s for s in samples if s['scene_token'] == scene['token']),
                     key=lambda s: (s['timestamp'], s['token']))
        assert len(seq) == scene['nbr_samples'] and seq
        for s in seq:
            frames.append(dict(index=len(frames),sample_token=s['token'],scene_token=scene['token'],
                               scene_name=name,timestamp_us=s['timestamp']))
    return frames


def schedule(frames, seed, steps=400):
    """Sorted scene round robin; per-scene seeded shuffle, repeat on exhaustion.

    Separate RNG from model/dropout; reshuffle each scene on its next cycle.
    Explicit integer-derived scene seed is stable across Python hash seeds.
    """
    names=sorted({f['scene_name'] for f in frames})
    pools={n:[f['index'] for f in frames if f['scene_name']==n] for n in names}
    assert all(pools.values()) and len({f['index'] for f in frames})==len(frames)
    rng={n:random.Random(seed+1009*i) for i,n in enumerate(names)}
    cursor={n:0 for n in names};cycle={n:0 for n in names}
    for n in names:rng[n].shuffle(pools[n])
    result=[]
    for step in range(steps):
        name=names[step%len(names)]
        if cursor[name]==len(pools[name]):
            cursor[name]=0;cycle[name]+=1;rng[name].shuffle(pools[name])
        result.append(dict(step=step+1,scene_name=name,index=pools[name][cursor[name]],scene_cycle=cycle[name]))
        cursor[name]+=1
    return result
