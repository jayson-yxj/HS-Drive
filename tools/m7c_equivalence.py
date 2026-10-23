"""Paired all-history equivalence and timings; no held-out new-road tuning."""
import argparse,time,cProfile,pstats,io
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import carla
from shapely.ops import unary_union
from tools.m7c_common import *
from tools.m7b_fixtures import map_fixture
from diagnostics.m7a_planning import evaluate,generate
from diagnostics.m7b_motion import ObservedMotion,ArcRoute,build_hypotheses,evaluate_dynamic_hypotheses
from diagnostics.m7c_geometry import GeometryCache,evaluate_cached
from datasets.m7a_contracts import *

FILES=['development_v2','validation_frozen','fresh_final_frozen','historical_regression_frozen']

def request_from(r):
    d=r['request'];s=d['scene'];scene=OracleSceneState(Stamp(**s['stamp']),tuple(s['ego_xy_yaw_speed']),tuple(Box(**b) for b in s['boxes']),tuple(s['body_half_extent']),tuple(s['body_center_offset']))
    return PlanningRequest(scene,tuple(map(tuple,d['route'])),d['scene_id'],tuple(d['goal_xy_yaw']))

def candidate_from(d):return CandidateTrajectory(**{**d,'stamp':Stamp(**d['stamp'])})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile-only',action='store_true');ap.add_argument('--suffix',default='');args=ap.parse_args()
    cfg=read('configs/m7b/protocol.json');map_sha=sha(cfg['planning_map']);start=time.perf_counter();m=carla.Map('Town03',Path(cfg['planning_map']).read_text());mapms=(time.perf_counter()-start)*1000;cache=GeometryCache(map_sha);records=[];setups=[];profile=cProfile.Profile()
    for filename in FILES[:1] if args.profile_only else FILES:
        d=read('reports/m7b_'+filename+'.json');fixtures={}
        for r in d['records'][:3] if args.profile_only else d['records']:
            if r['fixture']['status']!='FixtureValid':continue
            start=time.perf_counter()
            if r['road'] not in fixtures:fixtures[r['road']]=map_fixture(m,r['road'],'regression');setups.append(dict(road=r['road'],ms=(time.perf_counter()-start)*1000))
            f=fixtures[r['road']];req=request_from(r);now=100.201 if r['spec'].get('fault')=='expired' else 99.99 if r['spec'].get('fault')=='future' else 100.
            t0=time.perf_counter();generated=generate(req,left_offset=f['sides'].get('left',{}).get('offset'),right_offset=f['sides'].get('right',{}).get('offset'),lane_permitted=r['spec']['kind']!='prohibited');genms=(time.perf_counter()-t0)*1000
            if r['spec'].get('fault') in ['missing','worker_error']:generated=[]
            obs=[ObservedMotion(o['object_id'],ArcRoute(o['route']),o['s0'],o['mode']) for o in r['observed_motion']]
            t0=time.perf_counter();hs,u=build_hypotheses(req.scene,obs,cfg);hypms=(time.perf_counter()-t0)*1000
            matches=equal([asdict(h) for h in hs],r['hypotheses']) and equal(u,r['prediction_unknown']);evs=[];times=dict(baseline_core=genms+hypms,optimized_cold=genms+hypms,optimized_warm=genms+hypms,stream=genms+hypms)
            for old in r['candidate_evaluations']:
                c=candidate_from(old['candidate']);allowed={f['lane']};s='left' if c.name=='B_left' else 'right' if c.name=='C_right' else None
                if s in f['sides'] and f['sides'][s]['permitted'] and r['spec']['kind']!='prohibited':allowed.add(f['sides'][s]['lane'])
                region=unary_union([f['polygons'][k] for k in allowed]).buffer(-.03)
                def exact(x,y):
                    w=m.get_waypoint(carla.Location(x=float(x),y=-float(y),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
                    return w is not None and w.road_id==r['road'] and w.lane_id in allowed
                ec=replace(c,name='D_stop') if c.name.startswith('E_stop') else c
                static={o.object_id for o in obs if o.mode=='static_geometry'};greq=replace(req,scene=replace(req.scene,boxes=tuple(b for b in req.scene.boxes if b.object_id in static)))
                t0=time.perf_counter()
                if args.profile_only:profile.enable()
                nominal=evaluate(req,ec,region,now,exact_road=exact)
                t1=time.perf_counter();geom=evaluate(greq,ec,region,now,exact_road=exact);dyn=evaluate_dynamic_hypotheses(c,req.scene,hs,u);base_ms=(time.perf_counter()-t1)*1000
                if args.profile_only:profile.disable()
                baseline_equal=equal(asdict(nominal),old['nominal']) and equal(asdict(geom),old['geometry']) and equal(dyn,old['dynamic'])
                new_modes={};permode={}
                # Per-candidate cold cache, then exact repeated candidate warm cache;
                # stream cache persists over scene sequence. No dynamic result caching.
                cold=GeometryCache(map_sha)
                for mode,store in [('cold',cold),('warm',cold),('stream',cache)]:
                    begin=time.perf_counter();reg=store.region(f['polygons'],allowed);query=store.exact([(r['road'],k) for k in allowed],f['z'],exact)
                    gg=evaluate_cached(greq,ec,reg,now,cache=store,exact_road=query);dd=evaluate_dynamic_hypotheses(c,req.scene,hs,u)
                    perms=(time.perf_counter()-begin)*1000;permode[mode]=perms
                    new_modes[mode]=equal(asdict(geom),asdict(gg)) and equal(dyn,dd)
                # Cross-mode CV reuses same static geometry but recomputes snapshot checks.
                nn=evaluate_cached(req,ec,region,now,cache=cache,exact_road=cache.exact([(r['road'],k) for k in allowed],f['z'],exact))
                match=baseline_equal and all(new_modes.values()) and equal(asdict(nominal),asdict(nn));matches &= match
                evs.append(dict(name=c.name,equivalent=match,baseline_matches_frozen=baseline_equal,modes=new_modes,robust=geom.planning_feasible and dyn['robust'],reasons=list(geom.reasons),dynamic_status=dyn['status'],collision_ids=dyn['collisions'],stop_margin=dyn['stop_margin_failures'],timing_ms=dict(baseline=base_ms,**permode),witness=c.name.startswith('E_stop')))
                if not c.name.startswith('E_stop'):
                    times['baseline_core']+=base_ms;times['optimized_cold']+=permode['cold'];times['optimized_warm']+=permode['warm'];times['stream']+=permode['stream']
            selected=next((e['name'] for e in evs if not e['witness'] and e['robust']),None);matches &= selected==r['decisions']['B_hypotheses']
            count=len([e for e in evs if not e['witness']]);matches &= count==len(generated)
            records.append(dict(scene_id=r['scene_id'],equivalent=bool(matches),candidate_count=count,selected=selected,old_selected=r['decisions']['B_hypotheses'],candidates=evs,unknown=u,timing_ms=times))
            print(r['scene_id'],bool(matches),times,flush=True)
    if args.profile_only:
        stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(65);Path('reports/m7c_execution_logs/baseline_profile.txt').write_text(stream.getvalue());return
    write(f'reports/m7c_geometry_cache_equivalence{args.suffix}.json',dict(kind='ActualServerPairedFrozenM7BAllValidReplay',provenance=provenance(),records=records,valid_scenes=len(records),all_equal=all(r['equivalent'] for r in records),map_setup_ms=mapms,fixture_setup=setups,cache=dict(hits=cache.hits,misses=cache.misses),scope='All candidate decisions including witnesses; no geometry tolerance on decisions; numeric1e-9',fresh_final=False))
if __name__=='__main__':main()
