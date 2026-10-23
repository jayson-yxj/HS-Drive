"""Finite offline paired A/B research; does not tick CARLA or change control."""
import argparse,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import carla
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tools.m7b_common import read,write,provenance,asset
from tools.m7b_fixtures import map_fixture,construct,scenarios
from diagnostics.m7b_motion import build_hypotheses,evaluate_dynamic_hypotheses,interpolate,stop_witness
from diagnostics.m7a_planning import generate,evaluate,center
from simulator.oracle_driver import footprint


def reference_audit(candidate,scene,references):
    """Independent Shapely samples, no SAT/hypothesis decision reuse."""
    conflicts=[];min_gap=None
    for b in scene.boxes:
        ref=references[b.object_id];tt=np.array(ref['times']);op=np.array(ref['poses']);ep=interpolate(candidate.points,candidate.times_s,tt)
        first=None;gap=float('inf')
        for p,o,t in zip(ep,op,tt):
            inflated=Polygon(footprint(*center(p,scene.body_center_offset),p[2],*(np.array(scene.body_half_extent)+[.2,.15])))
            actual=Polygon(footprint(*center(p,scene.body_center_offset),p[2],*scene.body_half_extent))
            other=Polygon(footprint(*o[:2],o[2],b.half_length,b.half_width))
            if inflated.intersects(other) and first is None:first=float(t)
            gap=min(gap,actual.distance(other))
        if first is not None:conflicts.append(dict(object_id=b.object_id,first_sample_s=first))
        min_gap=gap if min_gap is None else min(gap,min_gap)
    return dict(collisions=conflicts,min_OBB_gap_m=min_gap,dt_s=.025,kind='IndependentSampledPolygon_NotContinuousMeshTruth')


def run(split,suffix,enable_c=False):
    cfg=read('configs/m7b/protocol.json');start=time.perf_counter();cmap=carla.Map('Town03',Path(cfg['planning_map']).read_text());read_ms=(time.perf_counter()-start)*1000;records=[];roadrecords=[]
    for road in cfg['splits'][split]:
        start=time.perf_counter()
        try:f=map_fixture(cmap,road,split)
        except Exception as e:
            roadrecords.append(dict(road=road,status='FixtureInvalid',reason=str(e)));continue
        roadrecords.append(dict(road=road,status='MapReady',map_setup_ms=(time.perf_counter()-start)*1000,route=f['route'],lane=f['lane'],sides=f['sides']))
        for i,spec in enumerate(scenarios()):
            ts=time.perf_counter();req,obs,refs,validity=construct(f,spec,i,cfg);fixture_ms=(time.perf_counter()-ts)*1000
            rec=dict(scene_id=req.scene_id,road=road,spec=spec,fixture=validity,request=asdict(req),evaluator_only_references=refs,observed_motion=[dict(object_id=o.object_id,s0=o.s0,mode=o.mode,source=o.source,route=o.route.points.tolist()) for o in obs],timing_ms=dict(fixture=fixture_ms))
            if validity['status']!='FixtureValid':records.append(rec);print(req.scene_id,validity['status'],validity['errors'],flush=True);continue
            start=time.perf_counter();side=f['sides'];candidates=generate(req,left_offset=side.get('left',{}).get('offset'),right_offset=side.get('right',{}).get('offset'),lane_permitted=spec['kind']!='prohibited');gen_ms=(time.perf_counter()-start)*1000
            if spec.get('fault') in ['missing','worker_error']:candidates=[]
            start=time.perf_counter();hs,unknown=build_hypotheses(req.scene,obs,cfg);hyp_ms=(time.perf_counter()-start)*1000
            rec['hypotheses']=[asdict(h) for h in hs];rec['prediction_unknown']=unknown
            witnesses=[]
            if not spec.get('fault'):
                for duration in [2.5,3.,3.5,4.,4.5,5.,5.5]:
                    try:witnesses.append(stop_witness(req,duration))
                    except ValueError:pass
            evaluations=[];stage=[]
            now=100.201 if spec.get('fault')=='expired' else 99.99 if spec.get('fault')=='future' else 100.
            for c in candidates+witnesses:
                allowed={f['lane']};s='left' if c.name=='B_left' else 'right' if c.name=='C_right' else None
                if s in side:
                    permit=side[s]['permitted'] and spec['kind']!='prohibited';c=replace(c,lane_change_permitted=permit)
                    if permit:allowed.add(side[s]['lane'])
                region=unary_union([f['polygons'][k] for k in allowed]).buffer(-.03)
                def exact(x,y):
                    w=cmap.get_waypoint(carla.Location(x=float(x),y=-float(y),z=f['z']),project_to_road=False,lane_type=carla.LaneType.Driving)
                    return w is not None and w.road_id==road and w.lane_id in allowed
                # E_stop uses the same original stop check; no privilege from its new name.
                ec=replace(c,name='D_stop') if c.name.startswith('E_stop') else c
                t0=time.perf_counter();nom=evaluate(req,ec,region,now,exact_road=exact);nomms=(time.perf_counter()-t0)*1000
                static_ids={o.object_id for o in obs if o.mode=='static_geometry'}
                geomreq=replace(req,scene=replace(req.scene,boxes=tuple(b for b in req.scene.boxes if b.object_id in static_ids)))
                t0=time.perf_counter();geom=evaluate(geomreq,ec,region,now,exact_road=exact);geomms=(time.perf_counter()-t0)*1000
                t0=time.perf_counter();dyn=evaluate_dynamic_hypotheses(c,req.scene,hs,unknown);dynms=(time.perf_counter()-t0)*1000
                t0=time.perf_counter();audit=reference_audit(c,req.scene,refs);auditms=(time.perf_counter()-t0)*1000
                usable=geom.planning_feasible and dyn['robust']
                evaluations.append(dict(name=c.name,candidate=asdict(c),nominal=asdict(nom),geometry=asdict(geom),dynamic=dyn,robust_accepted=usable,reference=audit,witness_only=c.name.startswith('E_stop') and not enable_c))
                stage.append(dict(name=c.name,legacy_CV_ms=nomms,body_road_static_ms=geomms,multi_hypothesis_collision_ms=dynms,independent_reference_ms=auditms,witness=c.name.startswith('E_stop')))
            t0=time.perf_counter();base=[e for e in evaluations if not e['name'].startswith('E_stop')]
            def choose(es,key):return next((e['name'] for e in es if key(e)),None)
            rec['decisions']=dict(A_CV=choose(base,lambda e:e['nominal']['planning_feasible']),B_hypotheses=choose(base,lambda e:e['robust_accepted']),C_dynamic_stop=choose(evaluations,lambda e:e['robust_accepted']) if enable_c else 'NotRun_ConditionalGate')
            rec['candidate_evaluations']=evaluations;rec['witness_feasible']=[e['name'] for e in evaluations if e['name'].startswith('E_stop') and e['robust_accepted']]
            rec['timing_ms'].update(generation=gen_ms,hypothesis_generation=hyp_ms,selection=(time.perf_counter()-t0)*1000,candidates=stage)
            records.append(rec);print(req.scene_id,rec['decisions'],'witness',rec['witness_feasible'],flush=True)
    write(f'reports/m7b_{split}{suffix}.json',dict(kind='OfflineCARLA_Map_SyntheticActors_NoVehicleExecution',provenance=provenance(),protocol=asset('configs/m7b/protocol.json'),map_read_ms=read_ms,roads=roadrecords,records=records,C_enabled=enable_c))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['development','validation','fresh_final','historical_regression']);p.add_argument('--suffix',required=True);p.add_argument('--enable-c',action='store_true');a=p.parse_args();run(a.split,a.suffix,a.enable_c)
