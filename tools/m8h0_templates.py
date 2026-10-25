"""Finite offline template selection from exact inventoried native coordinates."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import yaml
from tools.m8h0_common import *
from datasets.m8h0_geometry import footprint,pair_relation,visibility
from tools.m8g_history import preflight

def camera_region_history(ego,index):
    xy=np.asarray(ego['geometry']['hypothetical_actor_matrix'])[:2,3];rows=[r for r in index['records'] if r['map']=='Town02' and not r['target']]
    nearby=[r for r in rows if np.linalg.norm(xy-r['center_xy'])<=40+r['radius_m']]
    return dict(nominal_40m_camera_disk_overlapping_records=len(nearby),kinds=dict(Counter(r['kind'] for r in nearby)),method='Conservative horizontal disk versus historical scene/Ego envelopes; not rendered pixel overlap',no_overlap_is_not_freshness_proof=True,coverage='HistoricalCoverageIncomplete')

def template_family(vehicle_view,person_view):
    vb,pb=vehicle_view['bearing_deg'],person_view['bearing_deg']
    if abs(vb)<45 and abs(pb)<75:return 'A_FrontVehicleSidewalkPerson'
    if abs(vb-pb)>120:return 'D_FrontRearObservation'
    return 'B_SideObservationWithSidewalkPerson'

def segment_box_interval(start,end,box):
    """Conservative visibility warning, never a mesh occlusion verdict."""
    m=np.linalg.inv(np.asarray(box['T_world_from_box']));p=m@np.r_[start,1];q=m@np.r_[end,1];delta=q[:3]-p[:3];ext=np.asarray(box['extent_m']);low,high=0.,1.
    for i in range(3):
        if abs(delta[i])<1e-12:
            if abs(p[i])>ext[i]:return None
        else:
            a,b=sorted([(-ext[i]-p[i])/delta[i],(ext[i]-p[i])/delta[i]]);low=max(low,a);high=min(high,b)
            if low>high:return None
    return [float(low),float(high)] if low<.99 and high>.01 else None

def occlusion_warnings(ego,target,rig,env):
    m=np.asarray(ego['geometry']['hypothetical_actor_matrix']);point=target['geometry']['hypothetical_vertical_only_box']['center_world_m'];views=visibility(m,target['geometry']['hypothetical_vertical_only_box'],rig);rows=[]
    for v in views['views']:
        if not v['center_in_frustum']:continue
        c=rig['cameras'][v['camera']];start=(m@np.r_[c['xyz_m'],1])[:3];possible=[]
        for e in env:
            if e['semantic_tags'][0] in [1,2,10,11,23,24,25,26,27]:continue
            interval=segment_box_interval(start,point,e)
            if interval:possible.append(dict(entity_id=e['entity_id'],name=e['name'],semantic_tags=e['semantic_tags'],segment_interval=interval,interpretation='OBBLineOfSightWarning_NotMeshOcclusion'))
        rows.append(dict(camera=v['camera'],origin_world_m=start.tolist(),target_world_m=point,possible_static_occluders=possible,actual_visibility='Unknown_NoRenderedImages'))
    return rows

def main():
    cfg=json.loads(Path('configs/m8h0/protocol.json').read_text());vehicles=json.loads((ROOT/'vehicle_inventory_full.json').read_text())['records'];people=json.loads((ROOT/'navigation_inventory_full.json').read_text())['records'];history=json.loads(HISTORY.read_text());freeze=json.loads((ROOT/'m8g_freeze.json').read_text());rig=yaml.safe_load(Path('configs/cameras.yaml').read_text());env=json.loads((ROOT/'environment_inventory.json').read_text())['objects']
    eligible=[r for r in vehicles if r['geometry']['status']=='GeometryCandidateProxySupported'];peligible=[r for r in people if r['geometry']['status']=='GeometryCandidateProxySupported'];shortlist=[]
    for ego in eligible:
        options={}
        for cat,rs in [('vehicle',eligible),('person',peligible)]:
            opts=[]
            for r in rs:
                v=next((v for v in r['nominal_20_40m_observation_options'] if v['ego_spawn_id']==ego['id']),None)
                if v is None:continue
                pair=pair_relation(ego['geometry']['hypothetical_vertical_only_box'],r['geometry']['hypothetical_vertical_only_box'])
                if not pair['pair_clearance_pass']:continue
                # Fixed proximity-to-30m ranking; no model/visibility outcomes used.
                opts.append((abs(v['planar_distance_m']-30),r['id'],r,v))
            options[cat]=min(opts,key=lambda x:(x[0],x[1])) if opts else None
        shortlist.append(dict(ego_id=ego['id'],vehicle_id=None if options['vehicle'] is None else options['vehicle'][1],person_id=None if options['person'] is None else options['person'][1],score=None if not all(options.values()) else options['vehicle'][0]+options['person'][0],both_source_candidates=all(options.values())))
    chosen=[]
    for s in sorted([s for s in shortlist if s['both_source_candidates']],key=lambda x:(x['score'],x['ego_id'])):
        ego=next(r for r in vehicles if r['id']==s['ego_id']);xy=np.array(ego['transform']['xyz'][:2])
        if any(np.linalg.norm(xy-np.array(t['ego']['transform']['xyz'][:2]))<50 for t in chosen):continue
        vehicle=next(r for r in vehicles if r['id']==s['vehicle_id']);person=next(r for r in people if r['id']==s['person_id']);v=visibility(ego['geometry']['hypothetical_actor_matrix'],vehicle['geometry']['hypothetical_vertical_only_box'],rig);p=visibility(ego['geometry']['hypothetical_actor_matrix'],person['geometry']['hypothetical_vertical_only_box'],rig)
        pairs={a['id']+'__'+b['id']:pair_relation(a['geometry']['hypothetical_vertical_only_box'],b['geometry']['hypothetical_vertical_only_box']) for a,b in [(ego,vehicle),(ego,person),(vehicle,person)]}
        good=all(r['pair_clearance_pass'] for r in pairs.values());hist={r['id']:dict(previous_inventory_preflight=r['historical_preflight'],before_template_selection=True) for r in [ego,vehicle,person]}
        chosen.append(dict(id='m8h0_template_%02d'%len(chosen),type=template_family(v,p),score=s['score'],ego=ego,vehicle=vehicle,person=person,nominal_views=dict(vehicle=v,person=p),full_obb_pairs=pairs,historical_preflight=hist,camera_history=camera_region_history(ego,history),status='CuratedTemplateCandidateGo' if good else 'TemplateGeometryPartial',source_scope='Native fixed-coordinate Development proposal only; not spawned or approved for training',independent_split_eligible=False,cone='StaticFacilityPlacementNotReady',barrier='StaticFacilityPlacementNotReady',unknowns=['Physical settling/contact and actual full footprint','Camera occlusion and rendered object identity','Background instance identity and missing static labels','Whole-volume mesh clearance and unsampled narrow hazards','Shifted pose not allocated; no counterfactual claim']))
        if len(chosen)==cfg['template_limit']:break
    # Templates frozen BEFORE any additional visibility-proxy review. No replacements after warnings.
    write(ROOT/'frozen_templates_before_review.json',dict(**provenance('OfflineFiniteNativeTemplateFreeze'),shortlist=shortlist,templates=chosen,inventory_vehicle=asset(ROOT/'vehicle_inventory_full.json'),inventory_navigation=asset(ROOT/'navigation_inventory_full.json'),geometry_contract=asset('configs/m8f/contract.json'),no_model_results=True))
    warnings=[]
    for t in chosen:
        warnings.append(dict(template=t['id'],vehicle=occlusion_warnings(t['ego'],t['vehicle'],rig,env),person=occlusion_warnings(t['ego'],t['person'],rig,env),replacement_after_review=False))
    # Explicit layout correlation in ego frame; same vehicle/person assets are shared.
    relations=[]
    for i,t in enumerate(chosen):
        for u in chosen[i+1:]:
            distance=float(np.linalg.norm(np.array(t['ego']['transform']['xyz'][:2])-u['ego']['transform']['xyz'][:2]));relations.append(dict(first=t['id'],second=u['id'],ego_distance_m=distance,same_blueprints=True,same_layout_family='One native vehicle + one sidewalk person',independent_distribution=False))
    oldg=json.loads(Path('configs/m8f/protocol.json').read_text())['groups'];historical_layout=[]
    for t in chosen:
        for g in oldg:
            xy=np.array(t['ego']['transform']['xyz'][:2]);historical_layout.append(dict(template=t['id'],historic_group=g['group'],ego_distance_m=float(np.linalg.norm(xy-np.array(g['ego']['xyz'][:2]))),layout_relation='Same map/vehicle/person family; changed poses do not create blind data',historical_FG_never_relabelled=True))
    # Same-neighborhood read-only stratification avoids claiming a strict A/B causal improvement.
    strata=[]
    for g in oldg:
        within=[r for r in vehicles if np.linalg.norm(np.array(r['transform']['xyz'][:2])-g['ego']['xyz'][:2])<=35]
        strata.append(dict(historical_group=g['group'],native_within_35m=len(within),native_geometry_supported=sum(r['geometry']['status']=='GeometryCandidateProxySupported' for r in within),ids=[r['id'] for r in within],comparison='Native single-pose vs M8G both-phase different locations; descriptive only'))
    full=dict(**provenance('OfflineCuratedReview_NoNewQueries'),freeze=asset(ROOT/'frozen_templates_before_review.json'),templates=chosen,occlusion_warnings=warnings,relations=relations,old_layout_relations=historical_layout,source_status='NativeCandidateSourceReviewed',template_status='CuratedTemplateCandidateGo' if sum(t['status']=='CuratedTemplateCandidateGo' for t in chosen)>=2 else 'NativeSourceNotSufficient',proposal_status='DevelopmentSmokeProposalEligible' if sum(t['status']=='CuratedTemplateCandidateGo' for t in chosen)>=2 else 'ArchitecturePivotRequired',proposal_not_execution_approval=True,scope='Geometry-only candidate templates; visibility warnings must be resolved in separately approved Smoke; do not replace candidates in this stage')
    write('reports/m8h0_curated_template_candidates.json',full)
    write('reports/m8h0_geometry_and_overlap_audit.json',dict(**provenance('ActualQueriesAndOfflineProxyReview'),vehicle_raw=asset(ROOT/'vehicle_inventory_full.json'),navigation_raw=asset(ROOT/'navigation_inventory_full.json'),static_inventory=asset(ROOT/'environment_inventory.json'),template_pairs=[dict(template=t['id'],pairs=t['full_obb_pairs']) for t in chosen],visibility_proxy=warnings,near_M8FG_strata=strata,original_validator_thresholds=asset('configs/m8f/contract.json'),no_rule_relaxation=True,initial_z_and_final_z_distinct=True,actual_final_actor_state='NotRun',background_annotation='Static inventory and approximate OBBs available; render identity/whole-scene annotation NotRun'))
    write('reports/m8h0_historical_preflight.json',dict(**provenance('PreTemplateHistoricalSpaceCheck'),M8G_freeze=asset(ROOT/'m8g_freeze.json'),index=asset(HISTORY),records=16562,source_count=freeze['history_source_count'],near_neighbor_regression=freeze['regression'],templates=[dict(id=t['id'],target_checks=t['historical_preflight'],camera_history=t['camera_history']) for t in chosen],template_relations=relations,historical_layout_relations=historical_layout,coverage='HistoricalCoverageIncomplete',map_version_equivalence='Same Town name insufficient; prior0.9.15/0.9.16 records conservatively included, mesh identity not assumed',new_Final=0,independent_validation=False,training_qualification=False))
    print('shortlisted Ego',sum(s['both_source_candidates'] for s in shortlist),'frozen templates',[(t['id'],t['ego']['id'],t['vehicle']['id'],t['person']['id'],t['status']) for t in chosen],flush=True)
if __name__=='__main__':main()
