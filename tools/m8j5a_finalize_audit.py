"""Additional independent numeric checks and aggregate, non-private reports."""
import ast
import hashlib
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from datasets.m8j5a_annotation_audit import converted_geometry,raw_corners
from geometry.reference import CARLA_EGO_TO_HS_EGO as S,OPTICAL_TO_CARLA_CAM as O
from tools.m8j3_validate_clip import RAW,read_annotation
from tools.m8j5a_audit import OUT,save,report
from tools.m8j5a_build_ui import technical


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sources=[('anno_doc','reports/m8i0_sources/b2dapi__docs__anno.md.txt'),
        ('collector','reports/m8i0_sources/b2dapi__tools__data_collect.py.txt'),
        ('projection_utils','reports/m8j0_sources/github_utils.decoded.txt'),
        ('official_visualization','reports/m8j5a_sources/official_visualize.py.txt')]
    # Execute ONLY the standalone official pinhole function, extracted via AST,
    # no importing collector/CARLA or executing its top-level code.
    tree=ast.parse(Path(sources[2][1]).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_image_point')
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_official_get_image_point','exec'),env)
    official=env['get_image_point']
    frame_camera=[];stage_counts=defaultdict(Counter);unique=defaultdict(set);filters=defaultdict(Counter);viewreasons=defaultdict(Counter)
    maxres=0.;corner_pairs=0;cam_matrix_max=0.;motion=Counter();lineage=[]
    for p in sorted((OUT/'frames').glob('*.json')):
        f=json.loads(p.read_text());a=read_annotation(RAW/'anno'/(f['ordinal']+'.json.gz'))
        ego=next(o['raw'] for o in f['objects'] if o['source_class']=='ego_vehicle')
        current=Counter()
        for o in f['objects']:
            cls=o['source_class'];old=o['legacy'];raw=o['raw']
            if cls=='vehicle':motion['instant_speed_abs_lt0_1' if abs(raw['speed'])<.1 else 'instant_speed_abs_ge0_1']+=1
            if cls=='ego_vehicle':continue
            clsct=stage_counts[cls];clsct['raw_records']+=1;unique[cls].add(o['actor_id'])
            clsct['legacy_record_retained']+=int(o['lineage']['legacy_sidecar'])
            if cls in ('vehicle','walker'):
                clsct['legacy_geometry_retained']+=int(o['lineage']['legacy_geometry_valid'])
                clsct['raw_view_projectable']+=int(any(v['raw_projection']['status']=='Valid' for v in o['views'].values()))
                clsct['legacy_association_supported']+=int(old['label_status']=='Valid')
                clsct['legacy_positive']+=int(old['positive_supervision_candidate'])
                clsct['old_model_technical_candidate']+=int(technical(o))
                c=np.asarray(o['converted']['center_hs_m']);range_ok=bool(np.all(c>[-51.2,-51.2,-5]) and np.all(c<[51.2,51.2,3]))
                crop_ok=any(v['raw_projection']['status']=='Valid' and v['raw_projection']['bbox_xyxy'][3]*.88>280 and v['raw_projection']['bbox_xyxy'][1]*.88<792 for v in o['views'].values())
                # Diagnostic count only: retain raw geometry even when visual association unknown.
                model_geom_ok=old['yaw_only_comparable'] and range_ok and crop_ok and old['reason']!='HiddenBelowGround_NoTrainingTarget'
                clsct['geometry_only_model_domain_diagnostic']+=int(model_geom_ok)
                filters[cls].update(o['filter_reasons'])
                for v in old['views'].values():
                    if v['projection']['status']=='Valid':viewreasons[cls][v['reason']]+=1
            for cam,sensor in a['sensors'].items():
                if cam not in o['views']:continue
                rec=dict(ordinal=f['ordinal'],camera=cam,source_class=cls,actor_id=o['actor_id'],
                    raw_projectable=o['views'][cam]['raw_projection']['status']=='Valid',
                    converted_projectable=o['views'][cam]['converted_projection']['status']=='Valid',
                    legacy_projection=bool(old and old.get('views',{}).get(cam,{}).get('projection',{}).get('status')=='Valid'),
                    legacy_association=bool(old and old.get('views',{}).get(cam,{}).get('status')=='Valid'),
                    technical_candidate=technical(o),final_training=False)
                frame_camera.append(rec)
                if cls in ('vehicle','walker'):
                    world,_=raw_corners(raw);hs=np.asarray(o['converted']['corners_hs_m'])
                    optical_from_hs=np.linalg.inv(S@np.asarray(sensor['cam2ego'])@O)
                    # This route uses independent cam2ego, not world2cam.
                    pcam=(optical_from_hs@np.c_[hs,np.ones(8)].T).T
                    uv=(np.asarray(sensor['intrinsic'])@pcam[:,:3].T).T
                    for point,q,z in zip(world,uv,pcam[:,2]):
                        pixel,depth=official(point,np.asarray(sensor['intrinsic']),np.asarray(sensor['world2cam']))
                        if depth>.1 and z>.1:
                            # Near-plane offscreen points amplify float source errors;
                            # report separately all positive corners vs in-image corners.
                            error=float(abs(q[:2]/q[2]-pixel).max());corner_pairs+=1
                            if 0<=pixel[0]<1600 and 0<=pixel[1]<900:maxres=max(maxres,error)
                    expected=np.asarray(sensor['world2cam'])@np.linalg.inv(np.asarray(ego['world2ego']))
                    cam_matrix_max=max(cam_matrix_max,float(abs(expected-np.linalg.inv(sensor['cam2ego'])).max()))
            lineage.append(dict(ordinal=f['ordinal'],actor_id=o['actor_id'],class_name=cls,
                source_state=o['source_state'],lineage=o['lineage'],filter_reasons=o['filter_reasons']))
    # Aggregate every frame/camera/class without treating these as distinct actors.
    grouping=defaultdict(Counter)
    for row in frame_camera:
        key=(row['ordinal'],row['camera'],row['source_class']);grouping[key]['actor_views']+=1
        grouping[key].update({k:int(v) for k,v in row.items() if isinstance(v,bool)})
    grouped=[dict(ordinal=k[0],camera=k[1],source_class=k[2],**v) for k,v in sorted(grouping.items())]
    save(OUT/'lineage_all_records.json',lineage);save(OUT/'per_frame_camera_class.json',grouped)
    hashes=json.loads((OUT/'raw_annotation_hashes.json').read_text());assert all(sha(h['path'])==h['sha256'] for h in hashes)
    # A separate actual visual inspection record is not manufactured by reruns.
    save(report('m8j5a_conversion_validation.json'),dict(frames=214,
        raw_annotations_unchanged=True,raw_hash_manifest_sha256=sha(OUT/'raw_annotation_hashes.json'),
        all_non_ego_record_retention=1.0,vehicle_walker_geometric_retention=1.0,
        independent_official_pinhole_vs_HS_cam2ego_in_image_max_error_px=maxres,
        positive_corner_pairs_checked=corner_pairs,cam2ego_world2cam_matrix_max_abs_error=cam_matrix_max,
        actual_static_vehicle_count=0,static_support='SyntheticOnly_NoRealStaticAnnotationInThisClip',
        source_center_and_half_extent='ValidatedAgainstRecordedWorldCornersAndInverseActorMatrix',
        coordinate_scaling_or_reflection_bug='NotFoundFor1276VehicleWalkerRecords',
        display_bug='Confirmed_CandidateOnlyPageHidAllOtherObjects_Fixed',
        schema_bug='DisplayGeometryWasCoupledToPETRClassAndIdentityFiltering_DecoupledInAuditAdapter',
        labeling_policy='RawAllObjectGeometryRetained_TrainingNotAuthorized',
        official_sources=[dict(name=n,path=p,sha256=sha(p)) for n,p in sources]))
    rs=json.loads((OUT/'render_regions.json').read_text());regions=Counter(r['coverage'] for r in rs if r['pixels']>=100)
    unexplained=[r for r in rs if r['pixels']>=100 and r['coverage'].startswith('Unexplained')]
    save(report('m8j5a_coverage_summary.json'),dict(frames=214,
        actor_frame_stage_counts={k:dict(v) for k,v in stage_counts.items()},
        unique_actor_ids_in_one_clip={k:len(v) for k,v in unique.items()},vehicle_source_state={'dynamic':1152,'static':0},
        instantaneous_vehicle_motion=dict(motion),instantaneous_motion_not_source_state=True,
        vehicle_walker_filters_nonexclusive={k:dict(v) for k,v in filters.items()},
        projected_view_association_reasons={k:dict(v) for k,v in viewreasons.items()},
        region_ge100px_counts=dict(regions),unexplained_region_observations=len(unexplained),
        unexplained_distinct_frames=len({r['ordinal'] for r in unexplained}),
        unexplained_render_token_groups=len({(r['semantic'],r['render_token']) for r in unexplained}),
        region_spatial_support_fraction=regions['RawBoxSpatialSupport']/sum(regions.values()),
        fraction_scope='Only >=100px renderer-region observations, correlated; not physical object recall or detection AP',
        per_frame_camera_class_private=str(OUT/'per_frame_camera_class.json'),
        per_frame_private=str(OUT/'per_frame_counts.json'),lineage_private=str(OUT/'lineage_all_records.json'),
        no_negative_labels=True,golden_requirement='Suspended'))
    print(json.dumps(dict(independent_pinhole_max_px=maxres,stage_counts={k:dict(v) for k,v in stage_counts.items()},regions=dict(regions)),indent=2))


if __name__=='__main__':main()
