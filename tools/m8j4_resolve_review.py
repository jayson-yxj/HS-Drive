"""Append-only conservative review: quarantine ambiguous render associations."""
import copy
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from tools.m8j4_sidecar import OUT,save


def main():
    summary=json.loads((OUT/'sidecar_summary.json').read_text())
    conflicts=set(summary['render_token_cross_actor_conflicts'])
    final=OUT/'reviewed_sidecar';final.mkdir(exist_ok=False)
    stats=Counter();classes=defaultdict(Counter);range_status=defaultdict(Counter)
    evidence=[];withdrawn=[];crossview=[]
    for path in sorted((OUT/'sidecar').glob('*.json')):
        frame=json.loads(path.read_text())
        for o in frame['objects']:
            tokens={v['match']['render_token'] for v in o['views'].values() if v.get('match')}
            inconsistent=len(tokens)>1
            if inconsistent:crossview.append(dict(ordinal=frame['ordinal'],actor_id=o['actor_id'],render_tokens=sorted(tokens)))
            for cam,v in o['views'].items():
                m=v.get('match')
                if m and (f"{m['semantic']}:{m['render_token']}" in conflicts or inconsistent):
                    withdrawn.append(dict(ordinal=frame['ordinal'],actor_id=o['actor_id'],camera=cam,
                        render_token=m['render_token'],reason='CrossFrameActorConflict' if not inconsistent else 'CrossCameraRenderConflict'))
                    v.update(status='Unknown',reason='RenderIdentityConflict_ReviewRequired',match=None,withdrawn_match=m)
            valid=[v for v in o['views'].values() if v['status']=='Valid']
            if o['label_status']=='Valid' and not valid:
                o.update(label_status='Unknown',reason='AllSupportingViewsQuarantined')
            o['positive_supervision_candidate']=o['label_status']=='Valid' and o.get('yaw_only_comparable',False) and o.get('model_class') in ('car','pedestrian')
            stats['objects_'+o['label_status']]+=1
            classes[o.get('model_class',o['source_class'])][o['label_status']]+=1
            if o['geometry_status']=='Valid':
                stats['supported_class_object_'+o['label_status']]+=1
                dist=o['distance_xy_m'];band='0-10' if dist<10 else '10-20' if dist<20 else '20-40' if dist<40 else '40+'
                range_status[band][o['label_status']]+=1
                for v in o['views'].values():
                    stats['all_actor_views_'+v['status']]+=1
                    if v['projection']['status']=='Valid' and o['reason']!='HiddenBelowGround_NoTrainingTarget':
                        stats['in_image_actor_views_'+v['status']]+=1
            if o['positive_supervision_candidate']:
                stats['positive_candidates']+=1
                evidence.append(dict(ordinal=frame['ordinal'],actor_id=o['actor_id'],model_class=o['model_class']))
        if any(o['positive_supervision_candidate'] for o in frame['objects']):stats['positive_frames']+=1
        frame.update(review_version='m8j4.identity-quarantine.v1',
                     human_review='Pending_VisualAuditIsNotPixelCertification')
        save(final/path.name,frame)
    save(OUT/'association_quarantine.json',dict(conflicting_tokens=sorted(conflicts),withdrawn=withdrawn,
        cross_camera_inconsistent=crossview,
        root_cause='Unresolved: geometry ambiguity, hidden actor changes, renderer token reuse/truncation cannot be separated from payload alone',
        initial_results_preserved=True,thresholds_unchanged=True))
    result=dict(status='BasicSidecarInterfaceGo_QualityPartial',frames=214,anomalies=23,
        active_ambiguous_matches=0,withdrawn_view_matches=len(withdrawn),
        original_ambiguities_retained=True,stats=dict(stats),
        class_status={k:dict(v) for k,v in classes.items()},distance_status={k:dict(v) for k,v in range_status.items()},
        valid_semantics='Geometry and conservative visible-region evidence only; no human certification',
        invalid_semantics='Explicit malformed or underground target; view Invalid may only mean out of FOV',
        unknown_semantics='No reliable association or unsupported class; never background negative',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_forward_gate='Only limited frozen engineering forward; not label/training quality approval')
    save(OUT/'reviewed_summary.json',result)
    save(OUT/'positive_candidate_index.json',evidence)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
