"""Additional denominator, source-ID, temporal and spatial leakage audits."""
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from tools.m8e_common import *
from datasets.m8e_reference import render_grade

p=json.loads(Path('configs/m8e/protocol.json').read_text())
rows=[];allframes=[];grade_diff=[]
for g in p['groups']:
    a=json.loads((ROOT/'audit'/(g['group']+'.json')).read_text())
    for f in a['frames']:
        allframes.append((g,f))
        for r in f['objects']:
            new=render_grade(r['views'])
            if new!=r['identity_grade']:grade_diff.append(dict(group=g['group'],phase=f['phase'],entity=r['entity_key'],old=r['identity_grade'],new=new))
            if f['phase']=='present':rows.append(r)
assert not grade_diff
present=[r for r in rows if r.get('is_controlled_target')]
unlinked=[]
for g,f in allframes:
    unlinked.append(dict(group=g['group'],phase=f['phase'],regions=len(f['unlinked_render_regions']),
        pixels=sum(v['native_pixels'] for v in f['unlinked_render_regions']),
        semantic_ids=dict(Counter(str(v['semantic_id']) for v in f['unlinked_render_regions']))))
# Search existing M5-O supervision target world centers, not just ego poses.
old={}
for split in ['train','val','calibration','sealed']:
    path=Path('runs/m5o_cache_'+split+'/supervision.json')
    for a in json.loads(path.read_text())['actors']:
        key=(a['group'],a['actor_id'],a['frame_id'])
        old[key]=dict(map=a['group'].split('_')[0],group=a['group'],center=a['box_center_world_carla'])
distances=[]
for g in p['groups']:
    targets=[r for r in present if r['group']==g['group']]
    references=[x for x in old.values() if x['map']==g['map']]
    nearest=min([(float(np.linalg.norm(np.array(r['center_world_m'][:2])-np.array(o['center'][:2]))),r['entity_key'],o['group'])
                 for r in targets for o in references],default=(None,None,None))
    distances.append(dict(group=g['group'],nearest_historical_target_m=nearest[0],entity=nearest[1],historical_group=nearest[2]))
id_examples=[dict(group=r['group'],source=r['source'],entity_id=r['entity_id'],camera=v['camera'],
                  opaque_byte_swapped_encoding=v['render_encoding_id'],
                  UE_unique_low16=((v['render_encoding_id']&255)<<8)+(v['render_encoding_id']>>8))
             for r in rows for v in r['views'] if v['accepted']]
counts=Counter((r['petr_class'],r['identity_grade']) for r in present)
write('reports/m8e_annotation_completeness_supplement.json',dict(**provenance('OfflineAdditionalCompletenessAndSeparationAudit'),
    raw_unknown_regions=unlinked,present_class_grades={'/'.join(k):v for k,v in counts.items()},
    historical_target_separation=distances,distinct_camera_grade_changes=grade_diff,
    identity_namespace_examples=id_examples,
    actual_instance_equal_actorID=sum(x['entity_id']==x['UE_unique_low16'] for x in id_examples if x['source']=='Actor'),
    all_environment_sources_not_mesh=True,
    budget_stop='Stopped at frozen8positions/32frames: source incompleteness and far support gaps persist; do not repeat adjacent frames or choose new successful scenes to satisfy quota.',
    rule_changes_after_validation='Only distinct-camera count implementation correction; no grade changed, no thresholds changed.',
    caveat='Same road IDs recur within split; maps shared with historical project. New heldout only1location, not sufficient independent final model evaluation.'))
print('present targets',len(present),'gradechanges',len(grade_diff),'historical target distances',distances)
