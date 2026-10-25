"""Check actual renderer IDs disappearing in empty/removed images, not just moved OBBs."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m8e_common import *

records=[]
for g in json.loads(Path('configs/m8e/protocol.json').read_text())['groups']:
    audit=json.loads((ROOT/'audit'/(g['group']+'.json')).read_text())
    byphase={f['phase']:f for f in audit['frames']}
    arrays={}
    for phase,f in byphase.items():
        side=json.loads(Path(f['sidecar']['path']).read_text())
        arrays[phase]={}
        for name,files in side['sensors'].items():
            with np.load(files['instance']['path'],allow_pickle=False) as z:
                arrays[phase][name]=(z['semantic'].copy(),z['instance'].copy())
    for obj in byphase['present']['objects']:
        if not obj.get('is_controlled_target') or not obj['identity_grade'].startswith('RenderLinked'):continue
        evidence=[]
        for v in obj['views']:
            if not v['accepted']:continue
            rid=v['render_encoding_id'];tag=v['semantic_id'];camera=v['camera']
            counts={phase:int(((tags==tag)&(ids==rid)).sum()) for phase in arrays
                    for tags,ids in [arrays[phase][camera]]}
            evidence.append(dict(camera=camera,semantic_id=tag,render_encoding_id=rid,
                                  image_wide_pixel_counts=counts,
                                  disappearance_verified=counts['empty']==0 and counts['removed']==0))
        shifted=next((r for r in byphase['shifted']['objects'] if r['entity_key']==obj['entity_key']),None)
        records.append(dict(group=g['group'],entity_key=obj['entity_key'],class_name=obj['petr_class'],
            range_m=obj['range_m'],evidence=evidence,
            all_disappearance_pass=all(v['disappearance_verified'] for v in evidence),
            shifted_grade=shifted['identity_grade'] if shifted else 'UnlabeledOrUnknown',
            shifted_source_match=shifted is not None and shifted['entity_id']==obj['entity_id']))
write('reports/m8e_counterfactual_identity.json',dict(**provenance('ActualRenderedIDDisappearanceAudit'),
    records=records,qualifying_present_objects=len(records),
    all_pass=all(r['all_disappearance_pass'] for r in records),
    failures=[r for r in records if not r['all_disappearance_pass']],
    qualification='Supports same episode controlled identity. Not motion quality, not exact mesh, not human verification. Shift may occlude or leave view.'))
print('actual image ID disappearance',sum(r['all_disappearance_pass'] for r in records),'/',len(records))
