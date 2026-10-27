"""Bounded offline supplements from completed M8-J8 caches; no model execution."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from training.m8j7_metrics import distribution, errors


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    with Path(path).open('x') as stream:
        stream.write(text + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = read('configs/m8j8_protocol.json')
    root = Path(protocol['prepared_root']).parent
    manifest = read(root / 'prepared/manifest.json')
    classes = manifest['training_classes']
    audit = read(root / 'residual_object_audit.json')
    # Verify the actual affected tensors with the frozen official routines on CPU.
    chain = []
    for case in audit['height_cases']:
        record = load(root / 'B/residual' / f'{case["frame"]:02}.npz')
        g = torch.from_numpy(record['gt'][case['GT_index']:case['GT_index']+1].copy())
        gt = LiDARInstance3DBoxes(g, box_dim=9)
        target = torch.cat((gt.gravity_center, gt.tensor[:, 3:]), dim=1)
        encoded = normalize_bbox(target, [-51.2,-51.2,-5,51.2,51.2,3])
        raw = torch.from_numpy(record['raw_bbox'][case['query_index']:case['query_index']+1].copy())
        decoded = denormalize_bbox(raw, None)
        decoded[:, 2] -= decoded[:, 5]/2
        box = LiDARInstance3DBoxes(decoded, box_dim=9)
        reference = torch.from_numpy(record['boxes'][case['prediction_index']].copy())
        checks = {
            'official_normalized_cz_error': abs(encoded[0,4].item()-case['training_encoded_cz']),
            'official_normalized_logh_error': abs(encoded[0,5].item()-case['training_encoded_logh']),
            'official_decode_full_box_max_error': (box.tensor[0]-reference).abs().max().item(),
            'corner_bottom_error': abs(box.corners[0,:,2].min().item()-case['display_bottom_z']),
            'corner_top_error': abs(box.corners[0,:,2].max().item()-case['display_top_z']),
        }
        assert max(checks.values()) < 1e-6, checks
        chain.append(dict(frame=case['frame'], **checks))

    records = {
        m: [load(root / m / 'mini_val' / f'{f["index"]:02}.npz') for f in manifest['frames']]
        for m in ('A', 'B')
    }
    comparison = read(root / 'mini_val_object_comparison.json')
    support = {c: set() for c in classes}
    values = {c: defaultdict(list) for c in classes}
    states = defaultdict(set)
    for row in comparison:
        a, b = records['A'][row['frame']], records['B'][row['frame']]
        j, ia, ib = row['GT_index'], row['A_prediction'], row['B_prediction']
        assert np.array_equal(a['gt'][j], b['gt'][j])
        c = classes[int(a['gt_labels'][j])]
        support[c].add(row['instance_token'])
        state = ('common' if ib is not None else 'A_only') if ia is not None else ('B_only' if ib is not None else 'neither')
        states[state].add(row['instance_token'])
        if ia is None or ib is None:
            continue
        ea = errors(a['boxes'],a['gt'],[(ia,j)],a['gt_labels'],classes)
        eb = errors(b['boxes'],b['gt'],[(ib,j)],b['gt_labels'],classes)
        for e, r, i in ((ea,a,ia),(eb,b,ib)):
            for k, dim in [('width_abs_m',3),('length_abs_m',4),('height_abs_m',5)]:
                e[k] = [abs(float(r['boxes'][i,dim]-r['gt'][j,dim]))]
        for k in ea:
            if ea[k]:
                values[c][k+'_A'].extend(ea[k])
                values[c][k+'_B'].extend(eb[k])
                values[c][k+'_B_minus_A'].append(eb[k][0]-ea[k][0])
    heights = {x['instance_token'] for x in audit['height_cases']}
    missed = {x['instance_token'] for x in audit['misses']}
    duplicates = {x['instance_token'] for x in audit['duplicate_predictions']}
    tails = read(root / 'mini_val_xy_tails.json')
    result = dict(
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_type='OfflineCacheDiagnosticsAndOfficialCPUDecode_NoParameterUpdate',
        actual_residual_official_chain=chain,
        residual_instances=dict(height=len(heights),missed=len(missed),duplicate=len(duplicates),
            union=len(heights|missed|duplicates),height_and_missed=len(heights&missed),
            height_and_duplicate=len(heights&duplicates),missed_and_duplicate=len(missed&duplicates)),
        independent_GT_instances_by_class={c:len(v) for c,v in support.items()},
        matched_status_unique_instances={c:len(v) for c,v in states.items()},
        instance_count_caveat='One physical instance may have different detection status in different frames; these status sets are not disjoint.',
        paired_errors_by_class={c:{k:distribution(v) for k,v in rows.items()} for c,rows in values.items()},
        largest_matched_XY_errors={m:[{k:x[k] for k in ('frame','class_name','xy_m')} for x in tails if x['model']==m][:5] for m in ('A','B')},
        tail_caveat='Conditional on fixed same-class XY<2m match; larger errors remain FN/FP, not included in matched-error summaries.',
        private_inputs_sha256={str(root/f):sha(root/f) for f in ('residual_object_audit.json','mini_val_object_comparison.json','mini_val_xy_tails.json')},
    )
    write('reports/m8j8_diagnostic_supplement.json', result)
    rows = audit['height_cases']
    numeric_keys = [k for k in rows[0] if k.startswith(('training_','network_','decoded_','box_object_','display_','delta_'))]
    page = '<meta charset="utf-8"><h1>M8-J8 private residual provenance</h1><p>AI diagnostic review; no human sign-off. GT green, normal decoded predictions orange. No training.</p>'
    page += '<table border="1"><tr><th>field</th>'+''.join(f'<th>frame {r["frame"]}</th>' for r in rows)+'</tr>'
    for key in numeric_keys:
        page += '<tr><th>'+html.escape(key)+'</th>'+''.join(f'<td>{r[key]:.9f}</td>' for r in rows)+'</tr>'
    page += '</table><p>Original GT, converted tensors, raw query outputs and identities: <a href="residual_object_audit.json">private audit JSON</a>.</p>'
    for frame in (0,1,2,3):
        page += f'<h2>Residual frame {frame}</h2>'
        for m in ('A','B'):
            page += f'<h3>{m}</h3><img width="1500" src="{m}/residual/{frame:02}_six.jpg"><img width="700" src="{m}/residual/{frame:02}_bev.jpg">'
    with (root/'residual_comparison.html').open('x') as stream:
        stream.write(page)
    print(json.dumps(dict(status='OfflineSupplementCompleted', official_chain_cases=len(chain), residual_unique_instances=len(heights|missed|duplicates))))


if __name__ == '__main__':
    main()
