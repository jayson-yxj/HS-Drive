"""Read-only 214-frame lineage, mask and frozen-asset revalidation. No GPU."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from datasets.m8j3_rgb_clip import CAMERAS, VIEW
from datasets.m8j5_supervision import hs_to_petr_center
from datasets.m8j5b_coverage import positive_eligibility, coverage_view
from tools.m8j3_validate_clip import RAW, read_annotation
from tools.m8j5bv_run import ROOT,PUBLIC

OLD = Path('/root/gpufree-data/hs-drive-private/m8j5b-coverage-supervision')
AUDIT = Path('/root/gpufree-data/hs-drive-private/m8j5a-raw-audit')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    os.umask(0o077)
    ROOT.mkdir(exist_ok=True,parents=True);PUBLIC.mkdir(exist_ok=True,parents=True)
    frozen = []
    for name in ('m8j5a', 'm8j5b'):
        manifest = Path('reports') / (name + '_release_manifest.json')
        data = json.loads(manifest.read_text())
        for row in data['files']:
            assert sha(row['path']) == row['sha256'], row['path']
        frozen.append(dict(manifest=str(manifest), sha256=sha(manifest), files_verified=len(data['files'])))
    for folder in (OLD, AUDIT):
        manifest = folder / 'artifact_manifest.json'
        data = json.loads(manifest.read_text())
        for row in data['files']:
            assert sha(folder / row['path']) == row['sha256'], row['path']
        frozen.append(dict(manifest=str(manifest), sha256=sha(manifest), files_verified=len(data['files'])))
    annotations = json.loads((AUDIT / 'raw_annotation_hashes.json').read_text())
    for row in annotations:
        assert sha(row['path']) == row['sha256']
    counts = Counter(); reasons = Counter(); classes = Counter(); states = Counter()
    ranges = defaultdict(Counter); views = defaultdict(Counter); per_frame = []; private = []
    distinct = defaultdict(set); mask_hashes = []
    for path in sorted((OLD / 'sidecar').glob('*.json')):
        side = json.loads(path.read_text()); ordinal = side['ordinal']
        frame = json.loads((AUDIT / 'frames' / (ordinal + '.json')).read_text())
        raw = read_annotation(RAW / 'anno' / (ordinal + '.json.gz'))
        # Preserve source order as well as values; no invented record or ID.
        assert [obj['raw'] for obj in frame['objects']] == raw['bounding_boxes']
        actors = {(obj['source_class'], obj['actor_id']): obj for obj in frame['objects']}
        target = [obj for obj in frame['objects'] if obj['source_class'] in ('vehicle', 'walker')]
        assert len(target) == len(side['objects'])
        assert not side['frame_negative_enabled'] and not side['safe_negative_regions']
        assert not side['training_authorized'] and side['timestamp_s'] is None
        counts['non_ego_records'] += sum(o['source_class'] != 'ego_vehicle' for o in frame['objects'])
        local = Counter()
        for obj in side['objects']:
            original = actors[(obj['source_class'], obj['actor_id'])]
            assert obj['source_record_sha256'] == hashlib.sha256(json.dumps(original['raw'], sort_keys=True).encode()).hexdigest()
            assert obj['corners_hs_m'] == original['converted'].get('corners_hs_m')
            recomputed = positive_eligibility(original, frame['render_regions'], CAMERAS)
            assert recomputed == obj['positive']
            qualified = recomputed['eligible']
            if qualified:
                assert obj['petr_box'] == hs_to_petr_center(original['legacy'])
                counts['positive_box_values_exact'] += 1
            counts['target_records_preserved'] += 1
            counts['target_geometry_valid'] += original['converted']['geometry_status'] == 'Valid'
            counts['positive_diagnostic'] += qualified
            counts['positive_with_multiple_supported_cameras'] += qualified and len(recomputed['supported_cameras']) > 1
            counts['positive_not_legacy_candidate'] += qualified and not original['legacy'].get('positive_supervision_candidate', False)
            category = obj['source_class']
            classes[category + '_records'] += 1; classes[category + '_positive'] += qualified
            states[original['source_state']] += 1
            distinct[category].add(obj['actor_id'])
            distance = float(np.linalg.norm(original['converted']['center_hs_m'][:2]))
            bucket = '0-10' if distance < 10 else '10-20' if distance < 20 else '20-40' if distance < 40 else '40-60' if distance < 60 else '60+'
            ranges[bucket]['source_object_frames'] += 1; ranges[bucket]['positive'] += qualified
            reasons.update(recomputed['reasons'])
            local['source_object_frames'] += 1; local['positive'] += qualified
            for camera in CAMERAS:
                projection = original['views'][camera]['raw_projection']
                v = views[camera]; v['source_object_view_observations'] += 1
                v['projected_geometry'] += projection.get('status') == 'Valid'
                v['positive_supported'] += qualified and camera in recomputed['supported_cameras']
                v['near_plane_clipped'] += bool(projection.get('near_plane_clipped'))
                legacy_view = original['legacy'].get('views', {}).get(camera, {})
                status = legacy_view.get('occlusion_state', 'Unresolved_NotPhysicalOcclusionTruth')
                v['occlusion_' + status] += 1
            private.append(dict(ordinal=ordinal, actor_id=obj['actor_id'], source_class=category,
                source_record_sha256=obj['source_record_sha256'], distance_m=distance,
                positive=qualified, reasons=recomputed['reasons'], supported_cameras=recomputed['supported_cameras']))
        maskpath = OLD / 'masks_v2' / (ordinal + '.npz')
        with np.load(maskpath, allow_pickle=False) as archive:
            mask = archive['ignore']; projection = archive['image_projection']
            assert mask.dtype == np.bool_ and mask.shape == (6, 512, 1408)
            assert np.isfinite(projection).all()
            for i, v in enumerate(side['views']):
                assert v['camera'] == CAMERAS[i]
                assert int(mask[i].sum()) == v['cropped_ignore_pixels']
                counts['original_pixels'] += v['stats']['pixels']
                counts['original_ignore_pixels'] += v['stats']['ignore_pixels']
                counts['cropped_pixels'] += mask[i].size
                counts['cropped_ignore_pixels'] += int(mask[i].sum())
        mask_hashes.append(dict(ordinal=ordinal, sha256=sha(maskpath)))
        per_frame.append(dict(ordinal=ordinal, **local, negative_certificates=0, frame_negative_enabled=False))
    cases = []
    for reference in json.loads((OLD / 'case_reproduction.json').read_text()):
        n = reference['ordinal']; cam = reference['camera']; token = reference['region']['render_token']; label = reference['region']['semantic']
        frame = json.loads((AUDIT / 'frames' / (n + '.json')).read_text())
        sem = cv2.imread(str(RAW / 'camera' / ('semantic_' + VIEW[cam]) / (n + '.png')), -1)
        ins = cv2.imread(str(RAW / 'camera' / ('instance_' + VIEW[cam]) / (n + '.png')), -1)
        result = coverage_view(sem, ins, frame['objects'], cam, dilation=4)
        tokens = ins[:, :, 1].astype(np.int32) + 256 * ins[:, :, 0].astype(np.int32)
        selected = (tokens == token) & (ins[:, :, 2] == label)
        assert int(selected.sum()) == reference['region']['pixels']
        assert result['ignore'][selected].all()
        region = next(r for r in result['records'] if r['render_token'] == token and r['semantic'] == label)
        assert region == reference['region']
        with np.load(OLD / 'masks_v2' / (n + '.npz'), allow_pickle=False) as archive:
            crop = cv2.resize(result['ignore'].astype(np.float32), (1408, 792), interpolation=cv2.INTER_AREA)[280:792] > 0
            assert np.array_equal(crop, archive['ignore'][CAMERAS.index(cam)])
        cases.append(dict(ordinal=n, camera=cam, semantic=label, render_token=token,
            pixels=int(selected.sum()), ignored_pixels=int(result['ignore'][selected].sum()),
            semantic_agreement=region['semantic_agreement'], raw_box_support_fraction=region['raw_support_fraction'],
            status='MissingSourceSupportCandidate_Unknown_NotNewBox',
            full_mask_recomputed_equal=True, actor_identity='Unknown_NotRenderToken',
            frame_negative_enabled=False, invented_boxes=0))
    assert len(per_frame) == 214 and counts['non_ego_records'] == 3741 and counts['target_records_preserved'] == 1276
    assert counts['positive_diagnostic'] == 570 and len(cases) == 4
    save(ROOT / 'four_cases_private.json', cases)
    public_cases = [{k:v for k,v in row.items() if k != 'render_token'} for row in cases]
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), evidence='ActualCPU_PrivatePayloadAndFrozenSidecars_ReadOnly',
        frames=214, counts=dict(counts), per_class=dict(classes), per_source_state=dict(states),
        distinct_recorded_actor_ids={k:len(v) for k,v in distinct.items()},
        independence='Repeated object-frame observations in one correlated clip; six views never count as six objects. IDs do not identify unboxed render regions.',
        distance_bins_m=dict(ranges), per_camera_view_observations=dict(views), exclusion_reasons_nonexclusive=dict(reasons),
        original_image_ignore_fraction=counts['original_ignore_pixels']/counts['original_pixels'],
        cropped_image_ignore_fraction=counts['cropped_ignore_pixels']/counts['cropped_pixels'],
        four_known_gaps=public_cases, negative_certificate_count=0, negative_enabled_frames=0,
        safe_negative_status='NotAvailable_2DSemanticBackgroundNot3DCertificate',
        static_vehicle_actual_support='NotAvailable_NoSourceStaticVehicleRecords',
        original_annotation_hashes_verified=len(annotations), frozen_manifests_verified=frozen,
        manual_review='Pending_Suspended', real_backward_calls=0, optimizer_steps=0)
    save(ROOT / 'all214_object_lineage.json', private)
    save(ROOT / 'all214_frame_summary.json', per_frame)
    save(ROOT / 'all214_mask_hashes.json', mask_hashes)
    save(PUBLIC/'m8j5bv_real_coverage_validation.json', result)
    print(json.dumps(dict(frames=214, counts=dict(counts), real_negative_certificates=0, cases=4)))


if __name__ == '__main__':
    main()
