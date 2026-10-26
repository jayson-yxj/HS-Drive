"""Read-only re-verification of authorized private clip and RGB-only packets."""
import hashlib
import json
from pathlib import Path
from datasets.m8j3_rgb_clip import load_rgb_packet
from tools.m8j3_download_clip import PRIVATE, NAME, SIZE, SHA, REV


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    archive=PRIVATE/'download'/NAME
    assert archive.stat().st_size==SIZE and sha(archive)==SHA
    root=PRIVATE/'extracted'/NAME[:-7]
    manifest=json.loads((PRIVATE/'validation/private_payload_manifest.json').read_text())
    for r in manifest:
        p=(root/r['path']).resolve()
        assert p.is_relative_to(root.resolve()) and p.is_file()
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    a=json.loads((PRIVATE/'validation/payload_validation.json').read_text())
    assert not a['image_decode_failures']
    assert a['camera_ego_matrix_residual_max']<2e-3 and a['intrinsic_fov_residual_max']<1e-8
    assert all(x['ordinal_set_matches'] for k,x in a['groups'].items() if k != 'expert_assessment')
    assert not a['groups']['expert_assessment']['ordinal_set_matches']
    assert sorted(p.stem for p in (root/'expert_assessment').glob('*.npz')) == ['-0001'] + [f'{i:05}' for i in range(213)]
    assert a['timestamp_frame_version_keys_found']==[]
    for ordinal in a['selected_RGB_packets']:
        packet=json.loads((PRIVATE/'rgb_only'/f'{ordinal}.json').read_text())
        assert packet['revision']==REV
        out=load_rgb_packet(PRIVATE/'rgb_only',f'{ordinal}.json')
        assert out['images'].shape==(6,900,1600,3)
        assert out['timestamps'] is None and not out['temporal_valid']
    print(json.dumps(dict(status='PrivatePayloadReverified',archive_sha256=SHA,
        extracted_files=len(manifest),rgb_packets=len(a['selected_RGB_packets']),
        expert_assessment_alignment='Failed_Quarantined_NoAutomaticShift',
        model_forward=False,temporal_certified=False,training_ready=False)))


if __name__=='__main__':
    main()
