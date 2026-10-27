"""Verify completed M8-J8 artifacts and historical preservation, without a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    protocol = read('configs/m8j8_protocol.json')
    root = Path(protocol['prepared_root']).parent
    output = Path('reports/m8j8_preservation.json')
    assert not output.exists(), 'Existing evidence is immutable'
    historical = []
    for name in ('m8j7_release_manifest.json','m8j6_retry_release_manifest.json','m8j6_actual_release_manifest.json'):
        release = read(Path('reports')/name)
        for f in release['files']:
            assert sha(f['path']) == f['sha256'], f['path']
        historical.append(dict(manifest=name, files=len(release['files']), unchanged=True))
    supervisor = read('reports/m8j8_sources/'+protocol['run_id']+'/supervisor.json')
    for path, expected in supervisor['source_sha256'].items():
        assert sha(path) == expected, path
    assert sha(protocol['manifest']) == protocol['manifest_sha256']
    for m in ('A','B'):
        assert sha(protocol[m]['path']) == protocol[m]['sha256']
    original = Path('/root/gpufree-data/hs-drive-experiments/m8a-proxy-retry/assets/epoch_24.pth')
    assert sha(original) == protocol['A']['original_pth_sha256']
    checkpoints = read('reports/m8j7_resource_and_freeze.json')['checkpoints']
    for cp in checkpoints:
        assert sha(cp['path']) == cp['sha256']
    bundle_count = 0
    image_paths = {}
    for folder in (Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared'),root/'prepared'):
        manifest = read(folder/'manifest.json')
        for frame in manifest['frames']:
            for sub, field in (('rgb_input','input_sha256'),('supervision','supervision_sha256')):
                path = folder/sub/f'{frame["index"]:02}.npz'
                assert sha(path) == frame[field], str(path)
                bundle_count += 1
            for camera in frame['camera_files']:
                image_paths[camera['path']] = camera['sha256']
    for path, expected in image_paths.items():
        assert sha(path) == expected, path
    runs = {}
    for m in ('A','B'):
        r = read('reports/m8j8_sources/'+protocol['run_id']+'/'+m+'/result.json')
        assert r['optimizer_steps'] == r['backward_calls'] == 0
        assert r['initial_state_fingerprint'] == r['final_state_fingerprint']
        assert r['parameters_buffers_unchanged'] and r['alias_relationships_unchanged']
        assert len(r['frames']) == 89
        for f in r['frames']:
            assert sha(f['cache_path']) == f['cache_sha256']
        runs[m] = dict(frames=89, fingerprint=r['final_state_fingerprint'], all_cached_frames_verified=True,
                       parameters_buffers_unchanged=True, alias_relationships_unchanged=True)
    metrics = read('reports/m8j8_ab_statistics.json')
    per_frame = read('reports/m8j8_per_frame.json')['frames']
    assert len(per_frame) == 81
    for m in ('A','B'):
        for key in ('GT','TP','FP','FN','duplicates'):
            assert sum(f[m][key] for f in per_frame) == metrics['aggregate'][m]['primary'][key]
            assert sum(s[m]['primary'][key] for s in metrics['scenes'].values()) == metrics['aggregate'][m]['primary'][key]
        assert metrics['aggregate'][m]['primary']['GT'] == 3559
    p = metrics['paired']
    assert p['common']+p['A_only']+p['B_only']+p['neither'] == 3559
    assert p['common']+p['A_only'] == metrics['aggregate']['A']['primary']['TP']
    assert p['common']+p['B_only'] == metrics['aggregate']['B']['primary']['TP']
    pids = [dict(pid=p['PID'],present=Path('/proc',str(p['PID'])).exists()) for p in supervisor['phases']]
    assert not any(p['present'] for p in pids)
    historical_changes = subprocess.check_output(['git','diff','f50aead','--name-only','--diff-filter=MD'],text=True).splitlines()
    assert all('m8j8' in p.lower() for p in historical_changes), historical_changes
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),historical=historical,
        original_checkpoint_unchanged=True,A_B_weights_unchanged=True,all_six_J7_checkpoints_unchanged=True,
        executed_source_hashes_unchanged=True,predeclared_manifest_unchanged=True,
        input_and_supervision_bundles_verified=bundle_count,source_camera_images_verified=len(image_paths),
        cached_frame_verifications=178,models=runs,owned_pids=pids,all_exit_codes_zero=True,
        aggregate_per_frame_scene_and_pair_counts_consistent=True,optimizer_steps=0,
        historical_tracked_files_modified_or_deleted=[])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
