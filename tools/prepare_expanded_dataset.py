"""Label complete recordings concurrently on CPU, then build a new immutable index."""
import argparse,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from simulator.recorder import file_hash


def expanded_registry(campaign_path,registry_path):
    campaign=yaml.safe_load(Path(campaign_path).read_text())
    original=yaml.safe_load(Path('configs/dataset_v1.yaml').read_text())
    original.update(dataset_version='hsd-dataset-v2-driving-static-lanes',
        scope='Independent short TM drives; static lane labels only; no traffic interaction benchmark',
        drivable_bev_config='bev.yaml')
    original['split_policy']['test_maps'].append('Town05')
    for r in campaign['runs']:
        if not r.get('include_training',True):continue
        original['scenes'].append(dict(scene_id=r['run_id'],path='raw/'+r['run_id'],
            group_id=r['map'].lower()+'_tm_routes',split=r['split']))
    for s in original['scenes']:s['drivable_labels']='labels_static_v1/'+s['scene_id']
    path=Path(registry_path)
    if path.exists() and yaml.safe_load(path.read_text())!=original:raise FileExistsError(path)
    if not path.exists():path.write_text(yaml.safe_dump(original,sort_keys=False))
    return original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',default='configs/capture_expanded.yaml')
    p.add_argument('--registry',default='configs/dataset_v2.yaml')
    p.add_argument('--output',default='/root/gpufree-data/hs-drive-data/processed/hsd-dataset-v2-driving')
    a=p.parse_args();registry=expanded_registry(a.campaign,a.registry);raw=Path(registry['raw_root'])
    def task(scene):
        source=raw/scene['path'];target=raw/scene['drivable_labels']
        # Bounded wait permits concurrently scheduled collection. Each subprocess logs independently.
        deadline=time.monotonic()+7200
        while True:
            manifest=source/'manifest.json'
            if manifest.exists():
                status=json.loads(manifest.read_text())['status']
                if status=='complete':break
                if status in ('failed','interrupted'):raise RuntimeError(f'Failed capture: {source}')
            if time.monotonic()>deadline:raise TimeoutError(source)
            time.sleep(10)
        if target.exists():
            m=json.loads((target/'manifest.json').read_text())
            if (m['raw_manifest_sha256']!=file_hash(manifest) or m['status']!='complete' or
                    m['bev_config_sha256']!=file_hash('configs/bev.yaml')):raise ValueError('Existing label source/geometry mismatch')
            return
        print('Labeling',scene['scene_id'],flush=True)
        with Path('logs','labels_'+scene['scene_id']+'.log').open('w') as log:
            subprocess.run([sys.executable,'-m','tools.build_drivable_labels','--scene',str(source),
                '--output',str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Labeled',scene['scene_id'],flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(task,registry['scenes']))
    subprocess.run([sys.executable,'-m','tools.build_dataset_index','--registry',a.registry,'--output',a.output],check=True)

if __name__=='__main__':main()
