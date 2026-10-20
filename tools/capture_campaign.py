"""Sequential six-camera campaign on a dedicated CARLA server; never overwrites."""
import argparse,json,subprocess,sys
from pathlib import Path
import yaml
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/capture_expanded.yaml')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());config_sha=file_hash(a.config)
    if cfg['schema_version']!='hsd-driving-campaign-v1':raise ValueError('Unknown campaign')
    for r in cfg['runs']:
        dest=Path(cfg['raw_root'])/r['run_id']
        if dest.exists():
            m=json.loads((dest/'manifest.json').read_text())
            if m['status']!='complete':raise RuntimeError(f'Existing incomplete capture: {dest}')
            if (m['map'].split('/')[-1],m['spawn_index'],m['seed'],m['frames_committed'],m['motion']) != (r['map'],r['spawn'],r['seed'],cfg['frames'],'autopilot'):
                raise ValueError('Existing capture differs from campaign')
            if abs(m['weather']['sun_altitude_angle']-r['sun'])>1e-5 or abs(m['speed_limit_mps']-cfg['speed_limit_mps'])>1e-5:
                raise ValueError('Existing weather/speed differs from campaign')
            print('Verified completed capture:',r['run_id'],flush=True);continue
        cmd=[sys.executable,'-m','tools.record_dataset','--capture-config',cfg['capture_config'],
             '--map',r['map'],'--load-map','--motion','autopilot','--spawn-index',str(r['spawn']),
             '--seed',str(r['seed']),'--sun-altitude',str(r['sun']),'--speed-limit',str(cfg['speed_limit_mps']),
             '--frames',str(cfg['frames']),'--output',cfg['raw_root'],'--run-id',r['run_id']]
        if 'tick_wall_delay' in r:cmd.extend(['--tick-wall-delay',str(r['tick_wall_delay'])])
        print('Starting:',r['run_id'],flush=True)
        with Path('logs',r['run_id']+'.log').open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(Path(cfg['raw_root'])/'campaign_complete.json',{
        'config_sha256':config_sha,'runs':[r['run_id'] for r in cfg['runs']]})

if __name__=='__main__':main()
