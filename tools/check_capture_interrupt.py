"""Launch a real recording, SIGTERM after commits, verify resource restoration."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from simulator.recorder import atomic_json
from datasets.carla_dataset import CarlaDataset


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    root=Path('/root/gpufree-data/hs-drive-data/raw');scene=root/a.run_id
    if scene.exists():raise FileExistsError(scene)
    import carla
    c=carla.Client('localhost',2000);c.set_timeout(20);world=c.get_world()
    initial=world.get_settings();before={x.id for x in world.get_actors()}
    with Path('logs/'+a.run_id+'.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','tools.record_dataset','--frames','1000',
            '--output',str(root),'--run-id',a.run_id],stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Recorder exited before interruption')
                if len(list((scene/'frames').glob('*.json')))>=3:break
                time.sleep(.5)
            else:raise TimeoutError('No committed frames before deadline')
            process.terminate();exit_code=process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.terminate();process.wait(timeout=30)
    after=world.get_settings();actors=list(world.get_actors())
    leaked=[x.id for x in actors if x.id not in before and x.type_id.startswith(('vehicle.','sensor.'))]
    manifest=json.loads((scene/'manifest.json').read_text())
    replay=CarlaDataset(scene)
    for i in range(len(replay)):
        replay.read(i,verify=True)
    result={'exit_code':exit_code,'manifest_status':manifest['status'],
        'committed_frames':len(list((scene/'frames').glob('*.json'))),'committed_replay_verified':True,'leaked_actors':leaked,
        'sync_before':initial.synchronous_mode,'sync_after':after.synchronous_mode,
        'render_before':initial.no_rendering_mode,'render_after':after.no_rendering_mode,
        'passed':exit_code==130 and manifest['status']=='interrupted' and not leaked and initial.synchronous_mode==after.synchronous_mode and initial.no_rendering_mode==after.no_rendering_mode}
    atomic_json(a.output,result);print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
