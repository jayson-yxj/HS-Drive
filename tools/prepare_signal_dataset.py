"""Validate controlled signal scenes and create a scene/map-isolated supervision index."""
import argparse,subprocess,sys
from pathlib import Path
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',default='m4signal_v2')
    p.add_argument('--output',required=True);p.add_argument('--registry',required=True);p.add_argument('--audit-prefix',required=True)
    a=p.parse_args();scenes=[]
    for town,split in [('town01','train'),('town04','val'),('town02','test')]:
        run=a.prefix+'_'+town;scene='/root/gpufree-data/hs-drive-data/raw/'+run
        subprocess.run([sys.executable,'-m','tools.validate_signal_capture','--scene',scene,'--output',a.audit_prefix+'_'+town],check=True)
        scenes.append(dict(scene_id=run,path='raw/'+run,group_id='signals_'+town,split=split))
    config=dict(schema_version='hsd-registry-v1',dataset_version='hsd-signal-probes-v1',raw_root='/root/gpufree-data/hs-drive-data',
        depth_evidence='../reports/depth_convention.json',allowed_profiles=['training-960-v4'],
        scope='Controlled stationary signal supervision diagnostics; no learned traffic classifier or driving claim',
        split_policy=dict(seed=7,val_fraction=.2,test_maps=['Town02'],min_cross_split_distance_m=100.),scenes=scenes)
    path=Path(a.registry)
    if path.exists():raise FileExistsError(path)
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    subprocess.run([sys.executable,'-m','tools.build_dataset_index','--registry',a.registry,'--output',a.output],check=True)

if __name__=='__main__':main()
