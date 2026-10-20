"""Supplementary CARLA collision-query reference, not exact rendered Mesh truth."""
import argparse,json
from pathlib import Path
import numpy as np
import carla
from geometry.reference import unproject_pixels_to_optical,transform_points
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',default='reports/m5e_new_visibility.json');p.add_argument('--index',default='/root/gpufree-data/hs-drive-data/processed/m5e_index');a=p.parse_args();dest=Path('reports/m5e_collision_visibility.json')
    if dest.exists():raise FileExistsError(dest)
    audit=json.loads(Path(a.audit).read_text());rows=json.loads(Path(a.index,'index.json').read_text())['samples'];out=[];client=carla.Client('127.0.0.1',2100);client.set_timeout(120)
    for group in sorted({r['group'] for r in audit['records']}):
        row=next(r for r in rows if r['group']==group);raw=Path(row['raw_frame']).parent.parent;man=json.loads((raw/'manifest.json').read_text());cal=json.loads((raw/'calibration.json').read_text())['cameras'];world=client.load_world(man['map'].split('/')[-1]);settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;world.apply_settings(settings);world.tick();counts={}
        try:
            for record in audit['records']:
                if record['group']!=group:continue
                with np.load(record['path']) as z:d={k:z[k] for k in z.files}
                frame=json.loads(next(Path(r['raw_frame']) for r in rows if r['group']==group and r['configuration']==record['configuration']).read_text());T=np.array(frame['T_world_from_ego']);camera=cal[record['camera']];origin=np.array(camera['T_carla_world_from_camera'])[:3,3]
                for j in np.flatnonzero(d['candidate']):
                    band='20_35' if d['distance'][j]>=20 else '10_20';key=(group,band)
                    if counts.get(key,0)>=50:continue
                    counts[key]=counts.get(key,0)+1;u,v=d['uv'][j];bg=transform_points(unproject_pixels_to_optical(np.array([u]),np.array([v]),np.array([d['depth_background'][j]]),camera['K'],'optical_z'),camera['T_hs_ego_from_opt'])[0];target=transform_points(d['points_hs'][j],T)[0]*[1,-1,1];bgworld=transform_points(bg,T)[0]*[1,-1,1];direction=bgworld-origin;distance=np.linalg.norm(direction);direction/=distance
                    hits=world.cast_ray(carla.Location(*origin),carla.Location(*(bgworld+direction*.5)));hitlist=[]
                    for hit in hits:
                        loc=np.array([hit.location.x,hit.location.y,hit.location.z]);dist=np.linalg.norm(loc-origin)
                        if dist>.05:hitlist.append(dict(distance_m=float(dist),label=str(hit.label),position=loc.tolist()))
                    hitlist.sort(key=lambda h:h['distance_m']);td=float(np.linalg.norm(target-origin));first=hitlist[0]['distance_m'] if hitlist else None
                    out.append(dict(group=group,configuration=record['configuration'],camera=record['camera'],uv=d['uv'][j].tolist(),band=band,target_distance_m=td,render_background_distance_m=float(distance),collision_first_distance_m=first,collision_before_target=bool(first is not None and first<td+.5),background_match_25cm=bool(first is not None and abs(first-distance)<=.25),hits=hitlist))
        finally:
            settings.synchronous_mode=False;settings.fixed_delta_seconds=None;world.apply_settings(settings)
        print(group,len(out),flush=True)
    atomic_json(dest,dict(kind='actual_CARLA_world_cast_ray_static_empty_layout',records=out,source_sha256=file_hash(__file__),candidate_promoted=False,limits='Static world collision query, no placedtarget/ego actors; renderer reference captured empty samepose. Collision geometry may differ from renderMesh; firsthit mismatch/missing is unresolved, never reliable negative. Fixed first50pergroupband sample, selection independent of outcome; not allcandidate census, no newtest.'))


if __name__=='__main__':main()
