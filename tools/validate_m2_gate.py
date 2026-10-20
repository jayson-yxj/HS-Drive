"""Close only the declared M2 engineering gate, referencing immutable evidence."""
import argparse,json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/m2_gate.json');a=p.parse_args()
    files={'recording':'reports/m2_town03_recording.json',
           'motion':'reports/m2_town03_motion_geometry.json',
           'slope':'reports/m2_town03_slope_geometry.json',
           'occlusion':'reports/m2_occlusion.json',
           'actors':'reports/m2_dynamic_actor_surfaces.json',
           'cleanup':'reports/m2_interrupt.json',
           'render':'reports/m2_render.json','depth':'reports/depth_convention.json'}
    data={k:json.loads(Path(v).read_text()) for k,v in files.items()}
    checks={'town03_render':data['render']['passed'],
        'thirty_second_recording':data['recording']['m1_automated_checks_passed'],
        'ego_motion':data['motion']['motion_passed'],'six_camera_overlap':data['motion']['cross_view_passed'],
        'sloped_geometry':data['slope']['cross_view_passed'],
        'controlled_occlusion':data['occlusion']['passed'],
        'dynamic_actor':data['actors']['dynamic_target_passed'],
        'self_body_audit':data['actors']['self_body_review_passed'],
        'interrupt_cleanup':data['cleanup']['passed'],
        'depth_semantics':data['depth']['convention']=='optical_z'}
    result={'schema_version':'hsd-m2-gate-v1','checks':checks,'passed':all(checks.values()),
        'evidence':{k:{'path':v,'sha256':file_hash(v)} for k,v in files.items()},
        'scope':'CARLA 0.9.15 Epic / Town03 / configured six-camera rig; Oracle geometry engineering gate',
        'exclusions':['learned perception','full occupancy/free-space','closed-loop safety','all weather/maps','real-time throughput']}
    atomic_json(a.output,result);print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
