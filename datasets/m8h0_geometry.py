"""Pure read-only candidate geometry. No Actor creation or acceptance override."""
import numpy as np
from datasets.m8f_quality import polygon_distance

def make_box(actor_matrix, asset):
    m=np.asarray(actor_matrix,float)@np.asarray(asset['local_box_matrix']);e=np.asarray(asset['extent'])
    corners=np.array([[x,y,z] for x in [-e[0],e[0]] for y in [-e[1],e[1]] for z in [-e[2],e[2]]])@m[:3,:3].T+m[:3,3]
    return dict(T_world_from_box=m.tolist(),extent_m=e.tolist(),dimensions_lwh_m=(e*2).tolist(),corners_world_m=corners.tolist(),center_world_m=m[:3,3].tolist())

def footprint(box):
    return np.asarray(box['corners_world_m'])[[0,4,6,2],:2]

def pair_relation(a,b):
    pa=np.asarray(a['corners_world_m']);pb=np.asarray(b['corners_world_m']);xy=polygon_distance(footprint(a),footprint(b));z=max(0,float(max(pa[:,2].min(),pb[:,2].min())-min(pa[:,2].max(),pb[:,2].max())))
    return dict(xy_polygon_distance_m=xy,z_interval_gap_m=z,conservative_3d_overlap=bool(xy==0 and z==0),pair_clearance_pass=bool(xy>.1),contract='Full OBB XY envelope + Z interval, conservative separation; not exact mesh collision')

def visibility(ego_matrix, target_box, rig):
    """Frustum support only; no occlusion, render identity or pixel-size certification."""
    from geometry.reference import intrinsics_from_hfov
    center=np.asarray(target_box['center_world_m']);m=np.asarray(ego_matrix);local=np.linalg.solve(m,np.r_[center,1])[:3]
    distance=float(np.linalg.norm(local[:2]));bearing=float(np.degrees(np.arctan2(local[1],local[0])))
    k=intrinsics_from_hfov(rig['image_width'],rig['image_height'],rig['horizontal_fov_deg']);views=[]
    # Same CARLA rotation convention as existing camera rig: local forward pitch-up.
    for name,c in rig['cameras'].items():
        yaw,pitch=np.radians([c['yaw_deg'],c['pitch_deg']]);cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch)
        r=np.array([[cp*cy,-sy,-sp*cy],[cp*sy,cy,-sp*sy],[sp,0,cp]])
        t=np.eye(4);t[:3,:3]=r;t[:3,3]=c['xyz_m'];world=m@t;corners=np.asarray(target_box['corners_world_m']);points=np.vstack([center,corners]);camera=(points-world[:3,3])@world[:3,:3];opt=np.c_[camera[:,1],-camera[:,2],camera[:,0]]
        pos=opt[:,2]>.1;uv=opt@k.T;uv=uv[:,:2]/np.maximum(opt[:,2,None],1e-6)
        valid=pos&(uv[:,0]>=0)&(uv[:,0]<rig['image_width'])&(uv[:,1]>=0)&(uv[:,1]<rig['image_height'])
        views.append(dict(camera=name,center_in_frustum=bool(valid[0]),corners_in_frustum=int(valid[1:].sum()),center_pixel=uv[0].tolist() if pos[0] else None,optical_depth_m=float(opt[0,2]),occlusion='Unknown',identity='NotRun'))
    return dict(planar_distance_m=distance,bearing_deg=bearing,in_20_40m=bool(20<=distance<=40),views=views,frustum_supported=any(v['center_in_frustum'] for v in views),reference='NominalHypotheticalEgoAndRig_NoActualSensor',visibility_certified=False)
