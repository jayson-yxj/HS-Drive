"""Snapshot-stamped privileged obstacle boxes, never RGB inference features."""
import math
import numpy as np


def snapshot_obstacle(snapshot,actor):
    current=snapshot.find(actor.id)
    if current is None:raise ValueError('actor absent from snapshot')
    tf=current.get_transform();bb=actor.bounding_box
    # CARLA BoundingBox applies its local offset and rotation to all corners.
    vertices=bb.get_world_vertices(tf)
    xyz=np.array([[p.x,-p.y,p.z] for p in vertices]);center=xyz.mean(0)
    yaw=-math.radians(tf.rotation.yaw+bb.rotation.yaw)
    basis=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    local=(xyz[:,:2]-center[:2])@basis
    ext=np.abs(local).max(0)  # conservative planar box for roll/pitch projection
    v=current.get_velocity()
    return dict(xy=center[:2].tolist(),yaw=yaw,half_length=float(ext[0]),half_width=float(ext[1]),
                velocity=[v.x,-v.y],actor_id=actor.id,source='Privileged',frame=snapshot.frame,
                reference='snapshot_OBB_center_planar_enclosure_not_render_mesh',
                timestamp=snapshot.timestamp.elapsed_seconds)
