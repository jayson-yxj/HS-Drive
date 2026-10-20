"""M5-A reserved-scene capture and independent collision geometry queries."""

import argparse, json, queue, time
from pathlib import Path
import cv2, numpy as np, yaml
from simulator.session import CarlaSession

from simulator.recorder import atomic_json, file_hash
from tools.capture_m45_semantic_pairs import semantic_groups
from geometry.reference import (
    intrinsics_from_hfov,
    camera_mount_to_hs_from_optical,
    transform_points,
    decode_carla_depth_bgra,
    CARLA_EGO_TO_HS_EGO,
)
from geometry.coordinates import carla_pose_to_hs

SCENES = [
    ("road_a", "road", 0, []),
    ("road_b", "road", 45, []),
    ("wall_a", "wall", 250, []),
    ("wall_b", "wall", 24, []),
    ("near_a", "near_obstacle", 0, [(8, 0)]),
    ("near_b", "near_obstacle", 45, [(11, 0)]),
    ("junction_a", "junction", 1, [(14, 0)]),
    ("junction_b", "junction", 3, []),
    ("slope_a", "slope", -1, []),
    ("slope_b", "slope", 152, []),
    ("occlusion_a", "occlusion", 0, [(8, 0), (16, 0)]),
    ("occlusion_b", "occlusion", 45, [(8, 0), (16, 0)]),
    ("dynamic_a", "dynamic", 0, [(10, -4)]),
]


def main():
    import carla

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", default="/root/gpufree-data/hs-drive-data/raw/m5a_reserved_capture"
    )
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--map", default="Town03")
    p.add_argument("--vehicle", default="vehicle.tesla.model3")
    p.add_argument("--spawn", type=int, default=100)
    p.add_argument("--role", choices=["development", "heldout"], default="development")
    p.add_argument("--groups",nargs="*")
    a = p.parse_args()
    protocol=json.loads(Path("configs/m5a_protocol.json").read_text())
    root = Path(a.output)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    cfg = yaml.safe_load(Path("configs/carla.yaml").read_text())
    cfg.update(port=2100, map=a.map)
    rig = yaml.safe_load(Path("configs/cameras.yaml").read_text())
    all_results = []
    for group in protocol['groups']:
        sid=group['group'];category='paired';spawn_id=-2
        if a.groups and sid not in a.groups:continue
        cfg['map']=group['map']
        directory = root / sid
        directory.mkdir()
        (directory / "frames").mkdir()
        q = queue.Queue()
        with CarlaSession(cfg, load_map=True) as session:
            world = session.world
            cmap = world.get_map()
            world.set_weather(carla.WeatherParameters.ClearNoon)
            x,y,z,pitch,yaw,roll=group['pose']
            spawn=carla.Transform(carla.Location(x=x,y=y,z=z),carla.Rotation(pitch=pitch,yaw=yaw,roll=roll))
            bp = world.get_blueprint_library().find(a.vehicle)
            ego = session.spawn(bp, spawn)
            ego.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            for _ in range(30):
                world.tick()
            pose = ego.get_transform()
            ego.set_simulate_physics(False)
            ego.set_transform(pose)
            ego_matrix = np.array(pose.get_matrix())
            targets=[]
            for index,kind in enumerate(['vehicle.tesla.model3','vehicle.audi.a2','walker.pedestrian.0001']):
                tf=carla.Transform(pose.transform(carla.Location(x=1000+index*10,y=0,z=10)),pose.rotation)
                actor=session.spawn(world.get_blueprint_library().find(kind),tf)
                actor.set_simulate_physics(False);targets.append(actor)
            def configure(index):
                placements=protocol['configurations'][index]
                for j,actor in enumerate(targets):
                    loc=next(([x,y] for x,y,kind in placements if kind==actor.type_id),[1000+j*10,0])
                    # Frozen test actors, geometry changes without trajectory/time confound.
                    zoff=.05 if actor.type_id.startswith('vehicle') else .85
                    actor.set_transform(carla.Transform(pose.transform(carla.Location(x=loc[0],y=loc[1],z=zoff)),pose.rotation))
            cal = {
                "schema_version": "hsd-m5a-calibration-v1",
                "depth_convention": "optical_z",
                "cameras": {},
            }
            for name, mount in rig["cameras"].items():
                local = carla.Transform(
                    carla.Location(*mount["xyz_m"]),
                    carla.Rotation(yaw=mount["yaw_deg"], pitch=mount["pitch_deg"]),
                )
                # Explicit Euler composition via rotation matrix; CARLA pitch/yaw/roll convention.
                matrix = ego_matrix @ np.array(local.get_matrix())
                rot = matrix[:3, :3]
                pitch = np.degrees(np.arcsin(np.clip(rot[2, 0], -1, 1)))
                yaw = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
                roll = np.degrees(np.arctan2(-rot[2, 1], rot[2, 2]))
                tf = carla.Transform(
                    carla.Location(*matrix[:3, 3]),
                    carla.Rotation(
                        pitch=float(pitch), yaw=float(yaw), roll=float(roll)
                    ),
                )
                assert np.allclose(tf.get_matrix(), matrix, atol=2e-4)
                cal["cameras"][name] = {
                    "K": intrinsics_from_hfov(640, 384, 100).tolist(),
                    "T_hs_ego_from_opt": camera_mount_to_hs_from_optical(
                        local.get_matrix()
                    ).tolist(),
                    "T_carla_world_from_camera": matrix.tolist(),
                    "width": 640,
                    "height": 384,
                }
                for kind in ("rgb", "depth", "instance_segmentation"):
                    blue = world.get_blueprint_library().find("sensor.camera." + kind)
                    attrs = dict(
                        image_size_x=640,
                        image_size_y=384,
                        fov=100,
                        sensor_tick=0,
                        lens_k=0,
                        lens_kcube=0,
                    )
                    if kind == "rgb":
                        attrs.update(motion_blur_intensity=0)
                    for key, value in attrs.items():
                        blue.set_attribute(key, str(value))
                    sensor = session.spawn(
                        blue, tf
                    )  # detached: body remains visible; cameras don't move on ego removal
                    sensor.listen(lambda image, n=name, k=kind: q.put((n, k, image)))
            atomic_json(directory / "calibration.json", cal)

            def capture(label, save=True):
                frame = world.tick()
                snap = world.get_snapshot()
                found = {}
                deadline = time.monotonic() + 40
                while len(found) < 18:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Missing camera bundle")
                    n, k, img = q.get(timeout=10)
                    if img.frame == frame:
                        found[n, k] = img
                    elif img.frame > frame:
                        raise ValueError("Frame mismatch")
                if not save:
                    return
                actors = []
                for actor in [ego, *targets]:
                    if not actor.is_alive:
                        continue
                    trans = actor.get_transform()
                    corners = np.array(
                        [
                            [v.x, v.y, v.z]
                            for v in actor.bounding_box.get_world_vertices(trans)
                        ]
                    )
                    actors.append(
                        dict(
                            actor_id=actor.id,
                            is_ego=actor.id == ego.id,
                            type_id=actor.type_id,
                            world_matrix=trans.get_matrix(),
                            corners_hs_ego_m=transform_points(
                                corners, CARLA_EGO_TO_HS_EGO @ np.linalg.inv(ego_matrix)
                            ).tolist(),
                        )
                    )
                f = dict(
                    frame_id=frame,
                    timestamp_s=snap.timestamp.elapsed_seconds,
                    phase=label,
                    T_world_from_ego=carla_pose_to_hs(ego_matrix).tolist(),
                    actors=actors,
                    cameras={},
                )
                for name in rig["cameras"]:
                    entry = {}
                    for kind, short in [
                        ("rgb", "rgb"),
                        ("depth", "depth"),
                        ("instance_segmentation", "instance"),
                    ]:
                        img = found[name, kind]
                        assert abs(img.timestamp - f["timestamp_s"]) < 1e-5
                        arr = np.frombuffer(img.raw_data, np.uint8).reshape(384, 640, 4)
                        path = directory / f"{frame:08d}_{name}_{short}"
                        path = path.with_suffix(".png" if short == "rgb" else ".npz")
                        if short == "rgb":
                            cv2.imwrite(str(path), arr[:, :, :3])
                        elif short == "depth":
                            np.savez_compressed(
                                path,
                                depth_m=decode_carla_depth_bgra(img.raw_data, 384, 640),
                            )
                        else:
                            np.savez_compressed(
                                path,
                                semantic=arr[:, :, 2],
                                instance=arr[:, :, 1].astype(np.uint16) * 256
                                + arr[:, :, 0].astype(np.uint16),
                            )
                        entry[short] = path.name
                        entry[short + "_sha256"] = file_hash(path)
                    f["cameras"][name] = entry
                atomic_json(directory / "frames" / f"{frame:08d}.json", f)
                return f

            # Let the render thread initialize before strict frame joins. First
            # render bundles can be incomplete while sensors are registering.
            for _ in range(20):
                world.tick()
                time.sleep(0.08)
                while not q.empty():
                    q.get_nowait()
            for _ in range(3):
                capture("warmup", False)
            frames=[]
            for index in range(len(protocol['configurations'])):
                configure(index)
                for _ in range(3):capture('warmup',False)
                record=capture('ego_present');record['configuration']=index
                atomic_json(directory/'frames'/f"{record['frame_id']:08d}.json",record)
                frames.append(record)
            ego_id=ego.id;ego.destroy();session.actors.remove(ego)
            references={}
            for index,record in enumerate(frames):
                configure(index)
                for _ in range(5):capture('warmup',False)
                reference=capture('ego_removed');references[str(record['frame_id'])]=reference['frame_id']
            manifest = dict(
                schema_version="hsd-m5a-body-visibility-record-v1",
                scene=sid,
                category=category,
                map=cmap.name,
                spawn_index=spawn_id,
                ego_actor_id=ego_id,
                frames=[f["frame_id"] for f in frames],
                counterfactual_frame=reference["frame_id"],
                references=references,split=group["split"],group=sid,configurations=protocol["configurations"],
                semantic_groups=semantic_groups(carla),
                calibration_sha256=file_hash(directory / "calibration.json"),
                ego_pose=carla_pose_to_hs(ego_matrix).tolist(),
                pitch_deg=float(pose.rotation.pitch),
                role=group["split"],
                vehicle=a.vehicle,
                camera_attachment="detached_fixed_world",
                note="Same camera transforms with and without ego. Instance IDs are renderer IDs, not assumed actor IDs. Each configuration has its own matched counterfactual.",
            )
            atomic_json(directory / "manifest.json", manifest)
            all_results.append(manifest)
            print(sid, "complete", flush=True)
    atomic_json(
        root / "manifest.json",
        dict(schema_version="hsd-m5a-campaign-v1", scenes=all_results, training=False),
    )


if __name__ == "__main__":
    main()
