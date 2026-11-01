"""One isolated official 0.9.16 candidate, at most two starts/900 active ticks.

No model or normal controller. Pixel diagnostic moves a visible red marker
vehicle, never the ego, and compares pixels with same-frame marker snapshots.
"""
import argparse
from collections import defaultdict, Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np

from tools.m8j13_common import config, digest, write, atomic, append, quantiles
from tools.m8j13_sync_diagnostic import native_camera_spec, settings_record, snapshot_record, CAMERAS
from simulator.m8j13_candidate_audit import CandidateInbox, marker_pixels, deadline_decision
from simulator.m8j13_sensor_audit import check_group

CFG=Path('configs/m8j13_candidate.json')


def worker(root):
    import carla
    c=config();cfg=json.loads(CFG.read_text());request=json.loads((root/'request.json').read_text())
    spec=cfg[request['run']];client=carla.Client('127.0.0.1',c['port']);client.set_timeout(45.)
    world=client.get_world();original=world.get_settings();actors=[];sensors=[];inbox=None
    snapshots={};records=[];groups=defaultdict(list);images={};decisions={};marker=None;marker_locations={}
    ticks=0;window=[];result=dict(status='Partial',MetadataConsistency='NotRun',CaptureAssociation='NotRun',
        DeadlineAndProtection='NotRun',model_forwards=0,model_controls=0,training=0)
    phase='initialize';actor_ids={};sensor_map={};identity=None;duplicate_records=[];pixel_checks=[]
    def event(kind,**kwargs):append(root/'lifecycle.jsonl',dict(event=kind,phase=phase,monotonic_s=time.monotonic(),**kwargs))
    def consume():
        for payload in inbox.drain():
            r=payload.stamp.record();r.update(record_id=len(records),width=payload.width,height=payload.height,
                format='BGRA8',bytes=len(payload.bgra),sha256=hashlib.sha256(payload.bgra).hexdigest(),
                copy_end_monotonic_s=payload.copied_monotonic_s,consume_monotonic_s=time.monotonic(),consume_phase=phase)
            f=r['frame']; prior=[x for x in groups[f] if x['camera']==r['camera']]
            if prior:
                duplicate_records.append(dict(first_record=prior[0]['record_id'],later_record=r['record_id'],frame=f,
                    same_header=all(prior[0][k]==r[k] for k in ['run','episode','sensor_generation','camera','actor_id','frame','timestamp_s','transform','width','height','format']),
                    same_payload=prior[0]['sha256']==r['sha256'],previously_accepted=f in decisions and decisions[f]['accepted']))
            records.append(r);groups[f].append(r)
            # Limited R1 marker images and every conflicting payload retained.
            if (marker is not None and r['camera']=='CAM_FRONT') or prior:
                images[r['record_id']]=payload.bgra
            snap=snapshots.get(f)
            if marker is not None and r['camera']=='CAM_FRONT' and snap is not None:
                p=snap['marker']['center_world'];opp=snap['marker']['opposite_center_world']
                scores=marker_pixels(payload.bgra,payload.width,payload.height,r['transform'],p,opp,
                    cameras['CAM_FRONT']['fov'],cfg['marker'])
                pixel_checks.append(dict(frame=f,record_id=r['record_id'],phase=snap['phase'],**scores))
        # Final acceptance decisions retain their first result; duplicates are
        # separate faults and never overwrite the original accepted identity.
        for f in sorted(groups):
            if f in decisions or f not in snapshots:continue
            if set(r['camera'] for r in groups[f])!=set(CAMERAS):continue
            snap=snapshots[f];check=check_group(groups[f],snap,identity,actor_ids,cfg['timestamp_tolerance_s'])
            pose_error=max(float(np.max(np.abs(np.asarray(r['transform'])-np.asarray(snap['camera_transforms'][r['camera']])))) for r in groups[f])
            now=time.monotonic();decision=deadline_decision(snap['elapsed_seconds'],snap['tick_start_monotonic_s'],
                snapshots[max(snapshots)]['elapsed_seconds'],now,cfg['deadline_sim_s'],cfg['deadline_wall_s'])
            if not check['valid'] or pose_error>cfg['pose_matrix_tolerance']:
                decision.update(accepted=False,reason='InvalidSensorMetadata',stop_command=dict(steer=0.,throttle=0.,brake=1.))
            decision.update(frame=f,metadata_valid=check['valid'],metadata_errors=check['errors'],pose_max_abs_error=pose_error,
                start_monotonic_s=snap['tick_start_monotonic_s'],decision_monotonic_s=now,
                last_callback_monotonic_s=max(r['receive_monotonic_s'] for r in groups[f]),
                sources=[r['record_id'] for r in groups[f]])
            decisions[f]=decision;append(root/'decisions.jsonl',decision)
    try:
        assert client.get_server_version()==client.get_client_version()==cfg['version']
        assert not original.synchronous_mode
        event('initial_world',episode=world.id,settings=settings_record(world))
        if world.get_map().name.split('/')[-1]!=c['map']:
            event('load_map_async',map=c['map']);world=client.load_world(c['map']);original=world.get_settings()
            event('loaded',episode=world.id,settings=settings_record(world))
        assert not original.synchronous_mode and original.fixed_delta_seconds is None
        s=world.get_settings();s.synchronous_mode=True;s.fixed_delta_seconds=.05;s.no_rendering_mode=False
        s.substepping=True;s.max_substep_delta_time=.01;s.max_substeps=10
        before=world.get_snapshot().frame;world.apply_settings(s);after=world.get_snapshot().frame
        result['apply_settings_frame_delta']=after-before
        event('settings_applied',settings=settings_record(world),observed_delta=after-before,charged_ticks=30)
        actual=settings_record(world)
        assert actual==dict(synchronous_mode=True,fixed_delta_seconds=.05,no_rendering_mode=False,substepping=True,max_substep_delta_time=.01,max_substeps=10)
        client.set_timeout(5.);world.set_weather(carla.WeatherParameters.ClearNoon)
        scene=c['scenes'][0];route=json.loads(Path(scene['route_file']).read_text());i=scene['start_index'];xyz=route['route_world_carla'][i]
        yaw=-np.degrees(route['route_world_hs'][i][2]);ego_tf=carla.Transform(carla.Location(x=xyz[0],y=xyz[1],z=xyz[2]+.5),carla.Rotation(yaw=float(yaw)))
        bp=world.get_blueprint_library().find(c['vehicle']);ego=world.spawn_actor(bp,ego_tf);actors.append(ego);ego.set_simulate_physics(False)
        identity=(root.name,world.id,root.name+':rgb:1');inbox=CandidateInbox(identity)
        cameras=native_camera_spec(Path(c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
        for name in CAMERAS:
            v=cameras[name];bp=world.get_blueprint_library().find('sensor.camera.rgb')
            for k,x in [('image_size_x',704),('image_size_y',396),('fov',v['fov']),('sensor_tick',cfg['sensor_tick'])]:bp.set_attribute(k,str(x))
            tf=carla.Transform(carla.Location(x=v['x'],y=v['y'],z=v['z']),carla.Rotation(pitch=v['pitch'],yaw=v['yaw'],roll=v['roll']))
            a=world.spawn_actor(bp,tf,attach_to=ego,attachment_type=carla.AttachmentType.Rigid);actors.append(a);sensors.append(a)
            actor_ids[name]=a.id;sensor_map[name]=a;event('camera_created',camera=name,id=a.id,attributes=dict(a.attributes))
            a.listen(inbox.callback(name,a.id))
        if spec['pixel_marker']:
            m=cfg['marker'];bp=world.get_blueprint_library().find(m['blueprint']);bp.set_attribute('color',m['color'])
            for sign in [-1,1]:
                marker_locations[sign]=ego_tf.transform(carla.Location(x=m['forward_m'],y=sign*m['lateral_m'],z=m['height_m']))
            marker=world.spawn_actor(bp,carla.Transform(marker_locations[-1],carla.Rotation(yaw=float(yaw))))
            actors.append(marker);marker.set_simulate_physics(False);event('rendered_marker_created',id=marker.id,attributes=dict(marker.attributes))
        write(root/'runtime.json',dict(version=client.get_server_version(),client_path=carla.libcarla.__file__,
            client_sha256=digest(carla.libcarla.__file__),episode=world.id,actors=actor_ids,settings=actual,
            sensor_tick=cfg['sensor_tick'],camera_order=CAMERAS,marker=spec['pixel_marker'],traffic_manager='NotCreated'))
        last=world.get_snapshot().frame;next_cycle=time.monotonic()
        total=spec['warmup_ticks']+spec['measurement_ticks']+spec['drain_ticks']
        for index in range(total):
            phase='warmup' if index<spec['warmup_ticks'] else 'measurement' if index<total-spec['drain_ticks'] else 'drain'
            time.sleep(max(0.,next_cycle-time.monotonic()));next_cycle=time.monotonic()+.05
            if marker is not None:
                pattern=(-1,1,1,-1,1,-1,-1,1);sign=pattern[(index//cfg['marker']['switch_every_ticks'])%len(pattern)]
                marker.set_transform(carla.Transform(marker_locations[sign],carla.Rotation(yaw=float(yaw))))
            start=time.monotonic();append(root/'tick_requests.jsonl',dict(index=index,phase=phase,monotonic_s=start));ticks+=1
            frame=world.tick(5.);returned=time.monotonic();snap=world.get_snapshot()
            if snap.frame!=frame or frame!=last+1:raise RuntimeError('CompetingTickOwnerOrFrameGap')
            last=frame;row=snapshot_record(snap,ego);row.update(phase=phase,index=index,tick_start_monotonic_s=start,tick_return_monotonic_s=returned,
                camera_transforms={n:snap.find(a.id).get_transform().get_matrix() for n,a in sensor_map.items()})
            if marker is not None:
                mt=snap.find(marker.id).get_transform();p=mt.transform(carla.Location(z=.7))
                actual_sign=min(marker_locations,key=lambda k:mt.location.distance(marker_locations[k]))
                opp=marker_locations[-actual_sign]
                row['marker']=dict(actor_id=marker.id,requested_sign=sign,transform=mt.get_matrix(),
                    snapshot_sign=actual_sign,center_world=[p.x,p.y,p.z],opposite_center_world=[opp.x,opp.y,opp.z+.7])
            snapshots[frame]=row;append(root/'snapshots.jsonl',row)
            if phase=='measurement':window.append(frame)
            consume()
            if inbox.overflow:raise RuntimeError('CallbackQueueOverflow')
        phase='receive_only_drain';event('receive_only_drain_begin')
        deadline=time.monotonic()+cfg['receive_only_drain_wall_s']
        while time.monotonic()<deadline:consume();time.sleep(.005)
        event('receive_only_drain_end');result['status']='Measured'
    except BaseException as exc:result.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        phase='stop_listeners'
        if inbox is not None:
            consume();inbox.close();event('generation_invalidated')
        for a in sensors:
            try:a.stop();event('listener_stopped',id=a.id)
            except Exception as exc:event('cleanup_error',error=repr(exc))
        for a in reversed(actors):
            try:event('actor_destroyed',id=a.id,success=a.destroy())
            except Exception as exc:event('cleanup_error',error=repr(exc))
        phase='restore_async'
        try:
            world.apply_settings(original);fresh=world.wait_for_tick(5.)
            result['cleanup']=dict(settings=settings_record(world),remaining_owned=[a.id for a in actors if fresh.find(a.id)])
            event('restored',**result['cleanup'])
        except Exception as exc:result['cleanup_error']=repr(exc)
        # Byte copies were made at callback entry, so serialization after
        # cleanup does not depend on a retained CARLA Image object.
        for rid,data in images.items():
            p=root/'images'/('%06d.bgra'%rid);p.parent.mkdir(exist_ok=True);p.write_bytes(data)
        write(root/'raw_metadata.json',records);write(root/'duplicate_records.json',duplicate_records);write(root/'pixel_checks.json',pixel_checks)
        measured=[decisions[f] for f in window if f in decisions];missing=[f for f in window if f not in decisions]
        failed_meta=[d for d in measured if not d['metadata_valid'] or d['pose_max_abs_error']>cfg['pose_matrix_tolerance']]
        pixel_measured=[p for p in pixel_checks if p['phase']=='measurement'];pixel_bad=[p for p in pixel_measured if not p['matched']]
        metadata_ok=len(window)==spec['measurement_ticks'] and not missing and not failed_meta and not duplicate_records
        capture_ok=metadata_ok and (not marker or (len(pixel_measured)==len(window) and not pixel_bad))
        reasons=Counter(d['reason'] for d in measured if d['reason'])
        result.update(MetadataConsistency='Supported' if metadata_ok else 'ConflictOrIncomplete',
            CaptureAssociation=('SupportedFiniteMarkerTest' if capture_ok and marker else 'ConsistentWithR1LimitedEvidence' if capture_ok else 'ConflictOrIncomplete'),
            DeadlineAndProtection=dict(expired_groups=sum(d['reason'] in ['PlanWallExpired','PlanSimulationExpired'] for d in measured),
                accepted_groups=sum(d['accepted'] for d in measured),rejected_groups=sum(not d['accepted'] for d in measured),
                reasons=dict(reasons),stop_outputs=sum(d['stop_command'] is not None for d in measured),actuated=0,
                wall_age_s=quantiles([d['wall_age_s'] for d in measured])),
            measurement_ticks=len(window),expected_groups=spec['measurement_ticks'],complete_groups=len(measured),missing_groups=len(missing),
            metadata_failures=len(failed_meta),duplicate_keys=len(duplicate_records),pixel_checks=len(pixel_measured),pixel_failures=len(pixel_bad),
            first_pixel_failures=pixel_bad[:3],raw_count=len(records),explicit_ticks=ticks,charged_ticks=ticks+30,
            all_phase_counts=dict(Counter(s['phase'] for s in snapshots.values())),callback_overflow=0 if inbox is None else inbox.overflow,
            run2_eligible=bool(capture_ok and result['status']=='Measured'),model_forwards=0,model_controls=0,training=0)
        write(root/'result.json',result);print(json.dumps(result,indent=2),flush=True)


def supervise(run,hypothesis):
    import carla
    from tools.m8j12_capture_supervisor import listening,group_members
    from tools.m8j13_supervisor import terminate
    c=config();cfg=json.loads(CFG.read_text());base=Path(cfg['private_root']);base.mkdir(parents=True,exist_ok=True)
    spec=cfg[run];reserve=30+sum(spec[k] for k in ['warmup_ticks','measurement_ticks','drain_ticks'])
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger_path=base/'budget.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'runs':[]}
        if len(ledger['runs'])>=cfg['max_starts'] or sum(x['reserved_ticks'] for x in ledger['runs'])+reserve>cfg['max_active_ticks']:raise RuntimeError('CandidateBudgetExhausted')
        if run=='R2' and not json.loads((base/'R1/result.json').read_text())['run2_eligible']:raise RuntimeError('R1HasUnresolvedContradictions')
        if any(listening(p) for p in range(c['port'],c['port']+3)):raise RuntimeError('PortsOccupied')
        root=base/run;root.mkdir();request=dict(run=run,hypothesis=hypothesis,reserved_ticks=reserve,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            sources={str(p):digest(p) for p in [CFG,Path('tools/m8j13_candidate.py'),Path('simulator/m8j13_candidate_audit.py')]})
        write(root/'request.json',request);ledger['runs'].append(dict(run=run,reserved_ticks=reserve));atomic(ledger_path,ledger)
        server=None;child=None;report=dict(request=request);env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        cmd=['taskset','-c',c['process_cpus']['server'],'runuser','-u','ubuntu','--','bash',str(Path('scripts/start_carla.sh').resolve()),cfg['server_root'],str(c['port']),'Epic']
        report['server_argv']=cmd
        try:
            with (root/'server.log').open('x') as f:server=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            until=time.monotonic()+150
            while time.monotonic()<until:
                if server.poll() is not None:raise RuntimeError('ServerExitedBeforeReady')
                client=carla.Client('127.0.0.1',c['port']);client.set_timeout(2.)
                try:
                    assert client.get_server_version()==client.get_client_version()==cfg['version'];client.get_world();break
                except RuntimeError:time.sleep(1.)
            else:raise TimeoutError('StartupTimeout')
            command=['taskset','-c',c['process_cpus']['capture'],cfg['python'],'-m','tools.m8j13_candidate','--worker-root',str(root)]
            report['worker_argv']=command
            with (root/'worker.log').open('x') as f:
                child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);until=time.monotonic()+600
                while child.poll() is None:
                    if server.poll() is not None:raise RuntimeError('ServerExitedDuringRun')
                    if time.monotonic()>until:raise TimeoutError('WorkerTimeout')
                    time.sleep(.2)
            report['worker_exit_code']=child.returncode
            if child.returncode:raise RuntimeError('WorkerFailed')
            report['result']=json.loads((root/'result.json').read_text())
        except BaseException as exc:report['error']=repr(exc)
        finally:
            terminate(child);terminate(server)
            report['ports_after']={str(p):listening(p) for p in range(c['port'],c['port']+3)}
            report['groups_after']=[group_members(p.pid) for p in [server,child] if p is not None]
            ledger['runs'][-1]['charged_ticks']=report.get('result',{}).get('charged_ticks',reserve);atomic(ledger_path,ledger)
            write(root/'supervisor.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',choices=['R1','R2']);p.add_argument('--hypothesis');p.add_argument('--worker-root',type=Path)
    a=p.parse_args();os.umask(0o077)
    if a.worker_root:
        write(a.worker_root/'worker_started.json',dict(pid=os.getpid()));worker(a.worker_root)
    elif a.run and a.hypothesis:supervise(a.run,a.hypothesis)
    else:p.error('run and hypothesis required')


if __name__=='__main__':main()
