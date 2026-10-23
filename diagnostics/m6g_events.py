"""Offline outer-frame event alignment. No CARLA dependency or control output."""
import math

CONTROL_FIELDS = ('throttle', 'brake', 'steer', 'hand_brake', 'reverse')
NUMERIC_FIELDS = ('throttle', 'brake', 'steer')


def equal(a,b,fields=CONTROL_FIELDS):
    return all(k in a and k in b and abs(float(a[k])-float(b[k]))<=1e-5 for k in fields)


def command_events(rows):
    """One event per requested state change, censored at next request change.

    First observed matching Applied sample bounds application from above;
    an outer frame is never labeled an exact physical input timestamp.
    """
    starts=[i for i,r in enumerate(rows) if i==0 or not equal(r['requested'],rows[i-1]['requested'])]
    events=[]
    for n,i in enumerate(starts):
        end=starts[n+1] if n+1<len(starts) else len(rows)
        target=rows[i]['requested'];previous=rows[i-1]['requested'] if i else rows[i]['body_before']['applied']
        applied=next((j for j in range(i,end) if equal(rows[j]['body']['applied'],target)),None)
        events.append(dict(request_index=i,end_index=end,applied_index=applied,
            changed_fields=[k for k in CONTROL_FIELDS if not equal(target,previous,[k])],
            delay_outer_ticks=None if applied is None else applied-i,
            status='Observed' if applied is not None else 'Unresolved_NoMatchingAppliedBeforeNextRequest',
            command=target))
    return events


def flatten(rows,rebuild=False):
    result=[];previous_acc=None
    for i,r in enumerate(rows):
        pre=r['body_before'];post=r['body'];dt=post['t']-pre['t']
        if dt<=0 or post['frame']!=pre['frame']+1:raise ValueError('NonconsecutiveBodyFrame')
        if i and (pre['frame']!=rows[i-1]['body']['frame'] or abs(pre['t']-rows[i-1]['body']['t'])>1e-6):raise ValueError('InterRowFrameGap')
        valid=not (rebuild and i==0)
        acc=(post['long_speed']-pre['long_speed'])/dt if valid else None
        jerk=(acc-previous_acc)/dt if acc is not None and previous_acc is not None else None
        previous_acc=acc
        tel=r['telemetry_after'];data=None if tel is None else tel['data']
        if tel is not None and (tel['associated_frame']!=post['frame'] or tel['frame_before']!=tel['frame_after']):raise ValueError('TelemetryFrameMismatch')
        result.append(dict(index=i,frame=post['frame'],timestamp_s=post['t'],phase=r['phase'],request=r['requested'],
          request_wall_time=r['timeline']['apply'],request_rpc_return_wall_time=r['timeline']['apply_end'],
          observation_wall_upper_bound=r['timeline']['end'],applied=post['applied'],
          body_signed_speed_mps=post['long_speed'],body_velocity_world=post['velocity'],body_position_world=post['position'],
          API_long_acceleration_mps2=post['long_acceleration_api'],acceleration_50ms_mps2=acc,jerk_50ms_mps3=jerk,
          derivative_valid=valid,physics_rebuild_boundary=rebuild and i==0,telemetry=data,
          engine_substep_timestamp='NotAvailable',telemetry_valid=tel is not None))
    return result


def first_index(samples,start,end,predicate):
    return next((i for i in range(start,end) if predicate(samples[i])),None)


def point(samples,index,origin):
    if index is None:return None
    s=samples[index]
    return dict(index=index,frame=s['frame'],timestamp_s=s['timestamp_s'],relative_to_first_applied_observation_s=s['timestamp_s']-samples[origin]['timestamp_s'])


def wheel_values(sample,key):
    t=sample['telemetry']
    if not t or len(t['wheels'])!=4:return None
    values=[w[key] for w in t['wheels']]
    return values if all(isinstance(x,(float,int)) and math.isfinite(x) for x in values) else None


def braking_metrics(samples,event,thresholds):
    a=event['applied_index'];end=event['end_index']
    if a is None:return {'status':'Unresolved'}
    base=samples[max(0,a-1)];om=wheel_values(base,'omega')
    def wheels(i,key,predicate):
        v=wheel_values(samples[i],key);return v is not None and predicate(v)
    predicates={
      'any_wheel_decrease':lambda i:om is not None and wheels(i,'omega',lambda v:any(abs(x)-abs(y)>=thresholds['wheel_decrease_rad_s'] for x,y in zip(om,v))),
      'all_wheels_locked':lambda i:wheels(i,'omega',lambda v:all(abs(x)<=thresholds['wheel_lock_abs_rad_s'] for x in v)),
      'large_slip':lambda i:wheels(i,'long_slip',lambda v:any(abs(x)>=thresholds['large_slip_abs'] for x in v)),
      'large_deceleration':lambda i:samples[i]['acceleration_50ms_mps2'] is not None and samples[i]['acceleration_50ms_mps2'] < -thresholds['deceleration_mps2'],
      'body_near_stopped':lambda i:abs(samples[i]['body_signed_speed_mps'])<=thresholds['body_motion_mps']}
    idx={k:next((i for i in range(a,end) if fn(i)),None) for k,fn in predicates.items()}
    decel=[i for i in range(a,end) if samples[i]['acceleration_50ms_mps2'] is not None]
    peak=min(decel,key=lambda i:samples[i]['acceleration_50ms_mps2']) if decel else None
    slip_pre=wheel_values(base,'long_slip');large_pre=slip_pre is not None and any(abs(x)>=thresholds['large_slip_abs'] for x in slip_pre)
    w=idx['any_wheel_decrease'];d=idx['large_deceleration']
    order='Unresolved' if w is None or d is None else 'same_outer_frame' if w==d else 'wheel_decrease_observed_first' if w<d else 'body_deceleration_observed_first'
    return dict(status='Observed_outer_frames_only',ordering=order,baseline_speed_mps=base['body_signed_speed_mps'],baseline_large_slip=large_pre,
      first={k:point(samples,v,a) for k,v in idx.items()},peak_deceleration=point(samples,peak,a),
      peak_deceleration_mps2=None if peak is None else -samples[peak]['acceleration_50ms_mps2'],
      baseline_wheels=None if base['telemetry'] is None else base['telemetry']['wheels'],
      first_slip_wheels=None if idx['large_slip'] is None else samples[idx['large_slip']]['telemetry']['wheels'],
      peak_wheels=None if peak is None or samples[peak]['telemetry'] is None else samples[peak]['telemetry']['wheels'])
