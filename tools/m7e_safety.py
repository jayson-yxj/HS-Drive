"""Synthetic fault injection against frozen contract, no vehicle control."""
import copy
from tools.m7e_common import *
from datasets.m7d_temporal import from_packet,result_clock
from diagnostics.m7e_trajectory import state_anchored_slow


def main():
    p=read('reports/m7d_inputs/dev_constant_r3/00000.json');old=from_packet(p);out=[]
    for name in ['missing','nonmonotonic','actor_stamp','identity','future_history','future_path','NaN']:
        q=copy.deepcopy(p);prev=None
        if name=='missing':q['missing']=True
        elif name=='nonmonotonic':prev=old
        elif name=='actor_stamp':q['actors'][0]['frame']+=1
        elif name=='identity':
            prev=old;q['frame']+=4;q['sim_time']+=.2
            for a in q['actors']:a.update(frame=q['frame'],timestamp=q['sim_time'])
            q['actors'][0]['blueprint']='changed'
        elif name=='future_history':q['history_stamps']=[[q['frame']+1,q['sim_time']+.05]]
        elif name=='future_path':q['future_path']='forbidden'
        else:q['ego']['yaw']=float('nan')
        try:from_packet(q).validate(prev);rejected=False;reason=None
        except (ValueError,TypeError) as e:rejected=True;reason=str(e)
        out.append(dict(case=name,rejected=rejected,reason=reason))
    expired=result_clock(old,100,100.201,'F_state_anchored_slow')
    out.append(dict(case='expired_complete_result',rejected=expired.expired_for_online,original_timestamp_preserved=expired.input_timestamp_s==p['sim_time']))
    # Guarded process attempt uses the real M7E read allowlist after imports.
    from tools import m7e_replay as replay
    original=replay.plan
    def probe(*a,**k):
        try:open('reports/m7d_evaluator/dev_constant_r3/00000.json').read()
        except PermissionError:pass
        else:raise AssertionError('future_read_allowed')
        return original(*a,**k)
    replay.plan=probe
    # Use first packet manifest in memory, not future fixtures.
    rr=replay.read
    def one(path):
        d=rr(path)
        if str(path).endswith('manifest.json'):d={**d,'records':d['records'][:1]}
        return d
    replay.read=one
    def output(path,d):write('reports/m7e_safety_regression.json',dict(kind='SyntheticFaultsAndRealGuardReadProbe_NoActuation',provenance=d['provenance'],cases=out,all_faults_rejected=all(x['rejected'] for x in out),guard=d['access_audit'],all_original_contract_tests='See test logs',vehicle_worker_faults='NotRun',actual_execution='NotValidated',probe_result=d['records']))
    replay.write=output;replay.run('dev_constant_r3','B','_probe')
if __name__=='__main__':main()
