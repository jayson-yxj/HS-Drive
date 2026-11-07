"""No-server tests of the single initialization diagnostic entry."""
import copy,json,os
from pathlib import Path
import pytest
from tools.m8j13_init_diagnostic import InitializationDiagnostic,validate,anchor,process_identity,crash_inventory
from tools.m8j13_common import atomic
from simulator.m8j13_logic_runtime import STOP

CFG=json.loads(Path('configs/m8j13_init_diagnostic.json').read_text())

@pytest.mark.parametrize('key,value',[('max_active_ticks',181),('max_starts',2),('max_model_starts',1),('max_forward_attempts',1),('max_pid_calls',1),('stable_ticks',20),('initialization_tick_wall_s',3.),('model_actuation_allowed',True)])
def test_budget_variants_rejected(key,value):
    c=copy.deepcopy(CFG);c[key]=value
    with pytest.raises(ValueError):validate(c)


def construct(tmp_path):
    atomic(tmp_path/'request.json',dict(runtime_deadline_wall_s=1e20))
    for n in ['io_requests','io_responses']:(tmp_path/n).mkdir()
    return InitializationDiagnostic(tmp_path,CFG)


def test_actual_entry_starts_only_io_watchdog_then_fixed_hold(tmp_path):
    r=construct(tmp_path);roles=[];commands=[];ticks=[];phases=[]
    r.start=lambda role,*a:roles.append(role);r.ready=lambda *a:{}
    r.phase=lambda name,deadline:(phases.append(name),r.state.update(phase=name,phase_deadline=deadline))
    r.call=lambda kind,*a,**kw:dict(geometry={'status':'Passed'},ego_id=9,identity=['run',1,'g'])
    r.issue=lambda source,control:commands.append((source,control))
    def tick(phase):
        ticks.append(phase);r.counts[phase]+=1
        return dict(speed_norm=.2 if len(ticks)<4 else 0.)
    r.tick=tick;r.latch=lambda reason:r.result.update(terminal_reason=reason)
    r.finish_owner=lambda:r.result.update(finalized=True);r.run()
    assert roles==['io','watchdog'] and commands==[('InitialProtectionBrake',STOP)]
    assert ticks==['initialization']*24 and r.result['initialization']['stable_ticks']==21
    assert r.result['terminal_reason']=='NormalInitializationCompletion' and r.result['finalized']
    assert r.task is None and r.channel is None


def test_failure_never_retries_and_always_finishes(tmp_path):
    r=construct(tmp_path);calls=[]
    r.start=lambda *a:None;r.ready=lambda *a:{};r.phase=lambda *a:None
    def fail(*a,**kw):calls.append(a[0]);raise TimeoutError('setup failed')
    r.call=fail;r.latch=lambda reason:r.result.update(terminal_reason=reason)
    r.finish_owner=lambda:r.result.update(finalized=True);r.run()
    assert calls==['setup'] and r.result['finalized'] and not r.result.get('normal_completed')


def test_180_total_cap_is_enforced_in_real_owner_call(tmp_path):
    r=construct(tmp_path);r.pump=lambda:None;r.counts['initialization']=59;r.settings_charges['settings_startup']=121
    with pytest.raises(RuntimeError,match='TickBudgetExhausted'):r.call('tick',dict(phase='initialization'),timeout=2.)
    assert not list((tmp_path/'io_requests').glob('*.json'))


def test_capture_identity_and_clock_do_not_infer_engine_signal():
    rec=process_identity(os.getpid());assert rec['pid']==os.getpid() and rec['start_ticks']>0
    assert set(rec['environment'])=={'HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','DISPLAY','VK_ICD_FILENAMES'}
    a=anchor();assert a['monotonic_before_s']<=a['monotonic_after_s'] and a['utc'].endswith('+00:00')


def test_audit_missing_final_is_partial_with_no_synthetic_hold(tmp_path):
    from tools.m8j13_init_diagnostic import audit_run
    (tmp_path/'io_requests').mkdir()
    r=audit_run(tmp_path)
    assert r['status']=='Partial' and not r['terminal_hold']['terminal_hold_verified']
    assert r['budget']['charged_ticks'] is None


def test_audit_full_synthetic_initialization_hold_and_independent_budget(tmp_path):
    from tools.m8j13_init_diagnostic import audit_run
    from tests.m8j13.test_logic_shadow_report import synthetic_run,dump,stream
    root=synthetic_run(tmp_path,n=0,normal=True)
    # Distinct terminal name, no model; reuse synthetic snapshots only.
    fault=json.loads((root/'watchdog_fault.json').read_text());fault['reason']='NormalInitializationCompletion';dump(root/'first_fault.json',fault)
    s=json.loads((root/'supervisor.json').read_text());s.update(whole_lifecycle_within240s=True,new_crash_candidates=0);dump(root/'supervisor.json',s)
    rows=[json.loads(x) for x in (root.parent/'budget.jsonl').read_text().splitlines()];rows[-1]['charged_ticks']=102;stream(root.parent/'budget.jsonl',rows)
    r=audit_run(root)
    assert r['status']=='Passed_SingleNoModelInitialization' and r['initialization']['last21_stable']
    assert r['terminal_hold']['terminal_hold_verified'] and not r['S0_review_ready']
