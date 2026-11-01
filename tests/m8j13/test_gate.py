import hashlib,json
import pytest
from simulator.m8j13_gate import require_a_gate


def test_missing_or_partial_gate_never_arms(tmp_path):
    p=tmp_path/'gate.json'
    with pytest.raises(RuntimeError,match='NotPassed'):require_a_gate(p,'p',{})
    p.write_text(json.dumps(dict(status='Partial',protocol_sha256='p')))
    with pytest.raises(RuntimeError,match='NotPassed'):require_a_gate(p,'p',{})


def test_gate_binds_all_checks_evidence_and_source(tmp_path):
    e=tmp_path/'evidence.json';e.write_text('{}')
    p=tmp_path/'gate.json'
    gate=dict(status='Passed',protocol_sha256='p',checks={k:True for k in ('sensor_stability_500','fault_injections','input_and_frozen_model','trajectory_controller_contracts','cleanup')},
        evidence_sha256={str(e):hashlib.sha256(e.read_bytes()).hexdigest()},source_sha256={'s.py':'h'})
    p.write_text(json.dumps(gate));assert require_a_gate(p,'p',{'s.py':'h'})==gate
    with pytest.raises(RuntimeError,match='SourceChanged'):require_a_gate(p,'p',{'s.py':'different'})
    e.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='EvidenceChanged'):require_a_gate(p,'p',{'s.py':'h'})
