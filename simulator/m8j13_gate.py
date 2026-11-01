"""Mandatory evidence check before a model can reach the live actuator."""
from pathlib import Path
import hashlib,json


def require_a_gate(path,protocol_sha256,current_source_hashes):
    path=Path(path)
    if not path.is_file():raise RuntimeError('J13ANotPassed')
    gate=json.loads(path.read_text())
    if gate.get('status')!='Passed' or gate.get('protocol_sha256')!=protocol_sha256:raise RuntimeError('J13ANotPassed')
    required=('sensor_stability_500','fault_injections','input_and_frozen_model','trajectory_controller_contracts','cleanup')
    if not all(gate.get('checks',{}).get(k) is True for k in required):raise RuntimeError('J13ANotPassed')
    for file,expected in gate.get('evidence_sha256',{}).items():
        p=Path(file)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError('J13AEvidenceChanged')
    if not gate.get('evidence_sha256'):raise RuntimeError('J13AEvidenceMissing')
    for file,expected in gate.get('source_sha256',{}).items():
        if current_source_hashes.get(file)!=expected:raise RuntimeError('J13ASourceChanged')
    if not gate.get('source_sha256'):raise RuntimeError('J13ASourceMissing')
    return gate
