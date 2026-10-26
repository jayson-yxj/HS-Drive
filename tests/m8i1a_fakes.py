"""Synthetic CARLA-shaped objects only; never imports CARLA or starts a server."""
import argparse
import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from tools.m8i1a_common import persist
from tools.m8i1a_identity import run_session, terminated

XODR = '<OpenDRIVE synthetic="true" />'


class Map:
    def __init__(self, case, calls):
        self.case, self.calls = case, calls
        self.name = '/Game/Carla/Maps/UnexpectedSyntheticMap' if case == 'map_mismatch' else '/Game/Carla/Maps/Town02'

    def to_opendrive(self):
        self.calls.append('map.to_opendrive')
        return 'synthetic different XODR' if self.case == 'xodr_mismatch' else XODR


class World:
    id = 123456

    def __init__(self, case, calls):
        self.case, self.calls = case, calls

    def get_map(self):
        self.calls.append('world.get_map')
        if self.case == 'map_unavailable':
            raise RuntimeError('Synthetic map RPC unavailable')
        return Map(self.case, self.calls)

    def get_snapshot(self):
        self.calls.append('world.get_snapshot')
        return SimpleNamespace(frame=321, timestamp=SimpleNamespace(
            elapsed_seconds=16.05, delta_seconds=.05, platform_timestamp=1234.5))

    def get_settings(self):
        self.calls.append('world.get_settings')
        return SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None,
                               no_rendering_mode=False, substepping=True)

    def get_weather(self):
        self.calls.append('world.get_weather')
        return 'SyntheticWeather'


class Client:
    def __init__(self, case, calls):
        self.case, self.calls = case, calls

    def set_timeout(self, seconds):
        self.calls.append('client.set_timeout')

    def get_client_version(self):
        self.calls.append('client.get_client_version')
        return '0.9.15'

    def get_server_version(self):
        self.calls.append('client.get_server_version')
        if self.case == 'connection_failure':
            raise RuntimeError('Synthetic connection failure')
        return 'wrong-version' if self.case == 'version_mismatch' else '0.9.15'

    def get_available_maps(self):
        self.calls.append('client.get_available_maps')
        return ['/Game/Carla/Maps/Town02']

    def get_world(self):
        self.calls.append('client.get_world')
        return World(self.case, self.calls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    import hashlib
    cfg = dict(expected_map='Town02', expected_version='0.9.15',
               expected_xodr_sha256=hashlib.sha256(XODR.encode()).hexdigest(),
               rpc_timeout_seconds=1, connect_seconds=0, identity_seconds=2,
               poll_seconds=0, stable_match_reads=2)
    calls = []
    signal.signal(signal.SIGTERM, terminated)
    context = dict(kind='SyntheticMockCARLA_NoServer', run_id=args.case,
                   server_argv=None, requested_server_cwd=None,
                   install='NotApplicableSynthetic', runtime_authorized=False)
    # Check the actual disk record at the exact start of validation.
    from tools import m8i1a_identity
    original = m8i1a_identity.validate_identity
    validations = []
    def checked(row, config):
        on_disk = json.loads((root / f"identity_{row['sequence']:03d}.json").read_text())
        assert on_disk == row
        validations.append(dict(sequence=row['sequence'], durable_before_check=True))
        if args.case == 'unexpected_exception':
            raise ValueError('Synthetic evaluator failure after identity persisted')
        original(row, config)
    m8i1a_identity.validate_identity = checked
    def sleep(seconds):
        if args.case in ['signal_termination', 'hard_signal']:
            persist(root / 'ready_for_signal.json', dict(identity_already_persisted=True))
            time.sleep(60)
    code = run_session(lambda: Client(args.case, calls), cfg, context, root,
                       cleanup=lambda: dict(ok=args.case != 'cleanup_failure', synthetic=True), sleep=sleep)
    persist(root / 'synthetic_call_trace.json', dict(calls=calls, validations=validations,
                                                   simulator_processes=0))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
