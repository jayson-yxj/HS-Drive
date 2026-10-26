"""Read-only runner. A business failure is a nonzero process exit after durable logs."""
import argparse
import json
import signal
from pathlib import Path
from tools.m8i1a_common import config
from tools.m8i1a_identity import run_session, terminated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text())
    if context.get('runtime_authorized') is not True:
        parser.error('Supervisor authorization evidence required')
    cfg = config()
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    def factory():
        import carla
        return carla.Client('127.0.0.1', cfg['port'])
    return run_session(factory, cfg, context, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
