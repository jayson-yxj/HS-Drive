"""Minimal smoke test for a RUNNING CARLA 0.9.15 simulator.

Run: python scripts/check_carla.py --host 127.0.0.1 --port 2000
It does not spawn vehicles, change settings, or save data.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()
    try:
        import carla
        cli = carla.Client(args.host, args.port)
        cli.set_timeout(15.0)
        world = cli.get_world()
        print('MAP:', world.get_map().name)
        print('CLIENT:', cli.get_client_version())
        print('SERVER:', cli.get_server_version())
        if (cli.get_client_version(), cli.get_server_version()) != ('0.9.15', '0.9.15'):
            print('ERROR: both client/server must be 0.9.15', file=sys.stderr)
            return 2
    except Exception as exc:
        print('CARLA smoke check failed:', repr(exc), file=sys.stderr)
        return 1
    print('CARLA connection OK; RGB/Depth sensors NOT YET tested.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
