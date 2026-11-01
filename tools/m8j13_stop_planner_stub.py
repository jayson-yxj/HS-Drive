"""Independent synthetic planning heartbeat for fault tests; no CARLA client."""
import argparse
from pathlib import Path
import time
from tools.m8j13_common import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    while not (root/'stub_stop').exists():
        if (root/'block_response').exists():
            # The blocked worker cannot renew normal-command validity.
            time.sleep(1.)
            atomic(root/'late_response.json',dict(source='SyntheticLatePlan',throttle=.25,brake=0.,steer=0.))
            while not (root/'stub_stop').exists():time.sleep(.02)
            break
        now=time.monotonic();atomic(root/'heartbeat.json',dict(wall_s=now,command_valid_until_wall_s=now+.3,
            armed=(root/'ARM').exists(),source='IndependentSyntheticPlanner'))
        time.sleep(.02)


if __name__=='__main__':main()
