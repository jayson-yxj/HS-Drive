"""Synthetic test planner consumes current owner requests; never ticks/actuates."""
import argparse,json,time
from pathlib import Path
from tools.m8j13_common import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    seen=None
    atomic(root/'planner_ready.json',dict(model=False,actuator=False))
    while not (root/'planner_stop').exists():
        if not (root/'planner_request.json').exists():time.sleep(.005);continue
        row=json.loads((root/'planner_request.json').read_text())
        if row['sequence']==seen:time.sleep(.005);continue
        seen=row['sequence']
        if (root/'block_planner').exists():
            atomic(root/'planner_blocked.json',dict(wall_s=time.monotonic(),request=row))
            time.sleep(1.)
            atomic(root/'late_plan.json',dict(row,heartbeat_wall_s=time.monotonic(),source='SyntheticLatePlan'))
            while not (root/'planner_stop').exists():time.sleep(.01)
            break
        atomic(root/'planner_status.json',dict(row,heartbeat_wall_s=time.monotonic(),source='IndependentTestPlanner'))


if __name__=='__main__':main()
