"""Executed only by the single-use M8-I1 supervisor."""
import argparse
import signal
from simulator.m8i1_session import TrafficDevelopmentSessionOwner


def abort(signum, frame):
    raise RuntimeError('SupervisorTerminationSignal:'+str(signum))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=float,required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,abort)
    TrafficDevelopmentSessionOwner(a.start).run()
