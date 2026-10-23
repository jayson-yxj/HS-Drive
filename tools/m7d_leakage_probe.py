"""Synthetic forbidden read attempt against the real planner subprocess guard."""
from tools import m7d_replay as replay
from tools.m7d_common import write
original=replay.plan

def probe(inp,*args,**kwargs):
    try:
        open('reports/m7d_evaluator/dev_constant_r3/00000.json').read()
    except PermissionError:
        pass
    else:
        raise AssertionError('future_read_not_blocked')
    return original(inp,*args,**kwargs)

def main():
    replay.plan=probe
    replay.write=lambda p,d:write('reports/m7d_forbidden_future_probe.json',d)
    replay.run('dev_constant_r3','stream',1)
if __name__=='__main__':main()
