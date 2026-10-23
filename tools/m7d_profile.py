"""Diagnostic cProfile outside reported clean latency experiment."""
import cProfile,pstats,io
from pathlib import Path
from tools import m7d_replay as replay
from tools.m7d_common import write


def main():
    original=replay.plan;profile=cProfile.Profile()
    def profiled(*args,**kwargs):
        profile.enable()
        try:return original(*args,**kwargs)
        finally:profile.disable()
    replay.plan=profiled
    replay.write=lambda p,d:write('reports/m7d_profile_diagnostic.json',d)
    replay.run('dev_constant_r3','stream',3)
    stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(60)
    with Path('reports/m7d_execution_logs/function_profile.txt').open('x') as f:f.write(stream.getvalue())
if __name__=='__main__':main()
