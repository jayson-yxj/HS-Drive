"""Append-only isolated GPU launcher, no dependency installation."""
from tools.m8b_common import *
def main():
    env=environment()
    run('real_rgb_gpu',['unshare','--net',str(RUNTIME/'env/bin/python'),'-m','tools.m8b_infer','--manifest',str(ROOT/'rgb_bundle/manifest.json'),'--output',str(ROOT/'forward')],timeout=1800,env=env)
if __name__=='__main__':main()
