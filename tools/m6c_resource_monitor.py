"""Out-of-loop process/resource diagnostics; no tick ownership, no CUDA context."""
import argparse,json,time,subprocess,os
from pathlib import Path


def proc(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text();values=stat[stat.rfind(')')+2:].split()
        return dict(pid=pid,utime=int(values[11]),stime=int(values[12]),cpu=int(values[36]),
            schedstat=Path(f'/proc/{pid}/schedstat').read_text().strip(),
            cmdline=Path(f'/proc/{pid}/cmdline').read_text().replace('\0',' '))
    except (OSError,IndexError):return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--until',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    gpu=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv','-l','1'],stdout=open(a.output+'.gpu.csv','x'),stderr=subprocess.STDOUT)
    records=[]
    try:
        while not Path(a.until).exists():
            start=time.perf_counter();ps=[]
            for f in Path('/proc').iterdir():
                if not f.name.isdigit():continue
                q=proc(f.name)
                if q and not any(v in q['cmdline'].split(' ',1)[0] for v in ['bash','sh']) and any(v in q['cmdline'] for v in ['CarlaUE4-Linux-Shipping','tools.m6c_drive','multiprocessing.spawn']):ps.append(q)
            records.append(dict(wall=start,loadavg=os.getloadavg(),processes=ps,source='Privileged_resource_observer'))
            time.sleep(max(0,.2-(time.perf_counter()-start)))
    finally:
        gpu.terminate();gpu.wait()
        with Path(a.output).open('x') as f:
            for r in records:f.write(json.dumps(r)+'\n')

if __name__=='__main__':main()
