"""Versioned repeated affinity/frequency isolation. Single CARLA tick owner."""
import json,subprocess,time,threading,os
from pathlib import Path
from tools.m5e_resources import snapshot
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5e_runtime_isolation.json')
    if dest.exists():raise FileExistsError(dest)
    cpus=sorted(os.sched_getaffinity(0));spec=dict(tick=cpus[:2],worker=cpus[2:4],load=cpus[4:12]);cases=[('default_mixed',False,False,True,10),('tick_reserved',True,False,True,10),('split_affinity',True,True,True,10),('split_lowgpu',True,True,True,5),('timeshare_quiet',True,True,False,0)]
    atomic_json('reports/m5e_runtime_protocol.json',dict(cpus=spec,cases=cases,repeats=2,ticks=1200,route_distance_m=100,order='forward then reverse',period_gates=dict(p99_ms=55,max_ms=100,deadline_fraction=.001),scope='Actualegomotion thenpark separately; explicitdynamiclead and faults appended. No serveraffinity alteration: affinityset client/worker/load only. Resourcepartition doesnotreserveexclusiveCPU from unrelatedhosttasks.'))
    records=[]
    for repeat in range(2):
        for case,tick,worker,load,hz in (cases if repeat==0 else cases[::-1]):
            name=f'{case}_{repeat}';loads=[];tele=[];stop=threading.Event()
            if load:
                for module in ['cpu','gpu']:
                    cmd=['.venv-gpu/bin/python','-m',f'tools.m5d_{module}_load','--duration','80','--report',f'reports/m5e_{name}_{module}load.json']
                    if module=='gpu':cmd+=['--hz',str(hz)]
                    if tick:cmd=['taskset','-c',','.join(map(str,spec['load'])),*cmd]
                    log=open(f'reports/m5e_logs/m5e_{name}_{module}load.log','w');loads.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
            def monitor():
                while not stop.is_set():tele.append(snapshot());stop.wait(1)
            thread=threading.Thread(target=monitor,daemon=True);thread.start();path=f'reports/m5e_runtime_{name}.json';cmd=['.venv/bin/python','-m','tools.m5e_async_drive','--output',path,'--artifacts',f'reports/m5e_runtime_{name}','--scenarios','straight','--ticks','1200','--route-distance','100','--hold-after-route','--load-description',name]
            if tick:cmd+=['--tick-cpus',','.join(map(str,spec['tick']))]
            if worker:cmd+=['--worker-cpus',','.join(map(str,spec['worker']))]
            with open(f'reports/m5e_logs/m5e_runtime_{name}.log','w') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            stop.set();thread.join();loadcodes=[]
            for proc,log in loads:loadcodes.append(proc.wait());log.close()
            atomic_json(f'reports/m5e_runtime_{name}_resources.json',dict(kind='actual_proc_gpu_1Hz',records=tele,clock_ticks_per_s=os.sysconf('SC_CLK_TCK')))
            records.append(dict(case=case,repeat=repeat,exit_code=code,load_exit_codes=loadcodes,command=cmd,report=path,results=json.loads(Path(path).read_text())['results'] if Path(path).exists() else []));atomic_json('reports/m5e_runtime_progress.json',dict(records=records));print(name,code,flush=True)
            if code:raise RuntimeError('Stop failed isolation trial; preserve output')
    atomic_json(dest,dict(kind='actual_CARLA_two_repeats_affinity_frequency_isolation',records=records,source_sha256=file_hash(__file__),protocol_sha256=file_hash('reports/m5e_runtime_protocol.json'),limits='Fixedsamehost; notexclusivehardware. Reverseorder secondround mitigatesnoteliminates temporalconfounding. No shorterSafety/noexpiredresultreuse. SavedRGBinference pressure notfullonlinesensorstream. Serverdefaultaffinity unchanged; timesharequiet runsnoinferencelabeljob.'))


if __name__=='__main__':main()
