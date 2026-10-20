"""Opt-in process placement only. Original async scheduling/safety unchanged."""
import os
from pathlib import Path
from simulator.m5a_async import AsyncPlanner


class AffinityPlanner(AsyncPlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        spec=os.environ.get('M5E_WORKER_CPUS','')
        if spec:
            cpus=set(map(int,spec.split(',')))
            for task in Path(f'/proc/{self.worker.pid}/task').iterdir():
                try:os.sched_setaffinity(int(task.name),cpus)
                except ProcessLookupError:pass
