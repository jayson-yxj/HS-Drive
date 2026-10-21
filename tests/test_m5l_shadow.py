import queue
from pathlib import Path
from simulator.m5l_shadow import latest_put

def test_latest_bounded_nonblocking():
    q=queue.Queue(1);assert latest_put(q,1);assert latest_put(q,2);assert q.get_nowait()==2

def test_worker_cannot_tick_or_control():
    text=Path('simulator/m5l_shadow.py').read_text()
    assert 'import carla'not in text and '.tick('not in text and 'apply_control('not in text
