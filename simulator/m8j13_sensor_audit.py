"""Value-only camera provenance and generation isolation; no world advancement."""
from collections import Counter
from dataclasses import dataclass, asdict
from queue import Queue, Empty, Full
from threading import Lock
import math
import time


@dataclass(frozen=True)
class RawImageStamp:
    run: str
    episode: int
    sensor_generation: str
    camera: str
    actor_id: int
    frame: int
    timestamp_s: float
    transform: tuple
    receive_monotonic_s: float

    def record(self):
        return asdict(self)


class RawInbox:
    """Only metadata copy + bounded enqueue in callbacks. Never read snapshots.

    Closing invalidates callbacks before listeners stop; an already executing
    callback must recheck under the lock before enqueueing. A new rig gets a new
    inbox, even when CARLA reuses frame numbers or actor IDs.
    """
    def __init__(self, run, episode, generation, capacity=256):
        self.identity = (run, episode, generation)
        self.queue = Queue(capacity)
        self.lock = Lock()
        self.active = True
        self.stats = Counter()

    def callback(self, camera, actor_id):
        identity = self.identity

        def receive(image):
            arrival = time.monotonic()
            stamp = RawImageStamp(*identity, camera, actor_id, int(image.frame),
                                  float(image.timestamp),
                                  tuple(tuple(float(x) for x in row)
                                        for row in image.transform.get_matrix()), arrival)
            with self.lock:
                if not self.active or identity != self.identity:
                    self.stats['old_generation_rejected'] += 1
                    return
                try:
                    self.queue.put_nowait((stamp, image))
                except Full:
                    self.stats['queue_overflow'] += 1
        return receive

    def drain(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except Empty:
                return

    def close(self):
        with self.lock:
            self.active = False
            while True:
                try:
                    self.queue.get_nowait()
                    self.stats['close_queue_discarded'] += 1
                except Empty:
                    return


def check_group(records, snapshot, identity, actors, tolerance=1e-5):
    """Keep ALL candidates; duplicates never silently select a winning sample."""
    errors = []
    selected = {}
    for name, actor in actors.items():
        candidates = [r for r in records if r['camera'] == name]
        if len(candidates) != 1:
            errors.append('MissingCamera' if not candidates else 'DuplicateCamera')
        for r in candidates:
            if (r['run'], r['episode'], r['sensor_generation']) != tuple(identity):
                errors.append('OldGeneration')
            if r['actor_id'] != actor:
                errors.append('WrongSensorInstance')
            if r['frame'] != snapshot['frame']:
                errors.append('MixedFrame')
            if not math.isfinite(r['timestamp_s']) or abs(r['timestamp_s']-snapshot['elapsed_seconds']) > tolerance:
                errors.append('MixedSimulationTimestamp')
        if len(candidates) == 1:
            selected[name] = candidates[0]
    if any(r['camera'] not in actors for r in records):
        errors.append('UnknownCamera')
    return dict(valid=not errors, errors=sorted(set(errors)), sources=selected,
                candidates=records, snapshot=snapshot)
