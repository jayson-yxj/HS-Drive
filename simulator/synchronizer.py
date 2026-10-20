"""Frame-number join, bounded memory, nonblocking sensor callbacks.

Only the main thread touches frame buffers. Callbacks enqueue raw CARLA images;
conversion/copying happens in drain(). Cadence either follows the first sensor frame or an explicit capture boundary
when every-tick rendering is enabled; global even/odd parity is never assumed.
"""
from collections import Counter
from queue import Queue, Empty, Full
from threading import Lock
import numpy as np
from datasets.schema import FrameBundle, SensorSample, validate_bundle


class FrameSynchronizer:
    def __init__(self, required, *, max_frames=20, queue_size=400, timeout_ticks=20,
                 cadence_ticks=2, on_event=None):
        if min(max_frames, queue_size, timeout_ticks, cadence_ticks) < 1:
            raise ValueError('Buffer and time limits must be positive')
        self.required = tuple(required)
        self.max_frames, self.timeout_ticks = max_frames, timeout_ticks
        self.cadence_ticks = cadence_ticks
        self.queue = Queue(queue_size)
        self.pending, self.snapshots = {}, {}
        self.latest = -1
        self.next_expected = None
        self.closed_through = -1
        self.first_capture_frame = None
        self.explicit_cadence = False
        self.stats = Counter()
        self.lock = Lock()
        self.on_event = on_event or (lambda event: None)

    def begin_interval(self, first_frame, *, explicit_cadence=False):
        """Declare capture boundary after renderer warmup, before any snapshot.

        Delayed warmup callbacks are diagnostics, not missing capture frames.
        Never relabel their frame numbers or let them establish sensor cadence.
        explicit_cadence is only for sensors rendering every physics step; it
        fixes the retained frame phase before callbacks arrive.
        """
        if self.pending or self.snapshots or self.latest != -1:
            raise RuntimeError('Capture boundary must be set before snapshots')
        self.first_capture_frame = first_frame
        self.explicit_cadence = explicit_cadence
        if explicit_cadence:
            self.next_expected = first_frame

    def callback(self, sensor_id):
        def enqueue(image):
            try:
                self.queue.put_nowait((sensor_id, image))
            except Full:
                with self.lock:
                    self.stats['queue_overflow'] += 1
        return enqueue

    def event(self, reason, frame, **kwargs):
        self.stats[reason] += 1
        self.on_event({'reason': reason, 'frame_id': frame, **kwargs})

    def add_snapshot(self, frame, timestamp, ego, objects, traffic_rules=None):
        if frame <= self.latest:
            raise ValueError('World frame must increase: possible competing tick client')
        if self.latest >= 0 and frame != self.latest + 1:
            raise RuntimeError('World tick gap: another client may be ticking')
        self.latest = frame
        self.snapshots[frame] = (timestamp, ego, objects, traffic_rules)
        if self.next_expected is not None:
            while self.next_expected <= frame:
                if self.next_expected > self.closed_through:
                    self.pending.setdefault(self.next_expected, {})
                self.next_expected += self.cadence_ticks

    def drain(self):
        while True:
            try:
                name, image = self.queue.get_nowait()
            except Empty:
                break
            frame = image.frame if hasattr(image, 'frame') else image.frame_id
            if self.first_capture_frame is not None and frame < self.first_capture_frame:
                self.event('pre_interval_sensor', frame, sensor_id=name)
                continue
            # Every-tick rendering avoids independent UE interval scheduling.
            # Select one global phase, never accept a nearest/shifted sensor frame.
            if self.explicit_cadence and (frame-self.first_capture_frame) % self.cadence_ticks:
                self.stats['subsampled_sensor'] += 1
                continue
            if name not in self.required:
                self.event('unknown_sensor', frame, sensor_id=name)
                continue
            if frame <= self.closed_through or frame < self.latest - self.timeout_ticks:
                self.event('late_sensor', frame, sensor_id=name)
                continue
            if self.next_expected is None:
                self.next_expected = frame
                while self.next_expected <= self.latest:
                    self.pending.setdefault(self.next_expected, {})
                    self.next_expected += self.cadence_ticks
            group = self.pending.setdefault(frame, {})
            if name in group:
                self.event('duplicate_sensor', frame, sensor_id=name)
                continue
            if isinstance(image, SensorSample):
                sample = image
            else:
                sample = SensorSample(name, frame, image.timestamp, image.width, image.height,
                    bytes(image.raw_data), np.asarray(image.transform.get_matrix()))
            group[name] = sample
        ready = []
        for frame in sorted(self.pending):
            group = self.pending[frame]
            if len(group) == len(self.required) and frame in self.snapshots:
                timestamp, ego, objects, traffic_rules = self.snapshots[frame]
                bundle = FrameBundle(frame, timestamp, group, ego, objects, traffic_rules)
                try:
                    validate_bundle(bundle, self.required)
                except ValueError as exc:
                    self.event('invalid_frame', frame, detail=str(exc))
                else:
                    ready.append(bundle)
                    self.stats['complete'] += 1
                self.pending.pop(frame)
                self.closed_through = frame
            elif self.latest - frame >= self.timeout_ticks or len(self.pending) > self.max_frames:
                self.event('dropped_frame', frame, missing=sorted(set(self.required)-set(group)),
                           missing_snapshot=frame not in self.snapshots)
                self.pending.pop(frame)
                self.closed_through = frame
            else:
                # Preserve commit order even if later frames arrive sooner.
                break
        for frame in list(self.snapshots):
            if frame <= self.closed_through or frame < self.latest - self.timeout_ticks:
                self.snapshots.pop(frame)
        return ready

    def finish(self):
        ready = self.drain()
        for frame, group in sorted(self.pending.items()):
            self.event('shutdown_incomplete', frame, missing=sorted(set(self.required)-set(group)))
        self.pending.clear()
        self.snapshots.clear()
        return ready
