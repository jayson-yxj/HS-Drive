"""Isolated real six-RGB buffer IPC and serialization microbenchmark (no simulator)."""

import json, time, pickle, multiprocessing as mp
from pathlib import Path
import cv2, numpy as np
from tools.m5l_common import write, summary, asset


def echo_worker(incoming, outgoing):
    while True:
        value = incoming.get()
        if value is None:
            return
        outgoing.put(
            dict(
                index=value["index"],
                bytes=sum(len(b) for b in value["buffers"]),
                received_wall=time.perf_counter(),
            )
        )


def main():
    root = Path("/root/gpufree-data/hs-drive-data/raw/m5m_dev_v3")
    m = json.loads((next(root.glob("*/manifest.json"))).read_text())
    scene = root / m["group"]
    f = json.loads((scene / "frames" / f"{m['frames'][0]:08d}.json").read_text())
    buffers = []
    for entry in f["cameras"].values():
        im = cv2.imread(str(scene / entry["rgb"]))
        bgra = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        buffers.append(bgra.tobytes())
    ctx = mp.get_context("spawn")
    incoming = ctx.Queue(1)
    outgoing = ctx.Queue(1)
    p = ctx.Process(target=echo_worker, args=(incoming, outgoing))
    p.start()
    records = []
    try:
        for i in range(31):
            value = dict(index=i, buffers=buffers)
            t = time.perf_counter()
            encoded = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            serialize = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            pickle.loads(encoded)
            deserialize = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            incoming.put_nowait(value)
            put = (time.perf_counter() - t) * 1000
            reply = outgoing.get(timeout=20)
            end = time.perf_counter()
            records.append(
                dict(
                    index=i,
                    cold=i == 0,
                    serialize_ms=serialize,
                    deserialize_ms=deserialize,
                    enqueue_call_ms=put,
                    receive_age_ms=(reply["received_wall"] - t) * 1000,
                    round_trip_ms=(end - t) * 1000,
                    bytes=reply["bytes"],
                )
            )
    finally:
        incoming.put(None)
        p.join(10)
        if p.is_alive():
            p.terminate()
            p.join(5)
    write("runs/m5m_logs/ipc_probe_raw.json", records)
    write(
        "reports/m5m_ipc_probe.json",
        dict(
            kind="actual_server_recorded_sixRGB_payload_microbenchmark_no_CARLA_no_GPU",
            cold=records[0],
            steady={
                k: summary([r[k] for r in records[1:]])
                for k in [
                    "serialize_ms",
                    "deserialize_ms",
                    "enqueue_call_ms",
                    "receive_age_ms",
                    "round_trip_ms",
                ]
            },
            bytes=records[0]["bytes"],
            raw=asset("runs/m5m_logs/ipc_probe_raw.json"),
            limits="Isolated timing, not additive decomposition of concurrent CARLA trial; uncompressed recorded RGB converted to same BGRA buffer size. Echo-worker startup included cold roundtrip.",
        ),
    )


if __name__ == "__main__":
    main()
