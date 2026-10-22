"""Append-only M6-C artifacts and explicit experiment provenance."""
import json
import subprocess
from pathlib import Path
from simulator.recorder import file_hash

BASE = "f047ade6d7802205abadcde4363a57e0a1d00981"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def provenance(kind):
    paths = []
    for directory in ("tools", "datasets", "diagnostics", "tests", "simulator", "configs/m6c"):
        paths.extend(p for p in Path(directory).rglob("*") if p.is_file()
                     and "__pycache__" not in str(p)
                     and ("m6c" in p.name.lower() or directory == "configs/m6c"))
    return dict(kind=kind, base=BASE,
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                sources={str(p): file_hash(p) for p in sorted(paths)},
                protocol_sha256=file_hash("configs/m6c/protocol.json"),
                visual_control=False)


def asset(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=file_hash(p))
