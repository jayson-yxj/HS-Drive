"""Append-only artifacts for the PETR P0; never imports CARLA or training code."""
import hashlib
import json
import subprocess
from pathlib import Path

BASE = "478070b51f03daf7e4ff2e7cb0b0a96e7507d9ba"
PROTOCOL = Path("configs/m8a/protocol.json")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def asset(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def provenance(kind):
    paths = [PROTOCOL, *sorted(Path("tools").glob("m8a_*.py")),
             *sorted(Path("tests").glob("test_m8a*.py"))]
    return {
        "kind": kind, "base": BASE,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sources": {str(p): sha(p) for p in paths},
        "protocol_sha256": sha(PROTOCOL),
        "training_steps": 0, "visual_control": False,
    }
