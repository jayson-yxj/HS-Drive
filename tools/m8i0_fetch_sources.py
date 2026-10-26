"""Bounded text-only source review. Never import or execute downloaded code.

Reproduction: python3 -m tools.m8i0_fetch_sources configs/m8i0/sources.json
Existing evidence is verified, never silently refreshed. HTTPS verification stays on.
"""
import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

LIMIT = 2 * 1024 * 1024
TOTAL_LIMIT = 25 * 1024 * 1024
HOSTS = {"raw.githubusercontent.com", "api.github.com", "huggingface.co",
         "carla.readthedocs.io", "www.nuscenes.org"}


def fetch(item):
    path = Path("reports/m8i0_sources") / item["name"]
    if path.name != item["name"]:
        raise ValueError("flat evidence filename required")
    url = item["url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username:
        raise ValueError("unapproved public HTTPS source")
    meta_path = path.with_name(path.name + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["url"] != url:
            raise ValueError("source changed; use a new evidence name")
        if meta["status"] == "Fetched":
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
        return meta
    meta = dict(item, checked_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        with urlopen(Request(url, headers={"User-Agent": "HS-Drive-M8-I0-source-review"}), timeout=20) as response:
            limit = min(item.get("max_bytes", LIMIT), LIMIT)
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError("source exceeds text budget")
            meta["transport_sha256"] = hashlib.sha256(body).hexdigest()
            if item.get("encoding") == "github_contents_base64":
                envelope = json.loads(body)
                if envelope.get("encoding") != "base64":
                    raise ValueError("unsupported contents response")
                meta["git_blob_sha"] = envelope["sha"]
                body = base64.b64decode(envelope["content"])
            body.decode("utf-8")
            if b"\x00" in body:
                raise ValueError("binary payload rejected")
            meta.update(status="Fetched", http_status=response.status,
                        bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            path.write_bytes(body)
    except HTTPError as exc:
        meta.update(status="Unavailable", http_status=exc.code, error_type=type(exc).__name__)
    except Exception as exc:
        # Do not log proxy URLs, credentials or private network details.
        meta.update(status="Unavailable", error_type=type(exc).__name__)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    items = json.loads(args.manifest.read_text())
    if len(items) > 60 or len({x["name"] for x in items}) != len(items):
        raise ValueError("at most 60 unique source files")
    # Predeclare an aggregate upper bound before any requests.
    declared = sum(x.get("max_bytes", LIMIT) for x in items)
    if declared > TOTAL_LIMIT:
        raise ValueError("declared aggregate source budget exceeded")
    Path("reports/m8i0_sources").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, items))
    total = sum(x.get("bytes", 0) for x in rows)
    if total > TOTAL_LIMIT:
        raise ValueError("aggregate source budget exceeded")
    for row in rows:
        print(row["name"], row["status"], row.get("bytes", row.get("http_status", row.get("error_type"))))
    print("Total fetched bytes:", total)


if __name__ == "__main__":
    main()
