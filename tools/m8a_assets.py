"""Bounded public official-entry acquisition. No pip, torch, CARLA, or bypasses.

Only a downloaded checkpoint candidate is reported here; this can NEVER grant
the subsequent dependency gate. CPU checkpoint validation is separate.
"""
import argparse
import html
import http.cookiejar
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tools.m8a_common import PROTOCOL, asset, provenance, sha, write


def is_checkpoint_candidate(path, minimum=1000000):
    p = Path(path)
    if not p.exists() or p.stat().st_size < minimum:
        return False
    with p.open("rb") as stream:
        signature = stream.read(4)
    return signature.startswith(b"PK\x03\x04") or signature[:1] == b"\x80"


def asset_gate(checks):
    required = ("official_source", "usage_scope_confirmed", "complete_bytes",
                "sha256_recorded", "readable_isolated", "detection_head_present",
                "key_shapes_match")
    return all(checks.get(k) is True for k in required)


def access_barrier(text):
    lowered = text.lower()
    return any(t in lowered for t in (
        "access denied", "request access", "too many users", "quota exceeded",
        "download quota", "verify you are human",
    ))


def _safe_url(url):
    # Do not preserve live session tokens, cookies, or signed download URLs.
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))


class Attempt:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.events = []
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))

    def event(self, value):
        self.events.append(value)
        write(self.output / f"event_{len(self.events):02d}.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def remaining(self):
        remaining = self.config["asset_wall_seconds_max"] - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError("asset_wall_budget_exhausted")
        return remaining

    def curl(self, name, url, *, download=False):
        target = self.output / (name + ".response")
        net = self.config["network"]
        timeout = min(self.remaining(), net["download_timeout_s"] if download
                      else net["request_timeout_s"])
        command = ["curl", "--disable", "--silent", "--show-error", "--location",
                   "--proto", "=https", "--proto-redir", "=https",
                   "--connect-timeout", str(net["connect_timeout_s"]),
                   "--max-time", str(timeout), "--max-redirs", str(net["max_redirects"]),
                   "--retry", "0", "--max-filesize", str(net["max_response_bytes"]),
                   "--output", str(target), "--write-out",
                   "%{http_code}\t%{size_download}\t%{time_total}\t%{content_type}", url]
        start = time.monotonic()
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=timeout + 5)
            record = dict(name=name, request_url=_safe_url(url), transport="curl_no_config_no_retries",
                          returncode=result.returncode, transfer=result.stdout,
                          error=result.stderr[-2000:], elapsed_s=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            record = dict(name=name, request_url=_safe_url(url), returncode=None,
                          error="subprocess_deadline", elapsed_s=time.monotonic()-start)
        if target.exists():
            record["response"] = asset(target)
            record["checkpoint_candidate"] = is_checkpoint_candidate(
                target, self.config["checkpoint"]["minimum_candidate_bytes"])
        self.event(record)
        return target, record

    def request(self, name, url, data=None):
        timeout = min(self.remaining(), self.config["network"]["request_timeout_s"])
        encoded = None if data is None else urllib.parse.urlencode(data).encode()
        req = urllib.request.Request(url, data=encoded, headers={
            "User-Agent": "Mozilla/5.0", "Referer": self.config["network"]["baidu_share"]})
        start = time.monotonic()
        try:
            with self.opener.open(req, timeout=timeout) as response:
                body = response.read(4_000_001)
                if len(body) > 4_000_000:
                    raise ValueError("metadata_response_too_large")
                record = dict(name=name, request_url=_safe_url(url), status=response.status,
                              bytes=len(body), elapsed_s=time.monotonic()-start)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            self.event(dict(name=name, request_url=_safe_url(url),
                            error=type(exc).__name__, elapsed_s=time.monotonic()-start))
            return None
        # Persist no cookies/session secrets. JSON listing is filtered separately.
        record["response_sha256"] = __import__("hashlib").sha256(body).hexdigest()
        self.event(record)
        return body.decode("utf-8", errors="replace")

    def baidu(self):
        net = self.config["network"]
        page = self.request("baidu_share", net["baidu_share"])
        if page is None:
            return None
        verify_url = "https://pan.baidu.com/share/verify?surl=-JkzOxKy4isMiiNHd20Z-w"
        response = self.request("baidu_public_code", verify_url,
                                {"pwd": net["baidu_public_code"]})
        if response is None:
            return None
        try:
            verified = json.loads(response)
        except ValueError:
            self.event(dict(name="baidu_verify", result="NonJSON_Stop"))
            return None
        self.event(dict(name="baidu_verify", errno=verified.get("errno"),
                        result="Verified" if verified.get("errno")==0 else "Stop_NoBypass"))
        if verified.get("errno") != 0:
            return None
        page = self.request("baidu_verified_page", net["baidu_share"])
        if page is None:
            return None

        def field(name):
            pattern = r'["\x27]' + re.escape(name) + r'["\x27]\s*:\s*(?:["\x27]([^"\x27]+)["\x27]|([0-9]+))'
            match = re.search(pattern, page)
            return next((v for v in match.groups() if v is not None), None) if match else None

        shareid, uk = field("shareid") or field("share_id"), field("uk")
        if not shareid or not uk:
            self.event(dict(name="baidu_fields", result="MissingPublicShareMetadata_Stop"))
            return None
        query = urllib.parse.urlencode({"shareid": shareid, "uk": uk,
            "dir": net["baidu_directory"], "page": 1, "num": 100, "order": "name", "desc": 0})
        listing_raw = self.request("baidu_directory", "https://pan.baidu.com/share/list?" + query)
        if listing_raw is None:
            return None
        try:
            listing = json.loads(listing_raw)
        except ValueError:
            self.event(dict(name="baidu_listing", result="NonJSON_Stop"))
            return None
        public = [{k: x.get(k) for k in ("server_filename", "fs_id", "size", "isdir")}
                  for x in listing.get("list", [])]
        write(self.output/"baidu_listing.json", dict(errno=listing.get("errno"), files=public))
        matches = [x for x in listing.get("list", []) if x.get("server_filename")
                   == self.config["checkpoint"]["expected_filename"] and not x.get("isdir")]
        if listing.get("errno") != 0 or len(matches) != 1:
            self.event(dict(name="baidu_matching_file", result="NoUniqueOfficialFile_Stop"))
            return None
        item = matches[0]
        sign, timestamp = field("sign"), field("timestamp")
        cookie = next((c.value for c in self.jar if c.name == "BDCLND"), None)
        if not sign or not timestamp or not cookie:
            self.event(dict(name="baidu_download", result="PublicSessionMetadataUnavailable_Stop",
                            listed_bytes=item["size"], file_obtained=False))
            return None
        query = urllib.parse.urlencode({"sign": sign, "timestamp": timestamp,
                                       "channel": "chunlei", "web": 1, "app_id": 250528})
        response = self.request("baidu_download_request",
            "https://pan.baidu.com/api/sharedownload?" + query,
            {"encrypt": 0, "extra": json.dumps({"sekey": urllib.parse.unquote(cookie)}),
             "fid_list": json.dumps([item["fs_id"]]), "primaryid": shareid,
             "uk": uk, "product": "share", "type": "nolimit"})
        if response is None:
            return None
        try:
            result = json.loads(response)
        except ValueError:
            self.event(dict(name="baidu_download_result", result="NonJSON_Stop"))
            return None
        self.event(dict(name="baidu_download_result", errno=result.get("errno"),
                        listed_bytes=item["size"], file_obtained=False))
        if result.get("errno") != 0:
            return None
        files = result.get("list", [])
        if len(files) != 1 or not files[0].get("dlink"):
            return None
        # Only use the official service's own signed link. Never log its query.
        url = files[0]["dlink"]
        host = urllib.parse.urlsplit(url).hostname or ""
        if not (host.endswith(".baidu.com") or host.endswith(".baidupcs.com")):
            self.event(dict(name="baidu_download_host", result="UnexpectedHost_Stop"))
            return None
        target, record = self.curl("baidu_checkpoint", url, download=True)
        if record.get("returncode") == 0 and record.get("checkpoint_candidate"):
            if target.stat().st_size != item["size"]:
                self.event(dict(name="baidu_size", result="Incomplete_Stop"))
                return None
            return target
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        raise ValueError("invalid_run_id")
    config = json.loads(PROTOCOL.read_text())
    out = Path("reports/m8a_execution_logs")/args.run_id
    attempt = Attempt(config, out)
    source = Path(config["historical_source"])
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                   text=True).strip() == config["official_commit"]
    official = {}
    for filename, key in [("README.md", "official_readme_sha256"),
                          ("LICENSE", "official_license_sha256"),
                          (config["official_config"], "official_config_sha256")]:
        assert sha(source/filename) == config[key], filename
        official[filename] = asset(source/filename)
    readme = (source/"README.md").read_text()
    assert config["network"]["google_view"] in readme
    assert config["network"]["baidu_share"] in readme
    row = next(line for line in readme.splitlines() if "| PETR-r50-c5-1408x512 " in line)
    write(out/"freeze.json", dict(**provenance("FrozenOfficialAssetAcquisition"),
          official_files=official, model_row=row, protocol=config))
    checkpoint = None
    try:
        target, view = attempt.curl("google_view", config["network"]["google_view"])
        text = target.read_text(errors="replace") if target.exists() and target.stat().st_size < 4_000_000 else ""
        if access_barrier(text):
            attempt.event(dict(name="google_access", result="AccessBarrier_StopProvider"))
        else:
            target, download = attempt.curl("google_checkpoint", config["network"]["google_download"],
                                            download=True)
            if download.get("returncode") == 0 and download.get("checkpoint_candidate"):
                checkpoint = target
        if checkpoint is None:
            checkpoint = attempt.baidu()
    except Exception as exc:
        attempt.event(dict(name="asset_exception", error=type(exc).__name__, result="Stop"))
    report = dict(**provenance("ActualServerOfficialAssetAcquisition"),
        status="BytesObtained_ValidationRequired" if checkpoint else "AssetNoGo",
        official_source_confirmed=True, official_files=official,
        checkpoint=asset(checkpoint) if checkpoint else None,
        elapsed_s=time.monotonic()-attempt.started, events=attempt.events,
        dependency_install_authorized=False, checkpoint_loaded=False,
        safety_gates_unchanged=True,
        limitation="Official filename/listing/header alone cannot establish a valid trained detector.")
    write(out/"outcome.json", report)
    print(json.dumps({"outcome":str(out/"outcome.json"),"status":report["status"]}), flush=True)


if __name__ == "__main__":
    main()
