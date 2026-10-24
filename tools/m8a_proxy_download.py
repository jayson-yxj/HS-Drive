"""Only the pinned official public model; bounded gdown, TLS on, redacted logs."""
import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import time

from tools.m8a_common import asset,sha,write
from tools.m8a_proxy_audit import CONFIG,provenance,redact


def main():
    import gdown
    cfg=json.loads(CONFIG.read_text())
    # Prevent credential/config inheritance by gdown. The proxy is explicit.
    for k in list(os.environ):
        if k.lower().endswith("_proxy"):
            del os.environ[k]
    target=Path(cfg["checkpoint_output"])
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError("Fresh attempt must not overwrite an existing checkpoint")
    source=Path("/root/gpufree-data/hs-drive-experiments/m6a-petr/source")
    old=json.loads(Path("configs/m8a/protocol.json").read_text())
    assert sha(source/"README.md")==old["official_readme_sha256"]
    assert sha(source/cfg["official_config"])==old["official_config_sha256"]
    assert cfg["sites"]["google_drive"] in (source/"README.md").read_text()
    stream=io.StringIO()
    started=time.monotonic()
    error=None
    result=None
    try:
        with contextlib.redirect_stderr(stream), contextlib.redirect_stdout(stream):
            result=gdown.download(url=cfg["sites"]["google_drive"],output=str(target),
                quiet=True,proxy=cfg["discovered_proxy"],use_cookies=False,
                verify=True,resume=True,timeout=cfg["download_inactivity_timeout_s"],
                retries=cfg["download_retries"])
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=redact(str(exc)))
    records=[]
    for d in sorted(importlib.metadata.distributions(),key=lambda d:d.metadata["Name"].lower()):
        records.append(dict(name=d.metadata["Name"],version=d.version))
    write("reports/m8a_proxy_retry_logs/download_env_lock.json",
          dict(**provenance("ActualIsolatedDownloaderEnvironment"),packages=records,
               gdown_module=asset(gdown.__file__)))
    write("reports/m8a_proxy_retry_logs/gdown_result.json",dict(
        **provenance("ActualOfficialGoogleDriveDownload"),url=cfg["sites"]["google_drive"],
        output=asset(target) if target.exists() else None,completed=result is not None,
        elapsed_s=time.monotonic()-started,error=error,tool_version=gdown.__version__,
        tls_verify=True,cookies=False,explicit_proxy=cfg["discovered_proxy"],
        retries_max=cfg["download_retries"],timeout_s=cfg["download_inactivity_timeout_s"],
        captured_log=redact(stream.getvalue()),
        official_source=[asset(source/"README.md"),asset(source/cfg["official_config"])],
        third_party_weights=False,model_deserialization=False))
    print(json.dumps(dict(completed=result is not None,bytes=target.stat().st_size
                          if target.exists() else 0,error=error),ensure_ascii=False))


if __name__=="__main__":
    main()
