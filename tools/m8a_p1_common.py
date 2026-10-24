"""Isolated P0 provenance; append-only output and explicit subprocess network."""
import json
import os
from pathlib import Path
import subprocess
import time
from tools.m8a_common import asset,sha,write

CONFIG=Path("configs/m8a/p1_protocol.json")
CFG=json.loads(CONFIG.read_text())
ROOT=Path(CFG["root"])
LOG=Path("reports/m8a_p1_logs")


def provenance(kind):
    paths=[CONFIG,*sorted(Path("tools").glob("m8a_p1_*.py")),
           *sorted(Path("tests").glob("test_m8a_p1*.py"))]
    return dict(kind=kind,base=CFG["base"],
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        sources={str(p):sha(p) for p in paths},training_steps=0,visual_control=False)


def environment(network=False):
    # No inherited credentials, proxy configuration or user package settings.
    env={"PATH":f"{ROOT}/env/bin:/usr/bin:/bin","HOME":str(ROOT/"home"),
         "LANG":"C.UTF-8","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1",
         "PIP_CONFIG_FILE":"/dev/null","PIP_DISABLE_PIP_VERSION_CHECK":"1","PIP_CACHE_DIR":str(ROOT/"wheel-cache"),
         "OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","MAX_JOBS":"4"}
    if network:env.update(HTTP_PROXY=CFG["proxy"],HTTPS_PROXY=CFG["proxy"])
    return env


def run(name,command,network=False,timeout=900,cwd=None,extra_env=None):
    LOG.mkdir(exist_ok=True)
    env=environment(network)
    if extra_env:env.update(extra_env)
    started=time.monotonic()
    with (LOG/f"{name}.txt").open("x") as output:
        try:
            result=subprocess.run(command,env=env,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,timeout=timeout)
            rc=result.returncode
        except subprocess.TimeoutExpired:
            rc=124
    write(LOG/f"{name}.json",dict(**provenance("ActualIsolatedSubprocess"),
        command=command,cwd=str(cwd) if cwd else None,exit_code=rc,
        elapsed_s=time.monotonic()-started,log=asset(LOG/f"{name}.txt"),
        explicit_proxy=network,timeout_s=timeout))
    print(name,"exit",rc,flush=True)
    if rc:raise RuntimeError(f"{name} failed; see retained log")
