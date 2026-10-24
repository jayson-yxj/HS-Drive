"""Archive integrity and reproducible restricted CPU inspection launcher."""
import argparse
import json
from pathlib import Path
import pickletools
import resource
import subprocess
import zipfile

from tools.m8a_common import asset,write
from tools.m8a_proxy_audit import CONFIG,provenance,redact


def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(90,90))
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(40*1024**2,40*1024**2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",default="v2",choices=["v2"])
    args=parser.parse_args()
    cfg=json.loads(CONFIG.read_text())
    p=Path(cfg["checkpoint_output"])
    if not p.exists():raise FileNotFoundError(p)
    if not zipfile.is_zipfile(p):raise ValueError("Not a complete ZIP checkpoint")
    with zipfile.ZipFile(p) as z:
        bad=z.testzip()
        files=z.infolist()
        names=[x.filename for x in files if x.filename.endswith("/data.pkl")]
        assert len(names)==1 and bad is None
        data=z.read(names[0])
        globals_used=sorted({arg for op,arg,pos in pickletools.genops(data) if op.name=="GLOBAL"})
        static=dict(file=asset(p),kind="ActualZIPCRCAndStaticPickleInspection_NoExecution",
                    zip_entries=len(files),crc_bad_member=bad,all_crc_passed=True,
                    total_uncompressed_bytes=sum(x.file_size for x in files),
                    pickle_globals=globals_used,html_error_page=False,
                    size_matches_historical_listing=p.stat().st_size==435218261,
                    official_expected_sha256=None,
                    local_sha_is_not_independent_authenticity_proof=True)
    root=p.parent.parent
    out=root/("inspection_"+args.run_id)
    out.mkdir(exist_ok=False)
    script=Path("tools/m8a_proxy_inspect.py").resolve()
    runtime=Path(".venv-gpu/lib/python3.10/site-packages").resolve()
    cmd=["bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL",
        "--ro-bind","/usr","/usr","--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64",
        "--dir","/etc","--ro-bind","/etc/ld.so.cache","/etc/ld.so.cache",
        "--ro-bind",str(root/"inspect-env"),"/inspect-env",
        "--ro-bind",str(runtime),"/runtime","--ro-bind",str(p),"/input/epoch_24.pth",
        "--ro-bind",str(script),"/checkpoint_reader.py","--bind",str(out),"/output",
        "--tmpfs","/tmp","--tmpfs","/proc","--dev","/dev","--clearenv",
        "--setenv","PATH","/usr/bin","--setenv","PYTHONPATH","/runtime",
        "--setenv","PYTHONDONTWRITEBYTECODE","1","--setenv","CUDA_VISIBLE_DEVICES","",
        "--setenv","OMP_NUM_THREADS","1","--setenv","M8A_ISOLATED","1",
        "--chdir","/tmp","/inspect-env/bin/python","-s","/checkpoint_reader.py"]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=120,preexec_fn=limits)
    write(f"reports/m8a_proxy_retry_logs/checkpoint_archive_{args.run_id}.json",dict(**provenance(static.pop("kind")),**static))
    write(f"reports/m8a_proxy_retry_logs/isolated_load_execution_{args.run_id}.json",
        dict(**provenance("ActualRestrictedCheckpointLoadExecution"),command=cmd,
             returncode=result.returncode,stdout=redact(result.stdout),stderr=redact(result.stderr),
             filesystem_mode="ReadOnlySystemAndTrustedRuntime_NoProjectOrCredentials",
             network_namespace="unshare-all",gpu="NoNvidiaDevices_CPUOnly",
             memory_limit_gib=12,cpu_limit_s=90,wall_limit_s=120,
             output=asset(out/"inspection.json") if (out/"inspection.json").exists() else None))
    if result.returncode:raise RuntimeError("Isolated loader failed; no unrestricted fallback")
    write(f"reports/m8a_proxy_retry_logs/checkpoint_structure_{args.run_id}.json",
          dict(**provenance("ActualRestrictedCheckpointStructure"),
               inspection=json.loads((out/"inspection.json").read_text())))
    print(result.stdout)


if __name__=="__main__":
    main()
