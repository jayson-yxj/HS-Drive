"""Credential-redacted, paired network checks for the newly configured proxy."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from urllib.parse import urlsplit

from tools.m8a_common import asset, write

CONFIG = Path("configs/m8a/proxy_retry.json")


def provenance(kind):
    files = [CONFIG, *sorted(Path("tools").glob("m8a_proxy*.py")),
             *sorted(Path("tests").glob("test_m8a_proxy*.py"))]
    return dict(kind=kind,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        sources={str(p): asset(p)["sha256"] for p in files}, training_steps=0,
        visual_control=False)


def redact(text):
    text = re.sub(r"https?://[^\s'\"<>]+", "<url_redacted>", str(text))
    text = re.sub(r"socks5h?://[^\s'\"<>]+", "<proxy_redacted>", text)
    text = re.sub(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", "<ip_redacted>", text)
    text = re.sub(r"(?i)(authorization|proxy-authorization|cookie|set-cookie|token|password)\s*[:=].*", r"\1=<redacted>", text)
    return text


def clean_env():
    env = os.environ.copy()
    for k in list(env):
        if k.lower() in ("http_proxy","https_proxy","all_proxy","no_proxy"):
            del env[k]
    return env


def discovery():
    proxies = {}
    for k,v in os.environ.items():
        if k.lower() in ("http_proxy","https_proxy","all_proxy","no_proxy"):
            if k.lower()=="no_proxy":
                proxies[k]={"set":True,"contents":"redacted"}
            else:
                p=urlsplit(v)
                proxies[k]=dict(scheme=p.scheme,loopback=p.hostname in ("127.0.0.1","localhost","::1"),
                    port=p.port,credentials_present=p.username is not None)
    procs=[]
    for path in Path("/proc").glob("[0-9]*/comm"):
        try:
            name=path.read_text().strip()
            if re.search(r"clash|mihomo|sing.box|xray|v2ray",name,re.I):
                procs.append({"pid":int(path.parent.name),"name":name})
        except OSError:
            pass
    raw=subprocess.run(["ss","-lntpH"],capture_output=True,text=True).stdout
    listeners=[]
    for line in raw.splitlines():
        if not re.search(r"mihomo|clash|sing.box|xray|v2ray",line,re.I):
            continue
        local=line.split()[3]
        listeners.append(dict(port=int(local.rsplit(":",1)[-1]),
            loopback=local.startswith(("127.","[::1]")),process=re.findall(r'\("([^"]+)"',line)))
    interfaces=json.loads(subprocess.check_output(["ip","-j","-d","link","show"],text=True))
    configs=[]
    for proc in procs:
        try:
            args=Path(f"/proc/{proc['pid']}/cmdline").read_bytes().split(b"\0")
            for rawpath in args:
                path=Path(rawpath.decode(errors="replace"))
                if path.is_absolute() and path.suffix in (".yaml",".yml") and path.is_file():
                    # Only selected scalar settings; never persist proxy nodes/secrets.
                    text=path.read_text(errors="replace")
                    entry={"process":proc["name"]}
                    for key in ("mixed-port","port","socks-port","mode","allow-lan"):
                        match=re.search(r"^"+key+r":\s*([^\n]+)",text,re.M)
                        if match:entry[key]=match.group(1).strip()
                    tun=re.search(r"^tun:\s*\n((?:[ \t]+[^\n]*\n)*)",text,re.M)
                    if tun:
                        m=re.search(r"^\s+enable:\s*(true|false)",tun.group(1),re.M)
                        entry["tun_enabled"]=(m.group(1)=="true") if m else None
                    configs.append(entry)
        except OSError:
            pass
    return dict(proxy_environment=proxies,proxy_processes=procs,listeners=listeners,
        proxy_config_selected_fields=configs,
        interface_types=[dict(name=x.get("ifname"),kind=x.get("linkinfo",{}).get("info_kind"),
                             type=x.get("link_type")) for x in interfaces],
        external_egress_probe="NotRun_private_exit_not_collected")


def curl_probe(name,url,proxy=None,inherited=False):
    env=os.environ.copy() if inherited else clean_env()
    cmd=["curl","--disable","--silent","--show-error","--head","--location",
         "--connect-timeout","10","--max-time","30","--max-redirs","5",
         "--proto","=https","--proto-redir","=https","--retry","0",
         "--output","/dev/null","--write-out","%{json}"]
    if proxy:cmd += ["--proxy",proxy]
    cmd.append(url)
    start=time.monotonic()
    r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=35)
    try: raw=json.loads(r.stdout)
    except ValueError:raw={}
    # Intentionally omit remote/local IP, all headers and query-bearing effective URLs.
    fields={k:raw.get(k) for k in ("http_code","http_connect","ssl_verify_result","time_total",
        "time_connect","time_appconnect","num_redirects","size_download","exitcode")}
    fields.update(name=name,proxy=proxy if proxy and urlsplit(proxy).hostname=="127.0.0.1" else None,
        mode="explicit_proxy" if proxy else ("inherited_environment" if inherited else "no_explicit_proxy"),
        returncode=r.returncode,elapsed_s=time.monotonic()-start,error=redact(r.stderr),
        tls_verification_disabled=False,
        connected_to_configured_loopback=(raw.get("remote_ip") in ("127.0.0.1","::1")) if proxy else None,
        https_verified=bool(r.returncode==0 and raw.get("ssl_verify_result")==0 and raw.get("http_code",0)>0))
    return fields


def python_probe(name,url,proxy):
    import requests
    session=requests.Session()
    session.trust_env=False
    session.proxies={"http":proxy,"https":proxy}
    start=time.monotonic()
    try:
        with session.get(url,timeout=(10,30),verify=True,stream=True) as response:
            cert=response.raw.connection.sock.getpeercert()
            host=response.raw.connection.sock.getpeername()[0]
            return dict(name=name,client="Python requests",status=response.status_code,
                elapsed_s=time.monotonic()-start,https_verified=True,verify=True,
                connected_to_configured_loopback=host in ("127.0.0.1","::1"),
                certificate_valid_from=cert.get("notBefore"),certificate_valid_until=cert.get("notAfter"),
                history_status=[x.status_code for x in response.history],
                response_body_not_logged=True,external_egress_not_collected=True)
    except Exception as exc:
        return dict(name=name,client="Python requests",elapsed_s=time.monotonic()-start,
                    error=type(exc).__name__,detail=redact(str(exc)),https_verified=False,
                    verify=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["curl","python"],required=True)
    args=parser.parse_args()
    cfg=json.loads(CONFIG.read_text())
    root=Path("reports/m8a_proxy_retry_logs")
    root.mkdir(exist_ok=True)
    if args.phase=="curl":
        write(root/"discovery.json",dict(**provenance("ActualServerProxyDiscovery"),**discovery()))
        jobs=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for name,url in cfg["sites"].items():
                jobs.append(pool.submit(curl_probe,name+"_inherited",url,None,True))
                jobs.append(pool.submit(curl_probe,name+"_explicit_http",url,cfg["discovered_proxy"]))
            # Mixed-port SOCKS5 test uses remote DNS; it does not pick a different exit.
            jobs.append(pool.submit(curl_probe,"google_drive_socks5h",cfg["sites"]["google_drive"],
                                    cfg["discovered_proxy"].replace("http://","socks5h://")))
            results=[x.result() for x in jobs]
        write(root/"curl_probes.json",dict(**provenance("ActualServerPairedCurlTLSProbes"),
            results=results,limitations="Same local proxy endpoint does not prove identical public exit; rule-based provider routing not inspected."))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(python_probe,name,url,cfg["discovered_proxy"]) for name,url in cfg["sites"].items()]
            results=[x.result() for x in futures]
        write(root/"python_probes.json",dict(**provenance("ActualPythonProxyTLSProbes"),results=results))
    for result in results:print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
