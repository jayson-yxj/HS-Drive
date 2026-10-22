"""Read-only progress artifact with wall-clock, source and no completion claims."""
import datetime,json,subprocess
from pathlib import Path
from tools.m5o_common import DATA,write,provenance

def main():
    d=json.loads(subprocess.check_output(['.venv/bin/python','-m','tools.m5o_status'],text=True));stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    write(f'runs/m5o_logs/progress_{stamp}.json',dict(time_utc=stamp,status=d,provenance=provenance()))
    print(d)
if __name__=='__main__':main()
