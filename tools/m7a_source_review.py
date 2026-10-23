"""Bounded pinned official Bench2Drive sources; no simulator or dataset download."""
import base64,concurrent.futures,urllib.request
from pathlib import Path
from tools.m7a_common import read,write,asset

PATHS=['README.md','LICENSE','docs/v004_update.md','leaderboard/README.md','leaderboard/CHANGELOG.md','leaderboard/requirements.txt',
 'leaderboard/leaderboard/autoagents/autonomous_agent.py','leaderboard/leaderboard/autoagents/agent_wrapper.py',
 'leaderboard/leaderboard/envs/sensor_interface.py','leaderboard/leaderboard/utils/statistics_manager.py',
 'leaderboard/leaderboard/utils/route_parser.py','leaderboard/leaderboard/leaderboard_evaluator.py',
 'leaderboard/data/bench2drive_0.0.4_val.xml','leaderboard/data/drivetransformer_bench2drive_dev10.xml',
 'scenario_runner/CARLA_VER','scenario_runner/requirements.txt','scenario_runner/README.md',
 'tools/merge_route_json.py','tools/ability_benchmark.py']


def main():
    root=Path('reports/m7a_sources');commit=read(root/'bench2drive_commit.json')['sha']
    def get(path):
        url=f'https://api.github.com/repos/Thinklab-SJTU/Bench2Drive/contents/{path}?ref={commit}'
        rec=dict(path=path,commit=commit,url=f'https://github.com/Thinklab-SJTU/Bench2Drive/blob/{commit}/{path}')
        try:
            import json
            with urllib.request.urlopen(url,timeout=25) as r:d=json.load(r)
            raw=base64.b64decode(d['content'])
            if not raw:raise ValueError('empty_API_content')
            p=root/(path.replace('/','__')+'.txt')
            with p.open('xb') as f:f.write(raw)
            rec.update(status='Retrieved',asset=asset(p),git_blob=d['sha'])
        except Exception as e:rec.update(status='Unavailable',error=repr(e))
        return rec
    results=list(concurrent.futures.ThreadPoolExecutor(4).map(get,PATHS))
    write(root/'manifest.json',results)
    print([(r['path'],r['status']) for r in results])

if __name__=='__main__':main()
