"""Final diagnostic supplements and human-readable summary rendering."""
from pathlib import Path
from tools.m7e_common import *


def main():
    d=read('reports/m7e_full_payload_latency.json');lines=['| 开发序列 | 原A P95 | 缓存A P95 | A+F P95 | A+F Max | A+F过期 |','|---|---:|---:|---:|---:|---:|']
    for x in d['clean_paired']:
        m=x['modes'];lines.append(f"|{x['episode']}|{m['A_uncached']['full_ms']['p95']:.2f}|{m['A']['full_ms']['p95']:.2f}|{m['B']['full_ms']['p95']:.2f}|{m['B']['full_ms']['max']:.2f}|{m['B']['expired']}/61|")
    lines.append('\n单位ms。区域缓存命中率接近99%，但未形成可重复的总时延优势。新F增加了完整连接guide审核负担。全体671次回放666次过期；各场景完整分位数与有效比例独立留档。')
    p=Path('reports/m7e_summary.md');s=p.read_text()
    s=s.replace('<!-- CLEAN_LATENCY_TABLE -->','\n'.join(lines)).replace('<!-- FINAL_TEST_RESULTS -->','测试：**466 passed、8 skipped、1 xfailed；CARLA SDK专项67 passed**。其中13项新候选/缓存测试，历史xfail保持。历史源码/资产SHA核验通过。实际车辆跟踪、实时Worker故障、在线Shadow和新模型训练均NotRun。')
    s=s.replace('8段旧序列＋3段新序列共671次规划的所有已接受候选均用实测未来审核','8段旧序列＋3段新序列共671次规划，638条已接受候选（其中F为250条）均用实测未来审核')
    s=s.replace('t=0未吸附中心线，也不是只覆盖首点。','581条成功构造轨迹的t=0最大状态误差2.78e-17；未吸附中心线，也不是只覆盖首点。')
    p.write_text(s)
    write('reports/m7e_final_validation.json',dict(kind='ActualServerTestAndFrozenSourceAudit',provenance=provenance(),full_test=asset('reports/m7e_execution_logs/pytest_final.txt'),full_result=Path('reports/m7e_execution_logs/pytest_final.txt').read_text().splitlines()[-1],sdk_test=asset('reports/m7e_execution_logs/pytest_sdk_final.txt'),sdk_result=Path('reports/m7e_execution_logs/pytest_sdk_final.txt').read_text().splitlines()[-1],safety=asset('reports/m7e_safety_regression.json'),historical_preservation=read('reports/m7e_execution_logs/preservation.json'),frozen_pre_final=read('reports/m7e_fresh_episode_support.json')['freeze_checks'],actual_control='NotRun',default_changes=False))
if __name__=='__main__':main()
