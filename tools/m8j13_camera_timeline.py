"""Read-only per-observation timeline; all counterfactuals are estimates."""
import json
from pathlib import Path
from collections import defaultdict
from tools.m8j13_common import write,quantiles
ROOT=Path('/root/gpufree-data/hs-drive-private/m8j13-per-camera-20261101')
OLD=Path('/root/gpufree-data/hs-drive-private/m8j13-input-pipeline-20261101/live')
def rows(name):return [json.loads(x) for x in (OLD/name).read_text().splitlines()]
def main():
    snapshots=rows('snapshots.jsonl');byframe={x['frame']:x for x in snapshots};cameras=defaultdict(list)
    for x in rows('camera_metadata.jsonl'):cameras[x['frame']].append(x)
    received={x['request']:x for x in rows('completion_received.jsonl')};worker={x['request']:x for x in rows('worker_timing.jsonl')}
    allrows=[];success=[]
    for candidate in rows('candidates.jsonl'):
        frame=candidate['frame'];snap=byframe[frame];origin=snap['input_wall_s'];cams=cameras[frame];request=candidate.get('request')
        record=dict(candidate=candidate,snapshot=snap,cameras=cams,origin=origin,snapshot_available_upper_wall=snap['tick_end_wall_s'],
            camera_overlap_tick_intervals=[sum(s['input_wall_s']<=c['receive_monotonic_s']<=s['tick_end_wall_s'] for s in snapshots) for c in cams],
            timing_limit='Recorded interval includes tick RPC and following snapshot/ego reads; exact tick-return and snapshot-ready instants unavailable')
        if len(cams)==6:
            record['spread_ms']=(max(x['receive_monotonic_s'] for x in cams)-min(x['receive_monotonic_s'] for x in cams))*1000
        if request and (OLD/'diagnostics'/request).exists():
            d=json.loads((OLD/'diagnostics'/request).read_text());w=worker[request];rcv=received[request]
            spread=record['spread_ms'];publish_wait=(rcv['wall_s']-d['published_wall_s'])*1000
            independent=sum(d['preprocessing'][k] for k in ('jpeg_encode_ms','jpeg_decode_float_ms','ResizeCropFlipImage_ms','NormalizeMultiviewImage_ms'))
            record.update(diagnostic=d,worker=w,received=rcv,
                counterfactual=dict(notification_wait_removed_ms=publish_wait,owner_endpoint_without_wait_ms=d['timings_ms']['end_to_end_checked']+d['timings_ms']['completion_publish'],
                    first_to_last_wait_cover_cap_ms=min(spread,independent),
                    single_unit_only_camera_spread_removed_checked_ms=d['timings_ms']['end_to_end_checked']-min(spread,independent),
                    meaning='Per-record algebra only; no realized speed or universal theoretical bound. Single-unit cap excludes extra scheduling/transport changes; two-unit candidate requires measurement.'))
            success.append(record)
        allrows.append(record)
    write(ROOT/'historical_timelines.json',allrows)
    summary=dict(candidates=len(allrows),forwards=len(success),complete_camera_groups=sum(len(x['cameras'])==6 for x in allrows),
        arrival_spread_ms=quantiles([x['spread_ms'] for x in allrows if 'spread_ms' in x]),
        spread_between25_and45_ms=sum(25<=x.get('spread_ms',-1)<=45 for x in allrows),
        callbacks_inside_tick_plus_snapshot_read_interval=sum(sum(x['camera_overlap_tick_intervals']) for x in allrows),callbacks=sum(len(x['cameras']) for x in allrows),
        exact_tick_return_available=False,exact_snapshot_available_available=False,
        subsequent31_counterfactual={k:quantiles([x['counterfactual'][k] for x in success[1:]]) for k in success[0]['counterfactual'] if k!='meaning'},
        no_server_GPU_RPC_causal_claim=True,counterfactual_not_measured=True)
    write(ROOT/'historical_timeline_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
