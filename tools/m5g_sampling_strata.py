"""Distance/Actor-size supplements to fixed historical Oracle sampling audit."""
import json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5g_sampling_strata.json')
    if dest.exists():raise FileExistsError(dest)
    records=json.loads(Path('reports/m5g_oracle_sampling_ceiling.json').read_text())['records'];rows={r['sample_id']:r for r in json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels/index.json').read_text())['records']};out=[]
    for stride,phase in sorted({(r['stride'],r['phase'])for r in records}):
        aggregate={}
        for r in records:
            row=rows[r['sample_id']]
            for a in r['actors']:
                # Actualcapture at most oneActor ofeachblueprint/config; association audited.
                pixels=sum(x['pixels']for c in row['actors']for x in c['actors']if x['type_id']==a['type_id']and abs(x['distance_m']-a['distance_m'])<.01);size='small_le32'if pixels<=32 else 'medium_le128'if pixels<=128 else 'large';dist=next((f'{lo}_{hi}'for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,35),(35,80)]if lo<=a['distance_m']<hi),'outside')
                if(r['stride'],r['phase'])!=(stride,phase):continue
                for key in ['size:'+size,'distance:'+dist]:
                    acc=aggregate.setdefault(key,dict(actor_observations=0,pixels=0,gt_cells=0,covered=0,matched_05=0,matched_1m=0));acc['actor_observations']+=1;acc['pixels']+=pixels;acc['gt_cells']+=a['visible_cells'];acc['covered']+=a['covered'];acc['matched_05']+=a['matching']['tolerances']['0.5']['matched_gt'];acc['matched_1m']+=a['matching']['tolerances']['1.0']['matched_gt']
        out.append(dict(stride=stride,phase=phase,strata=aggregate))
    atomic_json(dest,dict(kind='offline_saved_sampling_size_distance_breakdown',records=out,source_sha256=file_hash(__file__),limits='HistoricalM5F positions; rawcounts correlated configurations, zero-supportnotnegative. Pixelsacrosscameras summed onceperActor/record; sameActor mayreappear acrossconfigurations.'))

if __name__=='__main__':main()
