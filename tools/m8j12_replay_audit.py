"""Audit existing replay caches and create at most six private figures. No forward."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from tools.m8j12_runtime import protocol
from datasets.m8j12_contract import CAMERAS,official_camera_contract


def stats(values):
    return {k:float(v) for k,v in zip(('min','p50','p95','max'),np.percentile(values,[0,50,95,100]))}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=protocol();root=Path(c['private_root']);replay=root/'replay'
    capture=json.loads(Path('reports/m8j12_sources/capture_result.json').read_text());observations=Path(capture['observation_root'])
    camera=official_camera_contract(Path(c['runtime_root'])/'official-source/leaderboard/team_code/sparsedrive_b2d_agent.py')
    rows=[json.loads(x) for x in (replay/'frames.jsonl').read_text().splitlines()]
    assert len(rows)==128
    visual=root/'visualizations';visual.mkdir()
    results={};max_extrinsic=0.;max_control_repeat=0.;private_files=[]
    with torch.no_grad():
        for sequence in c['sequence_ids']:
            manifest=json.loads((observations/sequence/'manifest.json').read_text());native=[];hs=[];speeds=[];controls=[];changes=[];selection_sizes=[]
            for index,row in enumerate(manifest):
                path=replay/('0_'+sequence)/(str(index)+'.pt');data=torch.load(path,map_location='cpu');other=torch.load(replay/('1_'+sequence)/(str(index)+'.pt'),map_location='cpu')
                prediction=data['prediction'];p=prediction['traj_final'].numpy();h=data['hs_points'];v=prediction['lon_reg_final'].numpy()
                assert np.array_equal(h[:,:2],np.stack([p[:,1],-p[:,0]],axis=-1))
                selected=data['selection'];idx=selected['traj_mode_idx'][0];candidates=selected['candidates']
                assert torch.equal(candidates[0,idx],prediction['traj_final'])
                selection_sizes.append(candidates.shape[1]);control=data['control']['native_agent_control'];controls.append(list(control.values()))
                max_control_repeat=max(max_control_repeat,max(abs(control[k]-other['control']['native_agent_control'][k]) for k in control))
                # Measured camera transform versus the fixed source extrinsic, all frames.
                ego=np.array(row['ego_world_matrix']);virtual=np.eye(4);virtual[:3,3]=[-.39,0,1.84]
                swap=np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1.]])
                cv=np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1.]])
                for cam in CAMERAS:
                    measured=cv@np.linalg.inv(row['camera_world_matrices'][cam])@ego@virtual@swap
                    max_extrinsic=max(max_extrinsic,float(np.abs(measured-camera['lidar2cam'][cam].T).max()))
                native.append(p);hs.append(h);speeds.extend(v.tolist())
                if index in (0,16,31):
                    images=dict(np.load(observations/sequence/row['images_file'],allow_pickle=False))
                    fig=plt.figure(figsize=(17,10));grid=fig.add_gridspec(3,3)
                    for i,cam in enumerate(CAMERAS):
                        ax=fig.add_subplot(grid[i//3,i%3]);ax.imshow(images[cam][...,::-1]);ax.set_title(cam);ax.axis('off')
                        # Visualization only: embed 2D trajectory on current ego z=0.
                        # This is NOT a measured road/ground surface or collision test.
                        points=np.vstack([h[:,0],-h[:,1],np.zeros(6),np.ones(6)])
                        cv_ego=np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1.]])
                        camera_points=cv_ego@np.linalg.inv(row['camera_world_matrices'][cam])@np.asarray(row['ego_world_matrix'])@points
                        pixels=camera['K'][cam]@camera_points[:3];valid=camera_points[2]>.1
                        uv=(pixels[:2]/np.maximum(pixels[2],1e-6)).T
                        valid &= (uv[:,0]>=0)&(uv[:,0]<704)&(uv[:,1]>=0)&(uv[:,1]<396)
                        ax.scatter(uv[valid,0],uv[valid,1],s=22,c='orange',marker='o')
                    ax=fig.add_subplot(grid[2,:2]);lat=prediction['lat_reg_final'].numpy()
                    ax.plot(-lat[:,0],lat[:,1],'--',label='native path: 1m spatial spacing')
                    ax.plot(h[:,1],h[:,0],'o-',label='selected trajectory: 0.5s temporal spacing')
                    ax.scatter(0,0,marker='^',color='black');ax.set_xlabel('HS left (m)');ax.set_ylabel('HS forward (m)');ax.set_aspect('equal');ax.set_xlim(-8,8);ax.set_ylim(-1,25);ax.legend(fontsize=8);ax.grid()
                    ax=fig.add_subplot(grid[2,2]);ax.plot(np.arange(1,7)*.5,v,'o-');ax.set_xlabel('future time (s)');ax.set_ylabel('window-average speed (m/s)');ax.grid()
                    fig.suptitle(f'{sequence} measured frame {row["frame_id"]}, t={row["timestamp_s"]:.6f}s\nPlanning suggestion on recorded data — NOT EXECUTED BY VEHICLE\nOrange camera dots assume current ego z=0; not measured ground. RGB + Oracle Ego State/navigation.')
                    fig.tight_layout();pfig=visual/(sequence+'_'+str(index)+'.png');fig.savefig(pfig,dpi=120);plt.close(fig)
                    private_files.append(dict(path=str(pfig),sha256=hashlib.sha256(pfig.read_bytes()).hexdigest()))
            controls=np.array(controls);native=np.array(native)
            relevant=[r for r in rows if r['sequence']==sequence]
            results[sequence]=dict(unique_observations=32,replayed_forwards=64,
                planning_speed_mps=stats(speeds),endpoint_hs_forward_m=stats(native[:,-1,1]),endpoint_hs_left_m=stats(-native[:,-1,0]),
                candidate_counts=sorted(set(selection_sizes)),selected_candidate_exactly_matches=True,
                control_ranges={k:stats(controls[:,i]) for i,k in enumerate(('steer','throttle','brake'))},
                max_adjacent_control_changes={k:float(np.abs(np.diff(controls[:,i])).max()) for i,k in enumerate(('steer','throttle','brake'))},
                saturation_counts=dict(steer=int((np.abs(controls[:,0])>=1).sum()),throttle=int((controls[:,1]>=1).sum()),brake=int((controls[:,2]>=1).sum())),
                negative_speed_count=int((np.array(speeds)<0).sum()),
                timings_ms={k:stats([r[k] for r in relevant]) for k in ('preprocess_ms','model_and_decode_ms','decode_ms','control_ms')},
                serialized_sum_ms=stats([r['preprocess_ms']+r['model_and_decode_ms']+r['control_ms'] for r in relevant]))
    assert max_extrinsic<2e-5
    assert max_control_repeat<=c['atol']
    public=dict(sequences=results,max_measured_camera_extrinsic_abs=max_extrinsic,max_control_repeat_abs=max_control_repeat,
        figure_count=len(private_files),private_visualization_directory=str(visual),all_raw_outputs_retained=True,
        native_selected_source='V13LatLonTrajDecoder.decode_v2 after native collision rescoring using predicted detection/motion; profiler logs actual local argmax and scores, no reselection',
        fixed_stats='All state_dict tensors hashed before/after each 32-frame segment; no differences',
        limitations=['Only two short static-environment sequences','Six figures automatically generated, no human correctness signoff; camera dots assume ego z=0, not certified ground','Native PID saturation is retained; dry-run alone does not establish safe/comfortable control',
        'Processing sums omit torch.save/cache audit overhead; no realtime claim','No perception/map precision evaluation'])
    Path('reports/m8j12_sources/replay_audit.json').write_text(json.dumps(public,indent=2)+'\n')
    (visual/'manifest.json').write_text(json.dumps(private_files,indent=2)+'\n')
    print(json.dumps(public,indent=2))


if __name__=='__main__':main()
