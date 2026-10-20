"""Road/obstacle edge diagnostics on independently queried collision surfaces."""
import json
from pathlib import Path
import numpy as np
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json,file_hash


def main():
    output=Path("reports/m47_local_quality.json")
    if output.exists():raise FileExistsError(output)
    quality=json.loads(Path("reports/m47_occupancy_quality.json").read_text())
    roots=[Path("/root/gpufree-data/hs-drive-data/raw")/name for name in
           ["m47_development","m47_heldout_town02","m47_heldout_town02_tesla"]]
    scene_roots={m["scene"]:(root/m["scene"],m) for root in roots
        for m in json.loads((root/"manifest.json").read_text())["scenes"]}
    g=VoxelGrid();records=[]
    for r in quality["records"]:
        root,man=scene_roots[r["scene"]];fid=r["frame_id"]
        frame=json.loads((root/"frames"/f"{fid:08d}.json").read_text())
        geometry=json.loads((root/f"m47_geometry_{fid:08d}.json").read_text())
        with np.load(r["repaired_path"]) as data:labels={k:data[k] for k in data.files}
        state=labels["occupancy_state"].copy()
        state[labels["m47_conflict_quarantine"]]=1 # recover BEFORE quarantine
        semantic={}
        for camera,ref in frame["cameras"].items():
            with np.load(root/ref["instance"]) as z:semantic[camera]=z["semantic"]
        groups={"road_semantic_edge":[],"obstacle_semantic_edge":[],"interior_pixel":[]}
        for q in geometry["ray_queries"]:
            u,v=q["uv"];s=semantic[q["camera"]]
            center=int(s[v,u]);patch=s[max(0,v-3):v+4,max(0,u-3):u+4]
            group=int(man["semantic_groups"].get(str(center),0))
            edge=bool((patch!=center).any())
            key="road_semantic_edge" if edge and group==1 else "obstacle_semantic_edge" if edge and group in [2,3] else "interior_pixel"
            for hit in q["hits"]:
                point=np.asarray(hit["point_hs"]).reshape(3);rank,inside=g.locate(point)
                if not inside or labels["ego_mask"].ravel()[rank]:continue
                cls=int(man["semantic_groups"].get(str(hit["semantic"]),0))
                groups[key].append((int(state.ravel()[rank]),cls,int(labels["surface_semantic"].ravel()[rank]),
                    bool(labels["surface_semantic_valid"].ravel()[rank])))
        summary={}
        for key,values in groups.items():
            a=np.asarray(values).reshape(-1,4)
            summary[key]=dict(points=len(a),free_conflicts=int((a[:,0]==1).sum()),
                unknown=int((a[:,0]==0).sum()),surface_recall=float((a[:,0]==2).mean()) if len(a) else None,
                ground_semantic_recall=float(((a[:,0]==2)&(a[:,2]==1)&(a[:,3]==1))[a[:,1]==1].mean()) if (a[:,1]==1).any() else None,
                obstacle_semantic_recall=float(((a[:,0]==2)&np.isin(a[:,2],[2,3])&(a[:,3]==1))[np.isin(a[:,1],[2,3])].mean()) if np.isin(a[:,1],[2,3]).any() else None)
        records.append(dict(scene=r["scene"],frame_id=fid,local=summary))
    atomic_json(output,dict(kind="offline_collision_query_and_native_semantic_edge_diagnostic",records=records,
        definition="7x7 native semantic patch differs from center: image-space boundary candidate, not metric distance to road polygon.",
        limitations=["Boundary labels come from renderer; geometry points from collision query; disagreement not resolved.",
            "Query endpoints may differ from render surface, so semantic recall is conditional supplementary evidence.",
            "No exact road-edge distance claim; no pixel edge samples means metric is null, not perfect."],
        source_sha256=file_hash(__file__)))


if __name__=="__main__":main()
