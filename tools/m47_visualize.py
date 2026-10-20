"""Measured conflict geometry, open-loop dynamics and closed-loop hold plots."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator.recorder import atomic_json,file_hash
from tools.m47_hold_replay import replay_hold


def load(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",default="reports/m47_figures")
    p.add_argument("--stopping-only", action="store_true")
    p.add_argument("--stopping",default="reports/m47_stopping_verified.json");a=p.parse_args()
    root=Path(a.output)
    if root.exists():raise FileExistsError(root)
    root.mkdir();audit=load("reports/m47_false_free_audit.json");manifest=[]
    # Every one of 104 records gets a concrete 3D visualization, not only examples.
    for r in ([] if a.stopping_only else audit["records"]):
        fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection="3d")
        lo,hi=np.asarray(r["bounds_xyz_m"]);corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        for i in range(8):
            for j in range(i+1,8):
                if np.count_nonzero(corners[i]!=corners[j])==1:
                    ax.plot(*np.stack([corners[i],corners[j]]).T,color="gray",alpha=.5)
        for ray in r["free_rays"]:
            ax.plot(*np.array([ray["entry"],ray["exit"]]).T,color="green",lw=3)
        pts=np.array([p["endpoint"] for p in r["heldout_surface"]])
        ax.scatter(*pts.T,c="red",s=30,label="Heldout surface")
        ax.set(xlabel="HS X m",ylabel="HS Y m",zlabel="HS Z m",
            xlim=(lo[0]-.05,hi[0]+.05),ylim=(lo[1]-.05,hi[1]+.05),zlim=(lo[2]-.1,hi[2]+.1))
        ax.set_title(r["id"]+"\nGreen: free ray segment; red: surface, same coarse cell")
        ax.legend();fig.tight_layout();path=root/("conflict_"+r["id"].replace(":","_")+".png")
        fig.savefig(path,dpi=100);plt.close(fig);manifest.append(dict(id=r["id"],path=str(path)))
    dynamics=load("reports/m47_vehicle_dynamics.json")
    for prefix in ([] if a.stopping_only else ["brake","coast","throttle","ramp","hold_release"]):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        selected=[r for r in dynamics["results"] if r["case"]["name"].startswith(prefix)]
        if prefix=="brake":selected=[r for r in selected if r["case"]["v"]==4.]
        for result in selected:
            rows=[json.loads(s) for s in Path(result["log"]).read_text().splitlines()]
            t=np.array([r["elapsed"] for r in rows]);v=np.array([r["long_speed"] for r in rows]);acc=np.diff(v)/np.diff(t);jerk=np.diff(acc)/np.diff(t[:-1])
            name=result["case"]["name"]
            axes[0,0].plot(t,v,label=name);axes[0,1].plot(t[:-1],acc);axes[1,0].plot(t[:-2],jerk)
            axes[1,1].plot(t,[r["command"]["throttle"]-r["command"]["brake"] for r in rows])
        for ax,title in zip(axes.flat,["Signed longitudinal speed m/s","Raw deceleration m/s2","Raw jerk m/s3","Throttle minus brake command"]):
            ax.set_title(title);ax.set_xlabel("time s");ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle("Actual CARLA open-loop: "+prefix);fig.tight_layout();fig.savefig(root/(prefix+"_response.png"));plt.close(fig)
    stopping=load(a.stopping)
    for scene in sorted(set(r["scenario"] for r in stopping["results"])):
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for result in [r for r in stopping["results"] if r["scenario"]==scene]:
            rows=[json.loads(s) for s in Path(result["log"]).read_text().splitlines()]
            t=np.array([r["t"] for r in rows]);v=np.array([r["measurement"]["long_speed"] for r in rows]);acc=np.diff(v)/np.diff(t);jerk=np.diff(acc)/np.diff(t[:-1])
            axes[0,0].plot(t,v,label=result["version"]);axes[0,1].plot(t,[r["gap"] if r["gap"] is not None else np.nan for r in rows])
            axes[0,2].plot(t[:-1],acc);axes[1,0].plot(t[:-2],jerk)
            axes[1,1].plot(t,[r["command"]["brake"] for r in rows])
            if result["version"] == "m46":
                source="simulator/motion_oracle.py"
                assert file_hash(source)==stopping["source_hashes_at_start"][source]
                hold, _ = replay_hold(rows)
                label="m46 (offline state replay)"
            else:
                hold=[r["command"].get("mode")=="HOLD" for r in rows]
                label="m47 (logged state)"
            axes[1,2].step(t,hold,where="post",label=label)
        for ax,title in zip(axes.flat,["Signed speed","Bumper clearance","Raw acceleration","Raw jerk","Brake command","Stop Hold active"]):
            ax.set_title(title);ax.set_xlabel("simulation seconds");ax.grid(alpha=.2)
        axes[0,0].legend();axes[1,2].legend(fontsize=7);fig.suptitle("Actual matched CARLA: "+scene);fig.tight_layout();fig.savefig(root/(scene+"_stopping.png"));plt.close(fig)
    atomic_json(root/"manifest.json",dict(conflict_figures=manifest,source_sha256=file_hash(__file__),stopping_sha256=file_hash(a.stopping)))


if __name__=="__main__":main()
