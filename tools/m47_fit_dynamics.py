"""Fit a measured signed-speed/actuation table, exposing low-speed discontinuities."""

import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json, file_hash


def main():
    dest = Path("reports/m47_dynamics_calibration.json")
    if dest.exists():
        raise FileExistsError(dest)
    report = json.loads(Path("reports/m47_vehicle_dynamics.json").read_text())
    rows = []
    coast = []
    brake = []
    for run in report["results"]:
        data = [json.loads(x) for x in Path(run["log"]).read_text().splitlines()]
        for i, r in enumerate(data[1:], 1):
            previous = data[i - 1]
            v = previous["long_speed"]
            dt = r["t"] - previous["t"]
            acc = (r["long_speed"] - v) / dt
            command = previous["command"]
            if r["elapsed"] < 0.3:
                continue
            item = dict(
                v=v,
                a=acc,
                throttle=command["throttle"],
                brake=command["brake"],
                case=run["case"]["name"],
            )
            rows.append(item)
            if run["case"]["kind"] == "coast" and 0.2 < v < 5:
                coast.append(item)
            if run["case"]["kind"] in ["brake", "ramp"] and command["brake"] > 0:
                brake.append(item)
    grid = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    table = []
    for v in grid:
        selected = [r for r in rows if abs(r["v"] - v) < max(0.15, v * 0.2)]
        cells = []
        for throttle, br in [
            (0, 0),
            (0.05, 0),
            (0.1, 0),
            (0.2, 0),
            (0.3, 0),
            (0.5, 0),
            (0, 0.005),
            (0, 0.02),
            (0, 0.05),
            (0, 0.1),
            (0, 0.2),
            (0, 0.4),
            (0, 1.0),
        ]:
            sample = [
                r["a"]
                for r in selected
                if abs(r["throttle"] - throttle) < 0.001
                and abs(r["brake"] - br) < 0.001
            ]
            if sample:
                cells.append(
                    dict(
                        throttle=throttle,
                        brake=br,
                        n=len(sample),
                        accel_median=float(np.median(sample)),
                        accel_p05=float(np.quantile(sample, 0.05)),
                        accel_p95=float(np.quantile(sample, 0.95)),
                    )
                )
        table.append(dict(speed=v, cells=cells))
    # Linear high-speed response; validity v>=2, excludes terminal stiction.
    train = [r for r in rows if 2 <= r["v"] <= 5.5]
    X = np.array([[1, r["v"], r["throttle"], r["brake"]] for r in train])
    y = np.array([r["a"] for r in train])
    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ coeff
    atomic_json(
        dest,
        dict(
            kind="offline_fit_actual_carla",
            source=file_hash("reports/m47_vehicle_dynamics.json"),
            table=table,
            high_speed_linear=dict(
                features=["bias", "speed", "throttle", "brake"],
                coefficients=coeff.tolist(),
                mae=float(np.mean(abs(res))),
                valid_speed_range=[2, 5.5],
                samples=len(y),
            ),
            brake_below_2=dict(
                samples=sum(r["v"] < 2 for r in brake),
                decel_gt5=sum(r["v"] < 2 and r["a"] < -5 for r in brake),
            ),
            policy="No service brake in terminal coast until speed<0.08, unless emergency. Coast stopping distance numerically derived from measured deceleration table. Stop-hold latch only at near-zero measured speed.",
            limitations=[
                "Map/vehicle/physics specific; table is identification, not a universal plant model.",
                "Terminal coast variability and reverse rolling require independent stopping gates.",
            ],
        ),
    )
    print(coeff, "MAE", abs(res).mean())


if __name__ == "__main__":
    main()
