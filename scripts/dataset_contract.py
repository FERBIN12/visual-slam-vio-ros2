#!/usr/bin/env python3
"""Deterministic EuRoC/TUM-VI dataset contract and provenance gate."""
import csv, hashlib, json
from pathlib import Path

def digest(rows):
    return hashlib.sha256("\n".join(",".join(map(str,r)) for r in rows).encode()).hexdigest()[:16]

def validate(name, rows, calib, sensor, license_ok=True):
    ts=[r[0] for r in rows]; mono=all(b>a for a,b in zip(ts,ts[1:]))
    return {"dataset":name,"rows":len(rows),"monotonic_timestamps":mono,
            "calibration_ref":calib,"sensor_contract":sensor,"license_provenance":license_ok,
            "digest":digest(rows),"verdict":"ACCEPT" if len(rows)>=8 and mono and calib and sensor and license_ok else "REJECT"}

def main():
    out=Path("out"); out.mkdir(exist_ok=True)
    euroc=[[i*0.01, f"cam0/{i:06d}.png", f"imu/{i:06d}.csv"] for i in range(12)]
    tum=[[i*0.005, f"cam0/{i:06d}.png", f"imu/{i:06d}.csv"] for i in range(12)]
    cases=[validate("EuRoC-MH_01",euroc,"MH_01.yaml","cam0+imu0"),validate("TUM-VI-room1",tum,"room1.yaml","cam0+imu0"),
           validate("bad_nonmonotonic",[[0.0,"a","i"],[0.0,"b","j"]],"room1.yaml","cam0+imu0"),
           validate("bad_missing_calib",euroc,"","cam0+imu0")]
    with (out/"dataset_contract_scorecard.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cases[0].keys()); w.writeheader(); w.writerows(cases)
    (out/"dataset_contract_report.json").write_text(json.dumps({"cases":cases,"rule":"timestamps strictly increase; calibration and sensor references required"},indent=2))
    print("dataset contract lab: PASS" if [c["verdict"] for c in cases]==["ACCEPT","ACCEPT","REJECT","REJECT"] else "dataset contract lab: FAIL")
if __name__=="__main__": main()
