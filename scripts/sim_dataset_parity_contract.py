#!/usr/bin/env python3
"""Compare simulation and dataset contracts on declared measurable fields."""
import json
from pathlib import Path

FIELDS=("camera_rate_hz","imu_rate_hz","resolution","time_unit","frame_convention","noise_model")
def compare(a,b):
    diffs={k:(a.get(k),b.get(k)) for k in FIELDS if a.get(k)!=b.get(k)}
    return diffs
def main():
    sim={"camera_rate_hz":30,"imu_rate_hz":200,"resolution":"752x480","time_unit":"ns","frame_convention":"T_body_world","noise_model":"seeded_gaussian"}
    data={**sim}; rate={**sim,"imu_rate_hz":100}; bad={**sim,"time_unit":"s"}
    cases=[("parity",sim,data),("rate_mismatch",sim,rate),("unit_mismatch",sim,bad)]
    rows=[]
    for n,a,b in cases:
        d=compare(a,b); rows.append({"case":n,"differences":len(d),"verdict":"ACCEPT" if not d else "DEGRADE"})
    Path("out").mkdir(exist_ok=True); Path("out/sim_dataset_parity_report.json").write_text(json.dumps({"cases":rows,"fields":FIELDS},indent=2))
    print("simulation-dataset parity lab: PASS" if [r["verdict"] for r in rows]==["ACCEPT","DEGRADE","DEGRADE"] else "simulation-dataset parity lab: FAIL")
if __name__=="__main__": main()
