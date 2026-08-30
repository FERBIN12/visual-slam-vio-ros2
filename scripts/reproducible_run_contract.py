#!/usr/bin/env python3
"""Deterministic end-to-end ORB-SLAM3 run manifest and release gate."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"out"; OUT.mkdir(exist_ok=True)
def sha(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()
def evaluate(c: dict):
    checks={"clean_execution":c["clean_execution"],"inputs_pinned":c["inputs_pinned"],"artifacts_hashed":c["artifacts_hashed"],"states_covered":c["states_covered"],"metrics_recorded":c["metrics_recorded"],"negative_control":c["negative_control"]}
    failed=[k for k,v in checks.items() if not v]; return failed,"ACCEPT" if not failed else ("DEGRADE" if len(failed)<=2 else "REJECT")
def main()->int:
    good=dict(clean_execution=True,inputs_pinned=True,artifacts_hashed=True,states_covered=True,metrics_recorded=True,negative_control=True)
    cases=[("release_ready",good),("dirty_workspace",{**good,"clean_execution":False}),("unversioned_dataset",{**good,"inputs_pinned":False}),("missing_hashes",{**good,"artifacts_hashed":False}),("no_state_coverage",{**good,"states_covered":False}),("metric_missing",{**good,"metrics_recorded":False}),("malformed_run",{**good,"clean_execution":False,"inputs_pinned":False,"artifacts_hashed":False,"negative_control":False})]
    rows=[]
    for name,c in cases:
        failed,status=evaluate(c); rows.append({"case":name,"status":status,"failed_checks":";".join(failed)})
    gates={"ready_accept":rows[0]["status"]=="ACCEPT","clean_degrade":rows[1]["status"]=="DEGRADE","inputs_degrade":rows[2]["status"]=="DEGRADE","hash_degrade":rows[3]["status"]=="DEGRADE","states_degrade":rows[4]["status"]=="DEGRADE","metrics_degrade":rows[5]["status"]=="DEGRADE","malformed_reject":rows[6]["status"]=="REJECT"}
    report={"status":"PASS" if all(gates.values()) else "FAIL","cases":rows,"gates":gates,"manifest_hash":sha(json.dumps(good,sort_keys=True)),"command":"python3 scripts/reproducible_run_contract.py"}
    with (OUT/"reproducible_run_scorecard.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/"reproducible_run_manifest.json").write_text(json.dumps({"environment":"container-pinned","inputs":["settings.yaml","vocabulary.bin","dataset.bag"],"outputs":["trajectory.tum","metrics.json","logs.txt"],"hash":report["manifest_hash"]},indent=2)+"\n")
    np.savez(OUT/"reproducible_run_trace.npz",frame=np.arange(12),state=np.array(["INIT","OK","OK","LOST","RELOCALIZING","OK"]*2),metric=np.linspace(.08,.03,12))
    canvas=np.full((720,1280,3),246,np.uint8);ink=(45,43,37);cv2.putText(canvas,"REPRODUCIBLE ORB-SLAM3 RUN CONTRACT",(40,52),cv2.FONT_HERSHEY_SIMPLEX,.68,ink,2,cv2.LINE_AA);cv2.rectangle(canvas,(55,105),(760,640),(215,208,198),1);cv2.rectangle(canvas,(820,105),(1225,640),(215,208,198),1)
    for y,s in [(180,"container + pinned build"),(260,"inputs + command + hashes"),(340,"states + metrics + faults"),(470,"RELEASE GATE")]:cv2.putText(canvas,s,(90,y),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA)
    cv2.putText(canvas,"FAULT MATRIX",(865,170),cv2.FONT_HERSHEY_SIMPLEX,.62,ink,1,cv2.LINE_AA);cv2.putText(canvas,"DEGRADE: missing evidence",(865,285),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,130,190),1,cv2.LINE_AA);cv2.putText(canvas,"REJECT: malformed run",(865,355),cv2.FONT_HERSHEY_SIMPLEX,.47,(40,45,190),1,cv2.LINE_AA);cv2.imwrite(str(OUT/"reproducible_run_plot.png"),canvas)
    (OUT/"reproducible_run_report.json").write_text(json.dumps(report,indent=2)+"\n");print("reproducible run contract lab:",report["status"]);return 0 if report["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
