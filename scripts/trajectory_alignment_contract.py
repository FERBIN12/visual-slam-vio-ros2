#!/usr/bin/env python3
"""Deterministic trajectory export and frame-alignment contract lab."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

def assess(c: dict) -> tuple[list[str], str]:
    checks = {
        "timestamps_monotonic": c["timestamps_monotonic"],
        "pose_direction_declared": c["pose_direction"] in {"T_world_cam", "T_cam_world"},
        "frame_declared": bool(c["frame"]),
        "scale_declared": c["scale"] == "metric",
        "format_columns": c["format"] in {"tum", "euroc", "ros_path"},
        "alignment_transform": c["alignment_transform"],
    }
    failed = [k for k,v in checks.items() if not v]
    return failed, "ACCEPT" if not failed else ("DEGRADE" if len(failed) <= 2 else "REJECT")

def main() -> int:
    good = dict(timestamps_monotonic=True, pose_direction="T_world_cam", frame="map",
                scale="metric", format="tum", alignment_transform=True)
    cases = [("metric_tum_ready", good),
             ("timestamp_regression", {**good, "timestamps_monotonic": False}),
             ("direction_unspecified", {**good, "pose_direction": ""}),
             ("frame_missing", {**good, "frame": ""}),
             ("scale_unknown", {**good, "scale": "unknown"}),
             ("alignment_missing", {**good, "alignment_transform": False}),
             ("malformed_export", {**good, "pose_direction": "", "frame": "", "format": "csv", "alignment_transform": False})]
    rows=[]
    for name,cfg in cases:
        failed,status=assess(cfg); rows.append({"case":name,"status":status,"failed_checks":";".join(failed)})
    gates={"ready_accept":rows[0]["status"]=="ACCEPT","timestamp_degrade":rows[1]["status"]=="DEGRADE",
           "direction_degrade":rows[2]["status"]=="DEGRADE","frame_degrade":rows[3]["status"]=="DEGRADE",
           "scale_degrade":rows[4]["status"]=="DEGRADE","alignment_degrade":rows[5]["status"]=="DEGRADE",
           "malformed_reject":rows[6]["status"]=="REJECT"}
    report={"status":"PASS" if all(gates.values()) else "FAIL","cases":rows,"gates":gates,
            "command":"python3 scripts/trajectory_alignment_contract.py"}
    with (OUT/"trajectory_alignment_scorecard.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    t=np.linspace(0,9,10); pose=np.c_[t,0.15*t,0.02*t*t]
    np.savez(OUT/"trajectory_alignment_trace.npz",timestamp=t,position_world=pose,scale=np.array([1.0]),aligned=np.array([1]))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37)
    cv2.putText(canvas,"TRAJECTORY EXPORT + FRAME ALIGNMENT",(40,52),cv2.FONT_HERSHEY_SIMPLEX,.72,ink,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(55,105),(760,640),(215,208,198),1); cv2.rectangle(canvas,(820,105),(1225,640),(215,208,198),1)
    for y,s in [(180,"timestamp → pose direction"),(260,"frame + scale + format"),(340,"alignment transform"),(470,"EXPORT CONTRACT")]: cv2.putText(canvas,s,(90,y),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA)
    cv2.putText(canvas,"FAULT MATRIX",(865,170),cv2.FONT_HERSHEY_SIMPLEX,.62,ink,1,cv2.LINE_AA)
    cv2.putText(canvas,"DEGRADE: missing convention",(865,285),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,130,190),1,cv2.LINE_AA)
    cv2.putText(canvas,"REJECT: malformed export",(865,355),cv2.FONT_HERSHEY_SIMPLEX,.47,(40,45,190),1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/"trajectory_alignment_plot.png"),canvas)
    (OUT/"trajectory_alignment_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("trajectory alignment contract lab:",report["status"])
    return 0 if report["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
