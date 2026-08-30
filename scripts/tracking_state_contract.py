#!/usr/bin/env python3
"""Deterministic tracking-state, reset, and relocalization contract audit."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

VALID = {"NO_IMAGES_YET", "NOT_INITIALIZED", "OK", "RECENTLY_LOST", "LOST", "RELOCALIZING"}

def audit(trace: list[dict], reset_scope: str = "active_map") -> tuple[list[str], str]:
    failures: list[str] = []
    previous_t = -1.0
    pose_valid = False
    reset_seen = False
    relocalized = False
    for i, e in enumerate(trace):
        state, t = e["state"], float(e["t"])
        if state not in VALID: failures.append(f"row{i}:unknown_state")
        if t <= previous_t: failures.append(f"row{i}:timestamp_regression")
        previous_t = t
        if state == "OK": pose_valid = True
        if state in {"NO_IMAGES_YET", "NOT_INITIALIZED", "RECENTLY_LOST", "LOST"}: pose_valid = False
        if e.get("publish_pose", False) != pose_valid: failures.append(f"row{i}:pose_validity_mismatch")
        if e.get("reset"):
            reset_seen = True
            if e.get("reset_scope") != reset_scope: failures.append(f"row{i}:reset_scope")
        if state == "RELOCALIZING":
            if e.get("publish_pose", False): failures.append(f"row{i}:relocalizing_published")
        if e.get("relocalized"):
            relocalized = True
            if state != "OK": failures.append(f"row{i}:relocalized_without_ok")
    if reset_seen and not relocalized: failures.append("reset_without_recovery")
    return failures, ("ACCEPT" if not failures else ("DEGRADE" if len(failures) <= 2 else "REJECT"))

def main() -> int:
    good = [
      {"t":0.00,"state":"NO_IMAGES_YET","publish_pose":False},
      {"t":0.10,"state":"NOT_INITIALIZED","publish_pose":False},
      {"t":0.20,"state":"OK","publish_pose":True},
      {"t":0.30,"state":"RECENTLY_LOST","publish_pose":False},
      {"t":0.40,"state":"RELOCALIZING","publish_pose":False},
      {"t":0.50,"state":"OK","publish_pose":True,"relocalized":True},
    ]
    cases = [
      ("nominal_relocalization", good),
      ("lost_pose_held", [*good[:3], {"t":0.30,"state":"LOST","publish_pose":True}]),
      ("relocalizing_published", [*good[:4], {"t":0.40,"state":"RELOCALIZING","publish_pose":True}, good[-1]]),
      ("reset_wrong_scope", [*good[:3], {"t":0.35,"state":"NOT_INITIALIZED","publish_pose":False,"reset":True,"reset_scope":"all_maps"}, good[-1]]),
      ("timestamp_regression", [*good[:4], {"t":0.15,"state":"RELOCALIZING","publish_pose":False}, {"t":0.10,"state":"TRACKING_MAYBE","publish_pose":True}]),
      ("unknown_state", [*good[:2], {"t":0.20,"state":"TRACKING_MAYBE","publish_pose":True}]),
    ]
    rows=[]
    for name, trace in cases:
        failed,status=audit(trace)
        rows.append({"case":name,"status":status,"failed_checks":";".join(failed)})
    gates={"nominal_accept":rows[0]["status"]=="ACCEPT","held_pose_degrade":rows[1]["status"]=="DEGRADE","relocalizing_degrade":rows[2]["status"]=="DEGRADE","scope_degrade":rows[3]["status"]=="DEGRADE","time_reject":rows[4]["status"]=="REJECT","unknown_degrade":rows[5]["status"]=="DEGRADE"}
    report={"status":"PASS" if all(gates.values()) else "FAIL","cases":rows,"gates":gates,"command":"python3 scripts/tracking_state_contract.py"}
    with (OUT/"tracking_state_scorecard.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(OUT/"tracking_state_trace.npz", t=np.array([e["t"] for e in good]), state=np.array([e["state"] for e in good]), pose_valid=np.array([e["publish_pose"] for e in good]))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37)
    cv2.putText(canvas,"TRACKING STATE / RESET / RELOCALIZATION",(40,52),cv2.FONT_HERSHEY_SIMPLEX,.72,ink,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(55,105),(770,640),(215,208,198),1); cv2.rectangle(canvas,(825,105),(1225,640),(215,208,198),1)
    for i,e in enumerate(good): cv2.putText(canvas,f"{e['t']:.2f}  {e['state']:<15} pose={e['publish_pose']}",(90,175+i*62),cv2.FONT_HERSHEY_SIMPLEX,.53,ink,1,cv2.LINE_AA)
    cv2.putText(canvas,"VALIDITY GATE",(865,190),cv2.FONT_HERSHEY_SIMPLEX,.65,(115,140,25),2,cv2.LINE_AA)
    cv2.putText(canvas,"OK → publish pose",(865,270),cv2.FONT_HERSHEY_SIMPLEX,.50,ink,1,cv2.LINE_AA)
    cv2.putText(canvas,"LOST → invalidate",(865,330),cv2.FONT_HERSHEY_SIMPLEX,.50,ink,1,cv2.LINE_AA)
    cv2.putText(canvas,"RELOCALIZING → hold",(865,390),cv2.FONT_HERSHEY_SIMPLEX,.50,ink,1,cv2.LINE_AA)
    cv2.putText(canvas,"RESET SCOPE → active map",(865,450),cv2.FONT_HERSHEY_SIMPLEX,.50,ink,1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/"tracking_state_plot.png"),canvas); (OUT/"tracking_state_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("tracking state contract lab:",report["status"]); return 0 if report["status"]=="PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
