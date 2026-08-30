#!/usr/bin/env python3
"""Deterministic QoS, synchronization, and bounded-queue transport audit."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

def evaluate(case):
    checks = {
        "qos_compatible": case["pub_qos"] == case["sub_qos"],
        "pair_skew_ms": case["max_skew_ms"] <= 4.0,
        "queue_age_ms": case["queue_age_ms"] <= 35.0,
        "interval_complete": case["imu_coverage"] >= 0.98,
        "drop_rate": case["drop_rate"] <= 0.05,
        "latency_ms": case["latency_ms"] <= 30.0,
    }
    critical = (not checks["qos_compatible"]) or (not checks["pair_skew_ms"]) or (not checks["queue_age_ms"])
    status = "ACCEPT" if all(checks.values()) else ("REJECT" if critical else "DEGRADE")
    return checks, status

def main():
    trusted = {"pub_qos":"best_effort/keep_last10", "sub_qos":"best_effort/keep_last10", "max_skew_ms":2.1, "queue_age_ms":11.0, "imu_coverage":1.0, "drop_rate":.01, "latency_ms":14.0}
    cases = [
        ("trusted_30hz_camera_200hz_imu", trusted),
        ("qos_mismatch", {**trusted, "sub_qos":"reliable/keep_all"}),
        ("timestamp_skew", {**trusted, "max_skew_ms":9.7}),
        ("queue_overflow", {**trusted, "queue_age_ms":72.0, "drop_rate":.14, "latency_ms":54.0}),
        ("incomplete_imu_interval", {**trusted, "imu_coverage":.91}),
        ("jitter_within_budget", {**trusted, "max_skew_ms":3.8, "queue_age_ms":28.0}),
    ]
    rows=[]
    for name, case in cases:
        checks, status = evaluate(case)
        rows.append({"case":name, "status":status, "failed_checks":";".join(k for k,v in checks.items() if not v), **case})
    checks = {
        "trusted_accept": rows[0]["status"] == "ACCEPT",
        "qos_mismatch_reject": rows[1]["status"] == "REJECT",
        "skew_reject": rows[2]["status"] == "REJECT",
        "overflow_reject": rows[3]["status"] == "REJECT",
        "incomplete_interval_degrade": rows[4]["status"] == "DEGRADE",
        "jitter_accept": rows[5]["status"] == "ACCEPT",
    }
    report={"status":"PASS" if all(checks.values()) else "FAIL", "cases":rows, "checks":checks, "command":"python3 scripts/qos_sync_backpressure_smoke.py", "artifacts":{"report":"out/qos_sync_backpressure_report.json","scorecard":"out/qos_sync_backpressure_scorecard.csv","model":"out/qos_sync_backpressure_model.npz","plot":"out/qos_sync_backpressure_plot.png"}}
    fields=list(rows[0])
    with (OUT/"qos_sync_backpressure_scorecard.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    np.savez(OUT/"qos_sync_backpressure_model.npz", camera_stamps=np.arange(30)/30, imu_stamps=np.arange(200)/200, queue_age_ms=np.array([r["queue_age_ms"] for r in rows],float), drop_rate=np.array([r["drop_rate"] for r in rows],float))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); teal=(115,140,25); red=(40,45,190); amber=(20,130,190)
    cv2.putText(canvas,"QOS / SYNC / BACKPRESSURE",(42,52),cv2.FONT_HERSHEY_SIMPLEX,.78,ink,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(55,100),(790,635),(215,208,198),1); cv2.putText(canvas,"30 Hz CAMERA  +  200 Hz IMU",(85,145),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,"bounded pairing window",(85,235),cv2.FONT_HERSHEY_SIMPLEX,.56,ink,1,cv2.LINE_AA); cv2.putText(canvas,"skew ≤ 4 ms  ·  age ≤ 35 ms",(85,290),cv2.FONT_HERSHEY_SIMPLEX,.56,ink,1,cv2.LINE_AA); cv2.putText(canvas,"ACCEPT",(85,385),cv2.FONT_HERSHEY_SIMPLEX,.8,teal,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(835,100),(1225,635),(215,208,198),1); cv2.putText(canvas,"CONTROL MATRIX",(870,145),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,"QoS mismatch",(870,235),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,"timestamp skew",(870,285),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,"queue overflow",(870,335),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,"DEGRADE",(870,430),cv2.FONT_HERSHEY_SIMPLEX,.68,amber,2,cv2.LINE_AA); cv2.putText(canvas,"REJECT",(870,500),cv2.FONT_HERSHEY_SIMPLEX,.68,red,2,cv2.LINE_AA)
    cv2.imwrite(str(OUT/"qos_sync_backpressure_plot.png"),canvas); (OUT/"qos_sync_backpressure_report.json").write_text(json.dumps(report,indent=2)+"\n"); print("QoS sync backpressure lab:",report["status"]); return 0 if report["status"]=="PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
