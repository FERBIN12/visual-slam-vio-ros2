#!/usr/bin/env python3
"""Deterministic ORB-SLAM3 runtime contract and failure audit."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

def evaluate(c: dict) -> tuple[dict, str]:
    checks = {
        "vocabulary_pinned": c["vocabulary_pinned"],
        "settings_loaded": c["settings_loaded"],
        "sensor_mode_valid": c["sensor_mode_valid"],
        "timestamps_monotonic": c["timestamps_monotonic"],
        "tracking_state_known": c["tracking_state_known"],
        "pose_frame_declared": c["pose_frame_declared"],
    }
    status = "ACCEPT" if all(checks.values()) else (
        "DEGRADE" if checks["vocabulary_pinned"] and checks["settings_loaded"]
        and checks["pose_frame_declared"] else "REJECT")
    return checks, status

def main() -> int:
    good = dict(vocabulary_pinned=True, settings_loaded=True, sensor_mode_valid=True,
                timestamps_monotonic=True, tracking_state_known=True, pose_frame_declared=True)
    cases = [
        ("stereo_inertial_ready", good),
        ("missing_imu_stream", {**good, "sensor_mode_valid": False}),
        ("timestamp_regression", {**good, "timestamps_monotonic": False}),
        ("unknown_tracking_state", {**good, "tracking_state_known": False}),
        ("unpinned_vocabulary", {**good, "vocabulary_pinned": False}),
        ("missing_pose_frame", {**good, "pose_frame_declared": False}),
    ]
    rows = []
    for name, cfg in cases:
        checks, status = evaluate(cfg)
        rows.append({"case": name, "status": status,
                     "failed_checks": ";".join(k for k, v in checks.items() if not v)})
    gates = {
        "ready_accept": rows[0]["status"] == "ACCEPT",
        "missing_imu_degrade": rows[1]["status"] == "DEGRADE",
        "timestamp_degrade": rows[2]["status"] == "DEGRADE",
        "unknown_state_degrade": rows[3]["status"] == "DEGRADE",
        "unpinned_vocab_reject": rows[4]["status"] == "REJECT",
        "missing_frame_reject": rows[5]["status"] == "REJECT",
    }
    report = {"status": "PASS" if all(gates.values()) else "FAIL", "cases": rows,
              "gates": gates, "command": "python3 scripts/orbslam3_runtime_smoke.py"}
    with (OUT / "orbslam3_runtime_scorecard.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(OUT / "orbslam3_runtime_model.npz",
             state_codes=np.array([0, 1, 2, 3]),
             timestamp=np.array([0.00, 0.05, 0.10, 0.15]),
             pose=np.array([[0., 0., 0.], [0.02, 0., 0.], [0.04, 0., 0.], [0.06, 0., 0.]]))
    canvas = np.full((720, 1280, 3), 246, np.uint8); ink = (45, 43, 37)
    cv2.putText(canvas, "ORB-SLAM3 RUNTIME CONTRACT", (42, 52), cv2.FONT_HERSHEY_SIMPLEX, .8, ink, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (55, 110), (760, 625), (215, 208, 198), 1)
    cv2.putText(canvas, "VOCABULARY · SETTINGS · SENSOR MODE", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, .52, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "timestamps · tracking state · pose frame", (90, 255), cv2.FONT_HERSHEY_SIMPLEX, .52, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "READY → TRACKING → LOST → RECOVER", (90, 400), cv2.FONT_HERSHEY_SIMPLEX, .62, (115, 140, 25), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (820, 110), (1225, 625), (215, 208, 198), 1)
    cv2.putText(canvas, "RUNTIME FAULT MATRIX", (855, 165), cv2.FONT_HERSHEY_SIMPLEX, .58, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "missing IMU · time regression", (855, 250), cv2.FONT_HERSHEY_SIMPLEX, .48, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "unknown state · frame absent", (855, 300), cv2.FONT_HERSHEY_SIMPLEX, .48, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "DEGRADE", (855, 425), cv2.FONT_HERSHEY_SIMPLEX, .68, (20, 130, 190), 2, cv2.LINE_AA)
    cv2.putText(canvas, "REJECT", (855, 495), cv2.FONT_HERSHEY_SIMPLEX, .68, (40, 45, 190), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "orbslam3_runtime_plot.png"), canvas)
    (OUT / "orbslam3_runtime_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ORB-SLAM3 runtime lab:", report["status"])
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
