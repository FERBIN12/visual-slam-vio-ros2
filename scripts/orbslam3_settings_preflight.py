#!/usr/bin/env python3
"""Deterministic ORB-SLAM3 settings, mode, and provenance preflight."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

REQUIRED = ("Camera.width", "Camera.height", "Camera.fx", "Camera.fy",
            "Camera.cx", "Camera.cy", "Camera.model", "Camera.fps")

def check(cfg: dict) -> tuple[dict, str]:
    flat = cfg.get("settings", {})
    mode = cfg.get("mode", "")
    checks = {
        "schema_complete": all(k in flat for k in REQUIRED),
        "dimensions_valid": flat.get("Camera.width", 0) > 0 and flat.get("Camera.height", 0) > 0,
        "intrinsics_valid": flat.get("Camera.fx", 0) > 0 and flat.get("Camera.fy", 0) > 0,
        "model_supported": flat.get("Camera.model") in {"pinhole", "KannalaBrandt8"},
        "mode_supported": mode in {"mono", "stereo", "rgbd", "stereo_inertial"},
        "baseline_valid": mode not in {"stereo", "stereo_inertial"} or flat.get("Camera.bf", 0) > 0,
        "imu_contract": mode != "stereo_inertial" or (flat.get("IMU.Frequency", 0) > 0 and flat.get("IMU.NoiseGyro", 0) > 0),
        "provenance_pinned": bool(cfg.get("vocabulary_sha256")) and bool(cfg.get("settings_sha256")),
    }
    status = "ACCEPT" if all(checks.values()) else (
        "DEGRADE" if checks["schema_complete"] and checks["dimensions_valid"]
        and checks["intrinsics_valid"] and checks["model_supported"]
        and checks["mode_supported"] and checks["baseline_valid"]
        and checks["provenance_pinned"] else "REJECT")
    return checks, status

def main() -> int:
    base = {"settings": {"Camera.width": 640, "Camera.height": 480,
        "Camera.fx": 430.0, "Camera.fy": 431.0, "Camera.cx": 320.0,
        "Camera.cy": 240.0, "Camera.model": "pinhole", "Camera.fps": 30.0,
        "Camera.bf": 42.0, "IMU.Frequency": 200.0, "IMU.NoiseGyro": 0.0018},
        "mode": "stereo_inertial", "vocabulary_sha256": "a"*64, "settings_sha256": "b"*64}
    cases = [
        ("stereo_inertial_ready", base),
        ("mono_without_baseline", {**base, "mode": "mono", "settings": {**base["settings"], "Camera.bf": 0.0}}),
        ("stereo_missing_baseline", {**base, "settings": {**base["settings"], "Camera.bf": 0.0}}),
        ("imu_frequency_missing", {**base, "settings": {**base["settings"], "IMU.Frequency": 0.0}}),
        ("unsupported_model", {**base, "settings": {**base["settings"], "Camera.model": "fisheye"}}),
        ("unpinned_vocabulary", {**base, "vocabulary_sha256": ""}),
        ("missing_intrinsic", {**base, "settings": {k:v for k,v in base["settings"].items() if k != "Camera.fx"}}),
    ]
    rows = []
    for name, cfg in cases:
        checks, status = check(cfg)
        rows.append({"case": name, "status": status,
                     "failed_checks": ";".join(k for k,v in checks.items() if not v)})
    gates = {
        "ready_accept": rows[0]["status"] == "ACCEPT",
        "mono_accept": rows[1]["status"] == "ACCEPT",
        "missing_baseline_reject": rows[2]["status"] == "REJECT",
        "imu_missing_degrade": rows[3]["status"] == "DEGRADE",
        "model_reject": rows[4]["status"] == "REJECT",
        "vocabulary_reject": rows[5]["status"] == "REJECT",
        "schema_reject": rows[6]["status"] == "REJECT",
    }
    report = {"status": "PASS" if all(gates.values()) else "FAIL", "cases": rows,
              "gates": gates, "command": "python3 scripts/orbslam3_settings_preflight.py"}
    with (OUT / "orbslam3_settings_scorecard.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(OUT / "orbslam3_settings_model.npz",
             intrinsics=np.array([430.,431.,320.,240.]),
             stereo_baseline=np.array([42.0]), imu=np.array([200., .0018]),
             mode_codes=np.array([0,1,2,3]))
    canvas = np.full((720,1280,3),246,np.uint8)
    import cv2
    for text,xy,color in [("ORB-SLAM3 SETTINGS PREFLIGHT",(40,55),(45,43,37)),
        ("schema · intrinsics · model · mode",(70,170),(45,43,37)),
        ("baseline · IMU · provenance",(70,235),(45,43,37)),
        ("ACCEPT",(830,270),(20,130,90)),("DEGRADE",(830,370),(20,130,190)),
        ("REJECT",(830,470),(40,45,190))]:
        cv2.putText(canvas,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.7 if xy[1]<100 else .58,color,2,cv2.LINE_AA)
    cv2.imwrite(str(OUT/"orbslam3_settings_plot.png"),canvas)
    (OUT/"orbslam3_settings_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ORB-SLAM3 settings lab:", report["status"])
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
