#!/usr/bin/env python3
"""Deterministic IMU readiness, timing, saturation, and excitation audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def make_stream(rate: float = 200.0, duration: float = 12.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(rate * duration)
    timestamps = np.arange(count, dtype=float) / rate
    gyro = np.column_stack([
        0.03 * np.sin(0.7 * timestamps) + 0.015 * (timestamps > 4.0),
        0.02 * np.cos(0.5 * timestamps),
        0.05 * np.sin(0.4 * timestamps + 0.2),
    ])
    accel = np.column_stack([
        0.25 * np.sin(0.9 * timestamps),
        0.18 * np.cos(0.6 * timestamps),
        9.80665 + 0.30 * np.sin(0.8 * timestamps),
    ])
    # A six-second dynamic excitation window is followed by a quiet window.
    quiet = timestamps > 9.0
    gyro[quiet] *= 0.03
    accel[quiet, :2] *= 0.03
    accel[quiet, 2] = 9.80665
    return timestamps, gyro, accel


def audit(name: str, timestamps: np.ndarray, gyro: np.ndarray, accel: np.ndarray, metadata: dict, limits: dict) -> dict:
    dt = np.diff(timestamps)
    finite = bool(np.isfinite(timestamps).all() and np.isfinite(gyro).all() and np.isfinite(accel).all())
    shape_ok = bool(gyro.ndim == 2 and accel.ndim == 2 and gyro.shape == accel.shape and gyro.shape[1] == 3 and len(timestamps) == len(gyro))
    positive_dt = bool(len(dt) > 0 and np.all(dt > 0.0))
    rate_hz = float(1.0 / np.median(dt)) if positive_dt else 0.0
    jitter = float(np.std(dt) / np.mean(dt)) if positive_dt else float("inf")
    gyro_peak = float(np.max(np.abs(gyro))) if gyro.size else float("inf")
    accel_peak = float(np.max(np.abs(accel))) if accel.size else float("inf")
    gyro_saturated = int(np.sum(np.any(np.abs(gyro) >= limits["gyro_range"] * 0.999, axis=1)))
    accel_saturated = int(np.sum(np.any(np.abs(accel) >= limits["accel_range"] * 0.999, axis=1)))
    dynamic_gyro = float(np.sqrt(np.mean(np.sum((gyro - gyro.mean(axis=0)) ** 2, axis=1)))) if len(gyro) else 0.0
    dynamic_accel = float(np.sqrt(np.mean(np.sum((accel - accel.mean(axis=0)) ** 2, axis=1)))) if len(accel) else 0.0
    calibration_ok = metadata.get("calibration_revision") == limits["calibration_revision"] and metadata.get("units") == "SI"
    checks = {
        "schema": shape_ok,
        "finite": finite,
        "strict_time": positive_dt,
        "rate": positive_dt and abs(rate_hz - limits["rate_hz"]) < limits["rate_tolerance_hz"],
        "jitter": positive_dt and jitter < limits["max_jitter_fraction"],
        "range": gyro_peak < limits["gyro_range"] and accel_peak < limits["accel_range"],
        "saturation": gyro_saturated == 0 and accel_saturated == 0,
        "calibration": calibration_ok,
        "excitation": dynamic_gyro > limits["min_gyro_excitation"] and dynamic_accel > limits["min_accel_excitation"],
    }
    hard_fail = not all(checks[key] for key in ["schema", "finite", "strict_time", "rate", "range", "saturation", "calibration"])
    status = "REJECT" if hard_fail else ("ACCEPT" if checks["excitation"] and checks["jitter"] else "DEGRADE")
    return {
        "case": name, "samples": len(timestamps), "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "rate_hz": rate_hz, "jitter_fraction": jitter, "gyro_peak_rad_s": gyro_peak, "accel_peak_m_s2": accel_peak,
        "gyro_saturated_samples": gyro_saturated, "accel_saturated_samples": accel_saturated,
        "gyro_excitation_rad_s": dynamic_gyro, "accel_excitation_m_s2": dynamic_accel,
        "calibration_revision": metadata.get("calibration_revision"), "checks": checks, "status": status,
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    timestamps, gyro, accel = make_stream()
    metadata = {"calibration_revision": "cam-imu-2026-08-28-r3", "units": "SI", "sensor": "imu0"}
    limits = {
        "rate_hz": 200.0, "rate_tolerance_hz": 0.5, "max_jitter_fraction": 0.02,
        "gyro_range": 4.0, "accel_range": 19.6, "min_gyro_excitation": 0.01,
        "min_accel_excitation": 0.05, "calibration_revision": metadata["calibration_revision"],
    }
    trusted = audit("trusted", timestamps, gyro, accel, metadata, limits)
    cases = [trusted]
    # Each case isolates one readiness contract failure.
    bad_time = timestamps.copy(); bad_time[1000] = bad_time[999]
    cases.append(audit("timestamp_duplicate", bad_time, gyro, accel, metadata, limits))
    bad_gap = timestamps.copy(); bad_gap[1400:] += 0.080
    cases.append(audit("timestamp_gap", bad_gap, gyro, accel, metadata, limits))
    bad_sat = gyro.copy(); bad_sat[700:708, 2] = limits["gyro_range"]
    cases.append(audit("gyro_saturation", timestamps, bad_sat, accel, metadata, limits))
    bad_cal = dict(metadata); bad_cal["calibration_revision"] = "stale-r1"
    cases.append(audit("stale_calibration", timestamps, gyro, accel, bad_cal, limits))
    bad_exc = gyro.copy(); bad_accel = accel.copy(); bad_exc[:] = 0.0; bad_accel[:, :2] = 0.0; bad_accel[:, 2] = 9.80665
    cases.append(audit("weak_excitation", timestamps, bad_exc, bad_accel, metadata, limits))
    bad_schema = accel[:, :2]
    cases.append(audit("schema_width", timestamps, gyro, bad_schema, metadata, limits))
    rows = [{"case": row["case"], "status": row["status"], "rate_hz": row["rate_hz"], "jitter_fraction": row["jitter_fraction"], "gyro_saturated_samples": row["gyro_saturated_samples"], "accel_saturated_samples": row["accel_saturated_samples"], "gyro_excitation_rad_s": row["gyro_excitation_rad_s"], "accel_excitation_m_s2": row["accel_excitation_m_s2"], "failed_checks": ";".join(k for k,v in row["checks"].items() if not v)} for row in cases]
    checks = {
        "trusted_accepts": trusted["status"] == "ACCEPT",
        "duplicate_rejects": cases[1]["status"] == "REJECT" and not cases[1]["checks"]["strict_time"],
        "gap_degrades": cases[2]["status"] == "DEGRADE" and not cases[2]["checks"]["jitter"],
        "saturation_rejects": cases[3]["status"] == "REJECT" and not cases[3]["checks"]["saturation"],
        "stale_calibration_rejects": cases[4]["status"] == "REJECT" and not cases[4]["checks"]["calibration"],
        "weak_excitation_degrades": cases[5]["status"] == "DEGRADE" and not cases[5]["checks"]["excitation"],
        "schema_rejects": cases[6]["status"] == "REJECT" and not cases[6]["checks"]["schema"],
    }
    report = {
        "contract": "accept only finite SI schema, strict time, nominal rate, bounded range, no saturation, matching calibration; degrade on jitter or weak excitation",
        "limits": limits, "cases": cases, "checks": checks,
        "command": "python3 scripts/imu_readiness_smoke.py",
        "artifacts": {"scorecard": "out/imu_readiness_scorecard.csv", "stream": "out/imu_readiness_stream.csv", "model": "out/imu_readiness_model.npz", "plot": "out/imu_readiness_plot.png"},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    with (OUT / "imu_readiness_scorecard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    stream_rows = [{"time_s": t, "gx": g[0], "gy": g[1], "gz": g[2], "ax": a[0], "ay": a[1], "az": a[2]} for t,g,a in zip(timestamps,gyro,accel)]
    with (OUT / "imu_readiness_stream.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stream_rows[0])); writer.writeheader(); writer.writerows(stream_rows)
    np.savez(OUT / "imu_readiness_model.npz", timestamps_s=timestamps, gyro_body_rad_s=gyro, accel_body_m_s2=accel, score_status=np.array([r["status"] for r in cases]))
    canvas = np.full((720,1280,3),246,np.uint8); ink=(58,46,36); blue=(205,132,42); red=(45,45,205); teal=(135,145,25)
    cv2.putText(canvas,"IMU READINESS SCORECARD",(45,52),cv2.FONT_HERSHEY_SIMPLEX,.8,ink,2,cv2.LINE_AA)
    for i,row in enumerate(rows):
        y=120+i*78; color=teal if row["status"]=="ACCEPT" else (blue if row["status"]=="DEGRADE" else red)
        cv2.putText(canvas,row["case"],(70,y),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,row["status"],(470,y),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA); cv2.putText(canvas,row["failed_checks"] or "all checks",(680,y),cv2.FONT_HERSHEY_SIMPLEX,.42,ink,1,cv2.LINE_AA)
    cv2.putText(canvas,"ACCEPT · DEGRADE · REJECT",(790,660),cv2.FONT_HERSHEY_SIMPLEX,.62,teal,2,cv2.LINE_AA); cv2.imwrite(str(OUT/"imu_readiness_plot.png"),canvas)
    (OUT/"imu_readiness_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("IMU readiness lab:",report["status"]); print(json.dumps(report,indent=2))
    return 0 if report["status"]=="PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
