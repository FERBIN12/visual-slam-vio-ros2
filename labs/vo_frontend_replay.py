#!/usr/bin/env python3
"""Deterministic visual-odometry frontend state machine, trajectory gate, and fault replay."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FRAMES, DT = 240, 0.1
STATES = ("BOOTSTRAP", "TRACKING", "DEGRADED", "LOST", "RECOVERING")


def frame_input(index: int, fault_free: bool = False) -> dict:
    truth_x = index * 0.02
    tracks, residual, offset, calibration, scale, relocalization, fault = 180, 0.58, 0.0, "calib-A", 1.0, 130, "none"
    parallax = min(index * 0.09, 1.8)
    error = 0.004 * np.sin(index * 0.17)
    if not fault_free:
        if 80 <= index < 95:
            tracks, residual, fault, error = 68, 2.8, "blur", 0.05 * np.sin(index)
        elif 120 <= index < 133:
            tracks, residual, relocalization, fault, error = 12, 8.0, 0, "blackout", 0.35 + 0.018 * (index - 120)
        elif 160 <= index < 171:
            offset, residual, fault, error = 0.080, 3.2, "timestamp_offset", 0.24
        elif 200 <= index < 208:
            calibration, residual, fault, error = "calib-B-stale", 4.8, "calibration_revision", 0.48
        elif 225 <= index < 231:
            scale, residual, fault = 1.18, 2.4, "scale_jump"
            error = truth_x * (scale - 1.0)
    return {
        "frame": index,
        "time_s": index * DT,
        "fault": fault,
        "tracks": tracks,
        "median_reprojection_px": residual,
        "parallax_deg": parallax,
        "timestamp_offset_s": offset,
        "calibration_revision": calibration,
        "scale_ratio": scale,
        "relocalization_matches": relocalization,
        "truth_x_m": truth_x,
        "estimate_x_m": truth_x + error,
    }


def run_replay(fault_free: bool = False) -> tuple[list[dict], list[dict]]:
    state = "BOOTSTRAP"
    healthy_count = 0
    bad_count = 0
    trace: list[dict] = []
    transitions: list[dict] = []
    for index in range(FRAMES):
        row = frame_input(index, fault_free)
        calibration_ok = row["calibration_revision"] == "calib-A"
        timing_ok = abs(row["timestamp_offset_s"]) <= 0.015
        scale_ok = abs(row["scale_ratio"] - 1.0) <= 0.03
        healthy = calibration_ok and timing_ok and scale_ok and row["tracks"] >= 120 and row["median_reprojection_px"] <= 1.5
        degraded = row["tracks"] < 90 or row["median_reprojection_px"] > 2.0 or not timing_ok or not scale_ok
        critical = row["tracks"] < 30 or abs(row["timestamp_offset_s"]) > 0.05 or not calibration_ok or abs(row["scale_ratio"] - 1.0) > 0.10
        previous = state
        reason = "hold"
        if state == "BOOTSTRAP":
            healthy_count = healthy_count + 1 if healthy and row["parallax_deg"] >= 1.0 else 0
            if healthy_count >= 3:
                state, reason, healthy_count = "TRACKING", "bootstrap_geometry_ready", 0
        elif state == "TRACKING":
            if not calibration_ok:
                state, reason, bad_count = "LOST", "calibration_revision_mismatch", 0
            elif critical:
                state, reason, bad_count = "DEGRADED", "critical_measurement_fault", 1
            elif degraded:
                bad_count += 1
                if bad_count >= 2:
                    state, reason, bad_count = "DEGRADED", "tracking_quality_below_gate", 0
            else:
                bad_count = 0
        elif state == "DEGRADED":
            if critical:
                bad_count += 1
                if bad_count >= 3:
                    state, reason, bad_count = "LOST", "critical_fault_persisted", 0
            elif healthy:
                healthy_count += 1
                if healthy_count >= 4:
                    state, reason, healthy_count = "TRACKING", "quality_recovered", 0
            else:
                healthy_count = 0
        elif state == "LOST":
            if healthy and row["relocalization_matches"] >= 100:
                healthy_count += 1
                if healthy_count >= 2:
                    state, reason, healthy_count = "RECOVERING", "relocalization_supported", 0
            else:
                healthy_count = 0
        elif state == "RECOVERING":
            if critical:
                state, reason, healthy_count = "LOST", "recovery_fault", 0
            elif healthy:
                healthy_count += 1
                if healthy_count >= 4:
                    state, reason, healthy_count = "TRACKING", "recovery_confirmed", 0
            else:
                healthy_count = 0
        if state != previous:
            transitions.append({"frame": index, "from_state": previous, "to_state": state, "reason": reason, "fault": row["fault"]})
        pose_authorized = state == "TRACKING" and healthy
        trace.append({
            **row,
            "state": state,
            "pose_authorized": int(pose_authorized),
            "calibration_ok": int(calibration_ok),
            "timing_ok": int(timing_ok),
            "scale_ok": int(scale_ok),
            "healthy": int(healthy),
            "critical": int(critical),
            "transition_reason": reason if state != previous else "",
        })
    return trace, transitions


def ate(rows: list[dict], authorized_only: bool) -> float:
    selected = [row for row in rows if (row["pose_authorized"] or not authorized_only)]
    error = np.array([row["estimate_x_m"] - row["truth_x_m"] for row in selected])
    return float(np.sqrt(np.mean(error * error)))


def render_timeline(rows: list[dict]) -> np.ndarray:
    width, height = 1280, 720
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    colors = {"BOOTSTRAP": (65, 145, 205), "TRACKING": (60, 165, 115), "DEGRADED": (45, 155, 215), "LOST": (55, 55, 205), "RECOVERING": (170, 105, 35)}
    cv2.putText(canvas, "VISUAL-ODOMETRY FRONTEND STATE AND AUTHORIZED POSE", (55, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 45, 65), 2, cv2.LINE_AA)
    left, right, top, bottom = 70, 1230, 130, 570
    for row in rows:
        x0 = left + int((right - left) * row["frame"] / FRAMES)
        x1 = left + int((right - left) * (row["frame"] + 1) / FRAMES)
        cv2.rectangle(canvas, (x0, top), (max(x0 + 1, x1), top + 95), colors[row["state"]], -1)
        if row["pose_authorized"]:
            cv2.line(canvas, (x0, bottom), (max(x0 + 1, x1), bottom), (60, 165, 115), 6)
        if row["fault"] != "none":
            cv2.line(canvas, (x0, top + 130), (x0, bottom - 45), (55, 55, 205), 1)
    cv2.putText(canvas, "state", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 65, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "authorized pose", (left, bottom + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 65, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "blur   blackout   timestamp   calibration   scale", (680, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 65, 80), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows, transitions = run_replay(False)
    clean_rows, clean_transitions = run_replay(True)
    authorized_ate = ate(rows, True)
    optimistic_ate = ate(rows, False)
    fault_names = ("blur", "blackout", "timestamp_offset", "calibration_revision", "scale_jump")
    fault_detection = {}
    for name in fault_names:
        indices = [row["frame"] for row in rows if row["fault"] == name]
        affected = [row for row in rows if row["frame"] in indices]
        fault_detection[name] = {
            "frames": len(indices),
            "pose_blocked_frames": sum(not row["pose_authorized"] for row in affected),
            "degraded_or_lost_frames": sum(row["state"] in {"DEGRADED", "LOST", "RECOVERING"} for row in affected),
        }
    recovery_frames = [event["frame"] for event in transitions if event["to_state"] == "TRACKING" and event["from_state"] in {"DEGRADED", "RECOVERING"}]
    manifest = {
        "frames": FRAMES,
        "dt_s": DT,
        "duration_s": (FRAMES - 1) * DT,
        "states": list(STATES),
        "transitions": transitions,
        "state_counts": {state: sum(row["state"] == state for row in rows) for state in STATES},
        "pose_authorized_frames": sum(row["pose_authorized"] for row in rows),
        "authorized_pose_ate_m": authorized_ate,
        "optimistic_all_frame_ate_m": optimistic_ate,
        "fault_detection": fault_detection,
        "recovery_to_tracking_frames": recovery_frames,
        "fault_free_control": {
            "transitions": clean_transitions,
            "pose_authorized_frames": sum(row["pose_authorized"] for row in clean_rows),
            "ate_m": ate(clean_rows, True),
        },
        "command": "python3 scripts/vo_frontend_replay.py",
        "artifacts": {
            "state_trace": "out/vo_state_trace.csv",
            "trajectory": "out/vo_trajectory.csv",
            "local_map": "out/vo_local_map.npz",
            "failure_report": "out/vo_failure_report.json",
            "timeline": "out/vo_state_timeline.png",
        },
    }
    checks = {
        "bootstrap_completed": any(event["from_state"] == "BOOTSTRAP" and event["to_state"] == "TRACKING" and event["frame"] <= 20 for event in transitions),
        "all_faults_block_pose": all(value["pose_blocked_frames"] == value["frames"] for value in fault_detection.values()),
        "all_faults_reach_diagnostic_state": all(value["degraded_or_lost_frames"] >= max(1, value["frames"] - 1) for value in fault_detection.values()),
        "trusted_trajectory_accurate": authorized_ate < 0.01,
        "optimistic_control_worse": optimistic_ate > authorized_ate * 15.0,
        "calibration_mismatch_immediate_lost": any(event["frame"] == 200 and event["to_state"] == "LOST" and event["reason"] == "calibration_revision_mismatch" for event in transitions),
        "recovery_exercised": len(recovery_frames) >= 4,
        "fault_free_control_clean": not any(event["to_state"] in {"DEGRADED", "LOST", "RECOVERING"} for event in clean_transitions),
        "state_enum_valid": all(row["state"] in STATES for row in rows),
    }
    manifest["checks"] = checks
    manifest["status"] = "PASS" if all(checks.values()) else "FAIL"

    fields = list(rows[0])
    with (OUT / "vo_state_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (OUT / "vo_trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["frame", "time_s", "state", "pose_authorized", "truth_x_m", "estimate_x_m", "absolute_error_m"])
        for row in rows:
            writer.writerow([row["frame"], row["time_s"], row["state"], row["pose_authorized"], row["truth_x_m"], row["estimate_x_m"], abs(row["estimate_x_m"] - row["truth_x_m"])])
    local_map_path = OUT / "local_map_model.npz"
    if not local_map_path.exists():
        raise RuntimeError("run landmark_lifecycle_smoke.py before frontend replay")
    local_map = np.load(local_map_path, allow_pickle=False)
    np.savez(OUT / "vo_local_map.npz", **{name: local_map[name] for name in local_map.files}, source_sha256=hashlib.sha256(local_map_path.read_bytes()).hexdigest())
    failure_report = {"fault_detection": fault_detection, "transitions": transitions, "optimistic_all_frame_ate_m": optimistic_ate, "authorized_pose_ate_m": authorized_ate}
    (OUT / "vo_failure_report.json").write_text(json.dumps(failure_report, indent=2) + "\n")
    cv2.imwrite(str(OUT / "vo_state_timeline.png"), render_timeline(rows))
    (OUT / "vo_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("visual-odometry frontend replay:", manifest["status"])
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
