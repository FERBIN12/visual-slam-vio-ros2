#!/usr/bin/env python3
"""Deterministic ROS 2 adapter contract audit for ORB-SLAM3 ingress messages."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def evaluate(c: dict) -> tuple[list[str], str]:
    checks = {
        "image_encoding": c["image_encoding"] in {"mono8", "bgr8"},
        "imu_encoding": c["imu_encoding"] == "sensor_msgs/Imu",
        "timestamps_monotonic": c["timestamps_monotonic"],
        "camera_frame_declared": bool(c["camera_frame"]),
        "imu_frame_declared": bool(c["imu_frame"]),
        "qos_sensor_data": c["qos"] == "sensor_data",
        "camera_info_match": c["camera_info_match"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    status = "ACCEPT" if not failed else ("DEGRADE" if len(failed) <= 2 else "REJECT")
    return failed, status


def main() -> int:
    good = dict(image_encoding="mono8", imu_encoding="sensor_msgs/Imu",
                timestamps_monotonic=True, camera_frame="camera_optical_frame",
                imu_frame="imu_link", qos="sensor_data", camera_info_match=True)
    cases = [
        ("stereo_adapter_ready", good),
        ("encoding_conversion", {**good, "image_encoding": "rgba8"}),
        ("timestamp_regression", {**good, "timestamps_monotonic": False}),
        ("missing_camera_frame", {**good, "camera_frame": ""}),
        ("wrong_qos", {**good, "qos": "reliable"}),
        ("missing_frame_and_calibration", {**good, "camera_frame": "", "camera_info_match": False}),
        ("bad_message_contract", {**good, "image_encoding": "rgba8", "imu_encoding": "unknown",
                                   "camera_frame": "", "imu_frame": "", "qos": "reliable",
                                   "camera_info_match": False}),
    ]
    rows = []
    for name, cfg in cases:
        failed, status = evaluate(cfg)
        rows.append({"case": name, "status": status, "failed_checks": ";".join(failed)})
    gates = {
        "ready_accept": rows[0]["status"] == "ACCEPT",
        "encoding_degrade": rows[1]["status"] == "DEGRADE",
        "timestamp_degrade": rows[2]["status"] == "DEGRADE",
        "frame_degrade": rows[3]["status"] == "DEGRADE",
        "qos_degrade": rows[4]["status"] == "DEGRADE",
        "calibration_degrade": rows[5]["status"] == "DEGRADE",
        "bad_contract_reject": rows[6]["status"] == "REJECT",
    }
    report = {"status": "PASS" if all(gates.values()) else "FAIL", "cases": rows,
              "gates": gates, "command": "python3 scripts/ros_adapter_contract.py"}
    with (OUT / "ros_adapter_scorecard.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez(OUT / "ros_adapter_contract_model.npz",
             camera_stamp=np.array([0.00, 0.033, 0.066, 0.099]),
             imu_stamp=np.array([0.00, 0.005, 0.010, 0.015]),
             accepted=np.array([1, 1, 0, 1]))
    canvas = np.full((720, 1280, 3), 246, np.uint8)
    ink = (45, 43, 37)
    cv2.putText(canvas, "PROJECT-OWNED ROS 2 ADAPTER CONTRACT", (40, 52),
                cv2.FONT_HERSHEY_SIMPLEX, .72, ink, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (55, 105), (765, 640), (215, 208, 198), 1)
    cv2.putText(canvas, "camera image + CameraInfo", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, .58, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "IMU samples + frame IDs", (90, 260), cv2.FONT_HERSHEY_SIMPLEX, .58, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "timestamps → QoS → ORB-SLAM3", (90, 340), cv2.FONT_HERSHEY_SIMPLEX, .58, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "CONTRACT GATE", (90, 470), cv2.FONT_HERSHEY_SIMPLEX, .7, (115, 140, 25), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (825, 105), (1225, 640), (215, 208, 198), 1)
    cv2.putText(canvas, "FAULT MATRIX", (870, 170), cv2.FONT_HERSHEY_SIMPLEX, .62, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "encoding · timestamp · frame", (870, 255), cv2.FONT_HERSHEY_SIMPLEX, .47, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "QoS · calibration · message type", (870, 305), cv2.FONT_HERSHEY_SIMPLEX, .47, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "DEGRADE", (870, 440), cv2.FONT_HERSHEY_SIMPLEX, .68, (20, 130, 190), 2, cv2.LINE_AA)
    cv2.putText(canvas, "REJECT", (870, 510), cv2.FONT_HERSHEY_SIMPLEX, .68, (40, 45, 190), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "ros_adapter_contract_plot.png"), canvas)
    (OUT / "ros_adapter_contract_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ROS 2 adapter contract lab:", report["status"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
