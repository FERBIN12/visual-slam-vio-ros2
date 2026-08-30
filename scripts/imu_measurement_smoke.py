#!/usr/bin/env python3
"""Deterministic IMU measurement-model fixtures with sign, unit, and frame faults."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
GRAVITY_WORLD = np.array([0.0, 0.0, -9.80665])


def rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def specific_force(rotation_world_body: np.ndarray, acceleration_world: np.ndarray) -> np.ndarray:
    return rotation_world_body.T @ (acceleration_world - GRAVITY_WORLD)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(71)
    fixtures = [
        ("static_upright", np.eye(3), np.zeros(3), np.zeros(3)),
        ("static_roll_90", rot_x(90), np.zeros(3), np.zeros(3)),
        ("static_pitch_90", rot_y(90), np.zeros(3), np.zeros(3)),
        ("static_yaw_90", rot_z(90), np.zeros(3), np.zeros(3)),
        ("tilted_static", rot_z(25) @ rot_y(-20) @ rot_x(30), np.zeros(3), np.zeros(3)),
        ("dynamic_world_accel", rot_z(30) @ rot_y(-10), np.array([1.2, -0.4, 0.3]), np.zeros(3)),
        ("constant_yaw_rate", rot_z(15), np.zeros(3), np.array([0.0, 0.0, np.radians(30.0)])),
    ]
    rows = []
    rotations = []
    for name, rotation, acceleration, omega_body in fixtures:
        predicted_accel = specific_force(rotation, acceleration)
        measured_accel = predicted_accel + rng.normal(0.0, 0.0015, 3)
        measured_gyro = omega_body + rng.normal(0.0, 0.00015, 3)
        rotations.append(rotation)
        rows.append({
            "fixture": name,
            "acceleration_world_x": acceleration[0], "acceleration_world_y": acceleration[1], "acceleration_world_z": acceleration[2],
            "specific_force_x": predicted_accel[0], "specific_force_y": predicted_accel[1], "specific_force_z": predicted_accel[2],
            "measured_accel_x": measured_accel[0], "measured_accel_y": measured_accel[1], "measured_accel_z": measured_accel[2],
            "omega_body_x_rad_s": omega_body[0], "omega_body_y_rad_s": omega_body[1], "omega_body_z_rad_s": omega_body[2],
            "measured_gyro_x_rad_s": measured_gyro[0], "measured_gyro_y_rad_s": measured_gyro[1], "measured_gyro_z_rad_s": measured_gyro[2],
            "accel_residual_norm": float(np.linalg.norm(measured_accel - predicted_accel)),
            "gyro_residual_norm": float(np.linalg.norm(measured_gyro - omega_body)),
        })
    tilted_rotation = fixtures[4][1]
    tilted_expected = specific_force(tilted_rotation, np.zeros(3))
    sign_flip = tilted_rotation.T @ GRAVITY_WORLD
    wrong_frame = tilted_rotation @ (-GRAVITY_WORLD)
    axis_swap = tilted_expected[[1, 0, 2]]
    correct_yaw_after_2s = np.radians(30.0) * 2.0
    wrong_yaw_after_2s = 30.0 * 2.0
    report = {
        "fixtures": len(rows),
        "gravity_world_m_s2": GRAVITY_WORLD.tolist(),
        "convention": "R_world_body maps body vectors to world; f_body = R_world_body.T @ (a_world - g_world); omega is body rad/s",
        "static_upright_specific_force_m_s2": float(rows[0]["specific_force_z"]),
        "static_specific_force_norms_m_s2": [float(np.linalg.norm([row["specific_force_x"], row["specific_force_y"], row["specific_force_z"]])) for row in rows[:5]],
        "max_accel_residual_m_s2": max(row["accel_residual_norm"] for row in rows),
        "max_gyro_residual_rad_s": max(row["gyro_residual_norm"] for row in rows),
        "known_yaw_rate_rad_s": float(rows[-1]["omega_body_z_rad_s"]),
        "faults": {
            "gravity_sign_error_m_s2": float(np.linalg.norm(sign_flip - tilted_expected)),
            "wrong_rotation_direction_error_m_s2": float(np.linalg.norm(wrong_frame - tilted_expected)),
            "axis_swap_error_m_s2": float(np.linalg.norm(axis_swap - tilted_expected)),
            "degrees_as_radians_rate_ratio": wrong_yaw_after_2s / correct_yaw_after_2s,
        },
        "command": "python3 scripts/imu_measurement_smoke.py",
        "artifacts": {"measurements": "out/imu_measurements.csv", "model": "out/imu_measurement_model.npz", "plot": "out/imu_fixture_plot.png"},
    }
    checks = {
        "upright_reads_positive_g": abs(report["static_upright_specific_force_m_s2"] - 9.80665) < 1e-9,
        "static_norm_is_g": max(abs(value - 9.80665) for value in report["static_specific_force_norms_m_s2"]) < 1e-9,
        "measurement_residuals_bounded": report["max_accel_residual_m_s2"] < 0.006 and report["max_gyro_residual_rad_s"] < 0.0006,
        "known_rate_in_radians": abs(report["known_yaw_rate_rad_s"] - np.pi / 6.0) < 1e-12,
        "gravity_sign_fault_visible": report["faults"]["gravity_sign_error_m_s2"] > 19.0,
        "frame_direction_fault_visible": report["faults"]["wrong_rotation_direction_error_m_s2"] > 5.0,
        "axis_fault_visible": report["faults"]["axis_swap_error_m_s2"] > 1.0,
        "unit_fault_visible": report["faults"]["degrees_as_radians_rate_ratio"] > 57.0,
        "rotations_valid": all(np.linalg.norm(rotation.T @ rotation - np.eye(3)) < 1e-12 and abs(np.linalg.det(rotation) - 1.0) < 1e-12 for rotation in rotations),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report["checks"] = checks
    report["status"] = "PASS" if all(checks.values()) else "FAIL"
    with (OUT / "imu_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez(OUT / "imu_measurement_model.npz", gravity_world_m_s2=GRAVITY_WORLD, rotations_world_body=np.array(rotations), predicted_specific_force_m_s2=np.array([[row["specific_force_x"], row["specific_force_y"], row["specific_force_z"]] for row in rows]), omega_body_rad_s=np.array([[row["omega_body_x_rad_s"], row["omega_body_y_rad_s"], row["omega_body_z_rad_s"]] for row in rows]))
    canvas = np.full((720, 1280, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, "IMU MEASUREMENT FIXTURES", (55, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 45, 65), 2, cv2.LINE_AA)
    for index, row in enumerate(rows):
        x = 90 + index * 165
        vector = np.array([row["specific_force_x"], row["specific_force_y"], row["specific_force_z"]])
        cv2.line(canvas, (x, 560), (x + int(vector[0] * 12), 560 - int(vector[2] * 28)), (55, 150, 110), 5, cv2.LINE_AA)
        cv2.putText(canvas, row["fixture"].replace("_", " ")[:16], (x - 45, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55, 65, 80), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "imu_fixture_plot.png"), canvas)
    (OUT / "imu_measurement_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("IMU measurement lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
