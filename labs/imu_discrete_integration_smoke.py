#!/usr/bin/env python3
"""Deterministic SO(3) midpoint IMU integration with timing and Euler controls."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
GRAVITY_WORLD = np.array([0.0, 0.0, -9.80665])


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(phi, dtype=float).reshape(3, 1))
    return rotation


def rotation_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    rotation_vector, _ = cv2.Rodrigues(estimate @ truth.T)
    return float(np.degrees(np.linalg.norm(rotation_vector)))


def integrate(timestamps: np.ndarray, gyro: np.ndarray, accel: np.ndarray, gyro_bias: np.ndarray | None = None, accel_bias: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    gyro_bias = np.zeros(3) if gyro_bias is None else gyro_bias
    accel_bias = np.zeros(3) if accel_bias is None else accel_bias
    rotation = np.eye(3)
    velocity = np.zeros(3)
    position = np.zeros(3)
    trace = [{"time_s": float(timestamps[0]), "px": 0.0, "py": 0.0, "pz": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "orthogonality_error": 0.0}]
    for index, dt in enumerate(np.diff(timestamps)):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("timestamps must be finite and strictly increasing")
        omega_mid = 0.5 * (gyro[index] + gyro[index + 1]) - gyro_bias
        force_mid = 0.5 * (accel[index] + accel[index + 1]) - accel_bias
        rotation_mid = rotation @ exp_so3(omega_mid * dt * 0.5)
        acceleration_world = rotation_mid @ force_mid + GRAVITY_WORLD
        position = position + velocity * dt + 0.5 * acceleration_world * dt * dt
        velocity = velocity + acceleration_world * dt
        rotation = rotation @ exp_so3(omega_mid * dt)
        trace.append({
            "time_s": float(timestamps[index + 1]),
            "px": position[0], "py": position[1], "pz": position[2],
            "vx": velocity[0], "vy": velocity[1], "vz": velocity[2],
            "orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
        })
    return rotation, velocity, position, trace


def fixed_fixture(duration: float, rate: float, omega: np.ndarray, acceleration_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.linspace(0.0, duration, int(round(duration * rate)) + 1)
    gyro = np.repeat(omega[None, :], len(timestamps), axis=0)
    accel = np.repeat((acceleration_world - GRAVITY_WORLD)[None, :], len(timestamps), axis=0)
    return timestamps, gyro, accel


def rotating_thrust_fixture(duration: float, rate: float, yaw_rate: float, thrust: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.linspace(0.0, duration, int(round(duration * rate)) + 1)
    gyro = np.repeat(np.array([[0.0, 0.0, yaw_rate]]), len(timestamps), axis=0)
    accel = []
    for time_s in timestamps:
        rotation = exp_so3(np.array([0.0, 0.0, yaw_rate * time_s]))
        accel.append(rotation.T @ (-GRAVITY_WORLD) + np.array([thrust, 0.0, 0.0]))
    return timestamps, gyro, np.asarray(accel)


def rotating_thrust_truth(duration: float, yaw_rate: float, thrust: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = yaw_rate * duration
    rotation = exp_so3(np.array([0.0, 0.0, angle]))
    velocity = np.array([thrust * np.sin(angle) / yaw_rate, thrust * (1.0 - np.cos(angle)) / yaw_rate, 0.0])
    position = np.array([thrust * (1.0 - np.cos(angle)) / yaw_rate**2, thrust * (duration / yaw_rate - np.sin(angle) / yaw_rate**2), 0.0])
    return rotation, velocity, position


def main() -> int:
    OUT.mkdir(exist_ok=True)
    # Closed-form fixtures.
    static_t, static_g, static_a = fixed_fixture(10.0, 200.0, np.zeros(3), np.zeros(3))
    static_r, static_v, static_p, static_trace = integrate(static_t, static_g, static_a)
    yaw_rate = np.radians(30.0)
    yaw_t, yaw_g, yaw_a = fixed_fixture(6.0, 100.0, np.array([0.0, 0.0, yaw_rate]), np.zeros(3))
    yaw_r, yaw_v, yaw_p, _ = integrate(yaw_t, yaw_g, yaw_a)
    yaw_truth = exp_so3(np.array([0.0, 0.0, yaw_rate * 6.0]))
    acceleration_truth = np.array([0.8, -0.3, 0.2])
    accel_t, accel_g, accel_a = fixed_fixture(8.0, 100.0, np.zeros(3), acceleration_truth)
    accel_r, accel_v, accel_p, accel_trace = integrate(accel_t, accel_g, accel_a)
    velocity_truth = acceleration_truth * 8.0
    position_truth = 0.5 * acceleration_truth * 8.0**2

    # Variable timing fixture: median dt differs from the true mean interval.
    variable_dt = np.tile(np.array([0.003, 0.003, 0.003, 0.011]), 500)
    variable_t = np.concatenate([[0.0], np.cumsum(variable_dt)])
    variable_g = np.zeros((len(variable_t), 3))
    variable_a = np.repeat((acceleration_truth - GRAVITY_WORLD)[None, :], len(variable_t), axis=0)
    _, variable_v, variable_p, variable_trace = integrate(variable_t, variable_g, variable_a)
    stale_t = np.arange(len(variable_t)) * np.median(variable_dt)
    _, stale_v, stale_p, _ = integrate(stale_t, variable_g, variable_a)
    variable_velocity_truth = acceleration_truth * variable_t[-1]
    variable_position_truth = 0.5 * acceleration_truth * variable_t[-1] ** 2

    # Step-size sweep on rotating body thrust with an analytic planar solution.
    sweep_rows = []
    sweep_duration, sweep_yaw_rate, sweep_thrust = 8.0, 0.4, 0.7
    truth_r, truth_v, truth_p = rotating_thrust_truth(sweep_duration, sweep_yaw_rate, sweep_thrust)
    for rate in [25.0, 50.0, 100.0, 200.0]:
        timestamps, gyro, accel = rotating_thrust_fixture(sweep_duration, rate, sweep_yaw_rate, sweep_thrust)
        rotation, velocity, position, _ = integrate(timestamps, gyro, accel)
        sweep_rows.append({
            "rate_hz": rate,
            "dt_s": 1.0 / rate,
            "rotation_error_deg": rotation_error_deg(rotation, truth_r),
            "velocity_error_m_s": float(np.linalg.norm(velocity - truth_v)),
            "position_error_m": float(np.linalg.norm(position - truth_p)),
        })

    # Reverse-time and naive Euler controls.
    roundtrip_rotation = exp_so3(np.array([0.2, -0.1, 0.3])) @ exp_so3(np.array([-0.2, 0.1, -0.3]))
    roundtrip_error_deg = rotation_error_deg(roundtrip_rotation, np.eye(3))
    euler_rotation = np.eye(3)
    euler_dt = 1.0 / 25.0
    for _ in range(int(6.0 / euler_dt)):
        euler_rotation = euler_rotation + euler_rotation @ skew(np.array([0.0, 0.0, yaw_rate])) * euler_dt
    euler_orthogonality_error = float(np.linalg.norm(euler_rotation.T @ euler_rotation - np.eye(3)))

    report = {
        "convention": "R_world_body maps body to world; R_next = R @ Exp((omega_body-b_g)dt); a_world = R_mid(f_body-b_a)+g_world",
        "gravity_world_m_s2": GRAVITY_WORLD.tolist(),
        "fixtures": {
            "stationary": {
                "duration_s": 10.0,
                "position_error_m": float(np.linalg.norm(static_p)),
                "velocity_error_m_s": float(np.linalg.norm(static_v)),
                "rotation_error_deg": rotation_error_deg(static_r, np.eye(3)),
            },
            "constant_yaw": {
                "duration_s": 6.0,
                "expected_yaw_deg": 180.0,
                "rotation_error_deg": rotation_error_deg(yaw_r, yaw_truth),
                "position_error_m": float(np.linalg.norm(yaw_p)),
            },
            "constant_world_acceleration": {
                "duration_s": 8.0,
                "velocity_error_m_s": float(np.linalg.norm(accel_v - velocity_truth)),
                "position_error_m": float(np.linalg.norm(accel_p - position_truth)),
            },
            "variable_dt": {
                "samples": len(variable_dt),
                "duration_s": float(variable_t[-1]),
                "min_dt_s": float(variable_dt.min()),
                "max_dt_s": float(variable_dt.max()),
                "velocity_error_m_s": float(np.linalg.norm(variable_v - variable_velocity_truth)),
                "position_error_m": float(np.linalg.norm(variable_p - variable_position_truth)),
            },
        },
        "step_sweep": sweep_rows,
        "faults": {
            "stale_dt_velocity_error_m_s": float(np.linalg.norm(stale_v - variable_velocity_truth)),
            "stale_dt_position_error_m": float(np.linalg.norm(stale_p - variable_position_truth)),
            "naive_euler_orthogonality_error": euler_orthogonality_error,
        },
        "reverse_rotation_error_deg": roundtrip_error_deg,
        "max_midpoint_orthogonality_error": max(row["orthogonality_error"] for row in variable_trace),
        "command": "python3 scripts/imu_discrete_integration_smoke.py",
        "artifacts": {
            "state_trace": "out/imu_integration_trace.csv",
            "step_sweep": "out/imu_integration_sweep.csv",
            "model": "out/imu_integration_model.npz",
            "plot": "out/imu_integration_plot.png",
        },
    }
    position_errors = [row["position_error_m"] for row in sweep_rows]
    checks = {
        "stationary_invariant": report["fixtures"]["stationary"]["position_error_m"] < 1e-10 and report["fixtures"]["stationary"]["velocity_error_m_s"] < 1e-10,
        "constant_yaw_matches": report["fixtures"]["constant_yaw"]["rotation_error_deg"] < 1e-8 and report["fixtures"]["constant_yaw"]["position_error_m"] < 1e-10,
        "constant_acceleration_matches": report["fixtures"]["constant_world_acceleration"]["position_error_m"] < 1e-9 and report["fixtures"]["constant_world_acceleration"]["velocity_error_m_s"] < 1e-10,
        "variable_dt_matches": report["fixtures"]["variable_dt"]["position_error_m"] < 1e-9 and report["fixtures"]["variable_dt"]["velocity_error_m_s"] < 1e-10,
        "midpoint_converges": all(position_errors[index] > position_errors[index + 1] * 3.5 for index in range(len(position_errors) - 1)),
        "rotations_stay_on_so3": report["max_midpoint_orthogonality_error"] < 1e-11,
        "reverse_rotation_matches": roundtrip_error_deg < 1e-8,
        "stale_dt_fault_visible": report["faults"]["stale_dt_position_error_m"] > 10.0,
        "euler_fault_visible": euler_orthogonality_error > 0.05,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report["checks"] = checks
    report["status"] = "PASS" if all(checks.values()) else "FAIL"

    with (OUT / "imu_integration_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variable_trace[0])); writer.writeheader(); writer.writerows(variable_trace)
    with (OUT / "imu_integration_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0])); writer.writeheader(); writer.writerows(sweep_rows)
    np.savez(
        OUT / "imu_integration_model.npz",
        gravity_world_m_s2=GRAVITY_WORLD,
        variable_timestamps_s=variable_t,
        variable_specific_force_body_m_s2=variable_a,
        variable_gyro_body_rad_s=variable_g,
        final_rotation_world_body=yaw_r,
        constant_acceleration_trace=np.array([[row["time_s"], row["px"], row["py"], row["pz"], row["vx"], row["vy"], row["vz"]] for row in accel_trace]),
    )
    canvas = np.full((720, 1280, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, "DISCRETE IMU INTEGRATION FIXTURES", (45, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 45, 65), 2, cv2.LINE_AA)
    # Step-size convergence.
    left, top, right, bottom = 70, 110, 600, 620
    cv2.rectangle(canvas, (left, top), (right, bottom), (205, 211, 218), 1)
    cv2.putText(canvas, "MIDPOINT POSITION ERROR", (left + 18, top + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 45, 65), 1, cv2.LINE_AA)
    errors = np.array(position_errors)
    for index, row in enumerate(sweep_rows):
        x = left + 80 + index * 120
        height = int(360 * np.log10(errors[index] / errors[-1] + 1.0) / np.log10(errors[0] / errors[-1] + 1.0))
        cv2.rectangle(canvas, (x, bottom - 55 - height), (x + 52, bottom - 55), (45, 135, 205), -1)
        cv2.putText(canvas, f"{int(row['rate_hz'])}Hz", (x - 4, bottom - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 45, 65), 1, cv2.LINE_AA)
    # Variable-dt trusted and stale trajectories.
    cv2.rectangle(canvas, (660, 110), (1210, 620), (205, 211, 218), 1)
    cv2.putText(canvas, "VARIABLE DT POSITION", (680, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 45, 65), 1, cv2.LINE_AA)
    origin = np.array([720.0, 400.0])
    scale = 7.0
    trusted_points = np.array([[row["px"], row["py"]] for row in variable_trace])
    trusted_pixels = np.column_stack([origin[0] + trusted_points[:, 0] * scale, origin[1] - trusted_points[:, 1] * scale]).astype(np.int32)
    cv2.polylines(canvas, [trusted_pixels], False, (205, 135, 45), 4, cv2.LINE_AA)
    stale_pixel = tuple((origin + np.array([stale_p[0], -stale_p[1]]) * scale).astype(int))
    cv2.line(canvas, tuple(origin.astype(int)), stale_pixel, (45, 45, 205), 4, cv2.LINE_AA)
    cv2.putText(canvas, "trusted timestamps", (930, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 135, 45), 1, cv2.LINE_AA)
    cv2.putText(canvas, "stale median dt", (730, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 205), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "imu_integration_plot.png"), canvas)
    (OUT / "imu_integration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("IMU discrete integration lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
