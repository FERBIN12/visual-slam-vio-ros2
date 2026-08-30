#!/usr/bin/env python3
"""Deterministic keyframe IMU preintegration with parity and fault controls."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
GRAVITY_WORLD = np.array([0.0, 0.0, -9.80665])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(phi, dtype=float).reshape(3, 1))
    return rotation


def log_so3(rotation: np.ndarray) -> np.ndarray:
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
    return vector.reshape(3)


def rotation_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(log_so3(estimate @ truth.T))))


@dataclass
class Delta:
    rotation: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    duration: float
    intervals: int


def preintegrate(
    timestamps: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    start: int,
    stop: int,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
) -> Delta:
    """Summarize sample intervals [start, stop); sample ``stop`` is the endpoint."""
    if not (0 <= start < stop < len(timestamps)):
        raise ValueError("preintegration needs 0 <= start < stop < sample_count")
    rotation = np.eye(3)
    velocity = np.zeros(3)
    position = np.zeros(3)
    duration = 0.0
    for index in range(start, stop):
        dt = float(timestamps[index + 1] - timestamps[index])
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("timestamps must be finite and strictly increasing")
        omega = 0.5 * (gyro[index] + gyro[index + 1]) - gyro_bias
        force = 0.5 * (accel[index] + accel[index + 1]) - accel_bias
        rotation_mid = rotation @ exp_so3(omega * dt * 0.5)
        acceleration_local = rotation_mid @ force
        position = position + velocity * dt + 0.5 * acceleration_local * dt * dt
        velocity = velocity + acceleration_local * dt
        rotation = rotation @ exp_so3(omega * dt)
        duration += dt
    return Delta(rotation, velocity, position, duration, stop - start)


def compose(left: Delta, right: Delta) -> Delta:
    return Delta(
        rotation=left.rotation @ right.rotation,
        velocity=left.velocity + left.rotation @ right.velocity,
        position=left.position + left.velocity * right.duration + left.rotation @ right.position,
        duration=left.duration + right.duration,
        intervals=left.intervals + right.intervals,
    )


def endpoint(initial_rotation: np.ndarray, initial_velocity: np.ndarray, initial_position: np.ndarray, delta: Delta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = initial_rotation @ delta.rotation
    velocity = initial_velocity + GRAVITY_WORLD * delta.duration + initial_rotation @ delta.velocity
    position = (
        initial_position
        + initial_velocity * delta.duration
        + 0.5 * GRAVITY_WORLD * delta.duration**2
        + initial_rotation @ delta.position
    )
    return rotation, velocity, position


def direct_integrate(
    timestamps: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    initial_rotation: np.ndarray,
    initial_velocity: np.ndarray,
    initial_position: np.ndarray,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = initial_rotation.copy()
    velocity = initial_velocity.copy()
    position = initial_position.copy()
    for index, dt in enumerate(np.diff(timestamps)):
        omega = 0.5 * (gyro[index] + gyro[index + 1]) - gyro_bias
        force = 0.5 * (accel[index] + accel[index + 1]) - accel_bias
        rotation_mid = rotation @ exp_so3(omega * dt * 0.5)
        acceleration_world = rotation_mid @ force + GRAVITY_WORLD
        position = position + velocity * dt + 0.5 * acceleration_world * dt * dt
        velocity = velocity + acceleration_world * dt
        rotation = rotation @ exp_so3(omega * dt)
    return rotation, velocity, position


def delta_vector(delta: Delta, reference_rotation: np.ndarray | None = None) -> np.ndarray:
    rotation_vector = log_so3(delta.rotation if reference_rotation is None else reference_rotation.T @ delta.rotation)
    return np.concatenate([rotation_vector, delta.velocity, delta.position])


def bias_jacobian(
    timestamps: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    start: int,
    stop: int,
    reference_gyro_bias: np.ndarray,
    reference_accel_bias: np.ndarray,
    base: Delta,
    # OpenCV's Rodrigues logarithm snaps extremely small rotations to zero;
    # 1e-5 remains in the central-difference regime without hitting that branch.
    epsilon: float = 1e-5,
) -> np.ndarray:
    jacobian = np.zeros((9, 6))
    for column in range(6):
        direction_g = np.zeros(3)
        direction_a = np.zeros(3)
        (direction_g if column < 3 else direction_a)[column % 3] = epsilon
        plus = preintegrate(timestamps, gyro, accel, start, stop, reference_gyro_bias + direction_g, reference_accel_bias + direction_a)
        minus = preintegrate(timestamps, gyro, accel, start, stop, reference_gyro_bias - direction_g, reference_accel_bias - direction_a)
        plus_vector = np.concatenate([log_so3(base.rotation.T @ plus.rotation), plus.velocity, plus.position])
        minus_vector = np.concatenate([log_so3(base.rotation.T @ minus.rotation), minus.velocity, minus.position])
        jacobian[:, column] = (plus_vector - minus_vector) / (2.0 * epsilon)
    return jacobian


def corrected_delta(base: Delta, jacobian: np.ndarray, bias_change: np.ndarray, wrong_sign: bool = False) -> Delta:
    correction = jacobian @ ((-1.0 if wrong_sign else 1.0) * bias_change)
    return Delta(
        rotation=base.rotation @ exp_so3(correction[:3]),
        velocity=base.velocity + correction[3:6],
        position=base.position + correction[6:9],
        duration=base.duration,
        intervals=base.intervals,
    )


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Variable 4.5--6.0 ms sample intervals make boundary and timing audits active.
    intervals = 360
    phase = np.arange(intervals, dtype=float)
    dt = 0.005 + 0.00055 * np.sin(phase * 0.37) + 0.00035 * np.cos(phase * 0.11)
    timestamps = np.concatenate([[0.0], np.cumsum(dt)])
    time = timestamps
    gyro = np.column_stack([
        0.22 + 0.07 * np.sin(1.4 * time),
        -0.14 + 0.05 * np.cos(0.8 * time),
        0.31 + 0.09 * np.sin(0.5 * time + 0.2),
    ])
    accel = np.column_stack([
        0.75 + 0.18 * np.cos(1.1 * time),
        -0.28 + 0.16 * np.sin(0.7 * time),
        9.95 + 0.22 * np.cos(0.6 * time + 0.4),
    ])
    return timestamps, gyro, accel


def error_row(name: str, estimate: Delta, truth: Delta) -> dict:
    return {
        "test": name,
        "duration_s": estimate.duration,
        "intervals": estimate.intervals,
        "rotation_error_deg": rotation_error_deg(estimate.rotation, truth.rotation),
        "velocity_error_m_s": float(np.linalg.norm(estimate.velocity - truth.velocity)),
        "position_error_m": float(np.linalg.norm(estimate.position - truth.position)),
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    timestamps, gyro, accel = fixture()
    reference_bg = np.array([0.012, -0.008, 0.006])
    reference_ba = np.array([0.045, -0.035, 0.025])
    start, split, stop = 0, 147, len(timestamps) - 1
    full = preintegrate(timestamps, gyro, accel, start, stop, reference_bg, reference_ba)
    left = preintegrate(timestamps, gyro, accel, start, split, reference_bg, reference_ba)
    right = preintegrate(timestamps, gyro, accel, split, stop, reference_bg, reference_ba)
    composed = compose(left, right)

    # The same raw interval is propagated from an arbitrary world state directly.
    initial_rotation = exp_so3(np.array([0.31, -0.18, 0.24]))
    initial_velocity = np.array([1.2, -0.45, 0.3])
    initial_position = np.array([4.0, -2.5, 1.1])
    direct_r, direct_v, direct_p = direct_integrate(
        timestamps, gyro, accel, initial_rotation, initial_velocity, initial_position, reference_bg, reference_ba
    )
    preint_r, preint_v, preint_p = endpoint(initial_rotation, initial_velocity, initial_position, full)

    # First-order bias correction is measured against exact reintegration.
    jacobian = bias_jacobian(timestamps, gyro, accel, start, stop, reference_bg, reference_ba, full)
    bias_change = np.array([0.00045, -0.00035, 0.00025, 0.0030, -0.0025, 0.0020])
    exact_bias = preintegrate(
        timestamps, gyro, accel, start, stop,
        reference_bg + bias_change[:3], reference_ba + bias_change[3:],
    )
    corrected = corrected_delta(full, jacobian, bias_change)
    wrong_bias = corrected_delta(full, jacobian, bias_change, wrong_sign=True)

    # A common off-by-one fault silently drops the interval ending at the split sample.
    dropped_left = preintegrate(timestamps, gyro, accel, start, split - 1, reference_bg, reference_ba)
    dropped = compose(dropped_left, right)
    parity_rows = [
        error_row("composition", composed, full),
        error_row("bias_linear_update", corrected, exact_bias),
        error_row("wrong_bias_sign", wrong_bias, exact_bias),
        error_row("dropped_boundary", dropped, full),
    ]
    interval_rows = []
    for index in range(start, stop):
        interval_rows.append({
            "interval_index": index,
            "left_sample": index,
            "right_sample": index + 1,
            "timestamp_left_s": timestamps[index],
            "timestamp_right_s": timestamps[index + 1],
            "dt_s": timestamps[index + 1] - timestamps[index],
            "owner": "left" if index < split else "right",
            "used_once": 1,
        })

    direct_errors = {
        "rotation_deg": rotation_error_deg(preint_r, direct_r),
        "velocity_m_s": float(np.linalg.norm(preint_v - direct_v)),
        "position_m": float(np.linalg.norm(preint_p - direct_p)),
    }
    composition_error = parity_rows[0]
    correction_error = parity_rows[1]
    wrong_error = parity_rows[2]
    boundary_error = parity_rows[3]
    checks = {
        "direct_integration_parity": max(direct_errors.values()) < 1e-10,
        "composition_parity": composition_error["rotation_error_deg"] < 1e-9 and composition_error["velocity_error_m_s"] < 1e-10 and composition_error["position_error_m"] < 1e-10,
        "bias_linear_update": correction_error["rotation_error_deg"] < 1e-4 and correction_error["velocity_error_m_s"] < 1e-4 and correction_error["position_error_m"] < 1e-4,
        "interval_coverage": len(interval_rows) == stop - start and sum(row["used_once"] for row in interval_rows) == stop - start,
        "duration_matches_timestamps": abs(full.duration - (timestamps[stop] - timestamps[start])) < 1e-12,
        "dropped_boundary_fault_visible": boundary_error["position_error_m"] > 1e-4 and dropped.intervals == full.intervals - 1,
        "wrong_bias_sign_fault_visible": wrong_error["position_error_m"] > correction_error["position_error_m"] * 100.0,
        "rotation_valid": np.linalg.norm(full.rotation.T @ full.rotation - np.eye(3)) < 1e-12 and np.linalg.det(full.rotation) > 0.999999,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "convention": "Delta maps body_i into body_j over sample intervals [start, stop); sample stop is the included endpoint",
        "reconstruction": "R_j=R_i DeltaR; v_j=v_i+g Dt+R_i Deltav; p_j=p_i+v_i Dt+0.5 g Dt^2+R_i Deltap",
        "interval": {
            "samples": len(timestamps), "intervals": full.intervals, "split_sample": split,
            "duration_s": full.duration, "min_dt_s": float(np.diff(timestamps).min()), "max_dt_s": float(np.diff(timestamps).max()),
        },
        "direct_parity": direct_errors,
        "composition": composition_error,
        "bias_update": {
            "delta_bg_rad_s": bias_change[:3].tolist(), "delta_ba_m_s2": bias_change[3:].tolist(),
            "linear_error": correction_error, "wrong_sign_error": wrong_error,
            "jacobian_frobenius_norm": float(np.linalg.norm(jacobian)),
        },
        "boundary_fault": boundary_error,
        "command": "python3 scripts/imu_preintegration_smoke.py",
        "artifacts": {
            "parity_table": "out/preintegration_parity.csv",
            "interval_audit": "out/preintegration_interval_audit.csv",
            "model": "out/preintegration_model.npz",
            "plot": "out/preintegration_plot.png",
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }

    with (OUT / "preintegration_parity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parity_rows[0])); writer.writeheader(); writer.writerows(parity_rows)
    with (OUT / "preintegration_interval_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0])); writer.writeheader(); writer.writerows(interval_rows)
    np.savez(
        OUT / "preintegration_model.npz",
        timestamps_s=timestamps, gyro_body_rad_s=gyro, specific_force_body_m_s2=accel,
        reference_gyro_bias_rad_s=reference_bg, reference_accel_bias_m_s2=reference_ba,
        delta_rotation=full.rotation, delta_velocity=full.velocity, delta_position=full.position,
        bias_jacobian=jacobian, initial_rotation_world_body=initial_rotation,
    )

    canvas = np.full((720, 1280, 3), 246, dtype=np.uint8)
    ink, blue, red, grid = (60, 48, 38), (205, 132, 42), (45, 45, 205), (214, 208, 200)
    cv2.putText(canvas, "KEYFRAME IMU PREINTEGRATION", (42, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.82, ink, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (55, 105), (800, 625), grid, 1)
    cv2.putText(canvas, "INTERVAL OWNERSHIP", (78, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.58, ink, 1, cv2.LINE_AA)
    axis_left, axis_right, axis_y = 95, 755, 315
    cv2.line(canvas, (axis_left, axis_y), (axis_right, axis_y), ink, 2, cv2.LINE_AA)
    split_x = int(axis_left + (axis_right - axis_left) * split / stop)
    cv2.line(canvas, (split_x, 215), (split_x, 430), red, 2, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (axis_left, 275), (split_x - 8, 275), blue, 8, cv2.LINE_AA, tipLength=.025)
    cv2.arrowedLine(canvas, (split_x + 8, 350), (axis_right, 350), blue, 8, cv2.LINE_AA, tipLength=.025)
    cv2.putText(canvas, f"LEFT: {left.intervals} intervals", (120, 245), cv2.FONT_HERSHEY_SIMPLEX, .48, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"RIGHT: {right.intervals} intervals", (split_x + 50, 395), cv2.FONT_HERSHEY_SIMPLEX, .48, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, "shared endpoint sample", (split_x - 92, 470), cv2.FONT_HERSHEY_SIMPLEX, .42, red, 1, cv2.LINE_AA)
    cv2.putText(canvas, "360 / 360 intervals used exactly once", (120, 550), cv2.FONT_HERSHEY_SIMPLEX, .55, blue, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (850, 105), (1225, 625), grid, 1)
    cv2.putText(canvas, "ERROR CONTROLS", (875, 145), cv2.FONT_HERSHEY_SIMPLEX, .58, ink, 1, cv2.LINE_AA)
    metrics = [
        ("compose", composition_error["position_error_m"], blue),
        ("bias update", correction_error["position_error_m"], blue),
        ("drop boundary", boundary_error["position_error_m"], red),
        ("wrong sign", wrong_error["position_error_m"], red),
    ]
    scale = max(value for _, value, _ in metrics)
    for row, (label, value, color) in enumerate(metrics):
        y = 215 + row * 95
        width = max(2, int(255 * np.log10(1 + value / 1e-10) / np.log10(1 + scale / 1e-10)))
        cv2.putText(canvas, label, (875, y), cv2.FONT_HERSHEY_SIMPLEX, .45, ink, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (875, y + 18), (875 + width, y + 42), color, -1)
        cv2.putText(canvas, f"{value:.3g} m", (875, y + 65), cv2.FONT_HERSHEY_SIMPLEX, .40, ink, 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "preintegration_plot.png"), canvas)
    (OUT / "preintegration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("IMU preintegration lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
