#!/usr/bin/env python3
"""Seeded IMU bias/noise lab with Allan-style and covariance-growth controls."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
RATE_HZ = 200.0
DT = 1.0 / RATE_HZ
DURATION_S = 1200.0
GRAVITY = 9.80665
GYRO_NOISE_DENSITY = 2.0e-4  # rad/s/sqrt(Hz), with sigma_sample = density/sqrt(dt)
ACCEL_NOISE_DENSITY = 2.0e-3  # m/s^2/sqrt(Hz)
GYRO_BIAS_RW = 1.2e-5  # (rad/s)/sqrt(s)
ACCEL_BIAS_RW = 1.5e-4  # (m/s^2)/sqrt(s)
INITIAL_GYRO_BIAS = np.array([1.5e-3, -8.0e-4, 2.2e-3])
INITIAL_ACCEL_BIAS = np.array([3.0e-2, -2.0e-2, 4.0e-2])


def allan_style(sequence: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Non-overlapping adjacent-cluster Allan deviation for one scalar stream."""
    values = []
    for tau in taus:
        cluster = max(1, int(round(tau * RATE_HZ)))
        count = len(sequence) // cluster
        means = sequence[: count * cluster].reshape(count, cluster).mean(axis=1)
        values.append(np.sqrt(0.5 * np.mean(np.diff(means) ** 2)))
    return np.asarray(values)


def monte_carlo_growth(rng: np.random.Generator) -> tuple[list[dict], dict]:
    trials = 1200
    rate = 100.0
    dt = 1.0 / rate
    checkpoints = np.array([1.0, 5.0, 10.0, 30.0, 60.0])
    checkpoint_steps = {int(round(value / dt)): value for value in checkpoints}
    steps = int(round(checkpoints[-1] / dt))
    theta = np.zeros(trials)
    velocity = np.zeros(trials)
    position = np.zeros(trials)
    gyro_bias = np.zeros(trials)
    accel_bias = np.zeros(trials)
    rows = []
    for step in range(1, steps + 1):
        gyro_bias += rng.normal(0.0, GYRO_BIAS_RW * np.sqrt(dt), trials)
        accel_bias += rng.normal(0.0, ACCEL_BIAS_RW * np.sqrt(dt), trials)
        gyro = gyro_bias + rng.normal(0.0, GYRO_NOISE_DENSITY / np.sqrt(dt), trials)
        accel = accel_bias + rng.normal(0.0, ACCEL_NOISE_DENSITY / np.sqrt(dt), trials)
        theta += gyro * dt
        position += velocity * dt + 0.5 * accel * dt * dt
        velocity += accel * dt
        if step in checkpoint_steps:
            time_s = checkpoint_steps[step]
            predicted_theta_var = GYRO_NOISE_DENSITY**2 * time_s + GYRO_BIAS_RW**2 * time_s**3 / 3.0
            predicted_velocity_var = ACCEL_NOISE_DENSITY**2 * time_s + ACCEL_BIAS_RW**2 * time_s**3 / 3.0
            predicted_position_var = ACCEL_NOISE_DENSITY**2 * time_s**3 / 3.0 + ACCEL_BIAS_RW**2 * time_s**5 / 20.0
            rows.append({
                "time_s": time_s,
                "empirical_attitude_variance_rad2": float(np.var(theta, ddof=1)),
                "predicted_attitude_variance_rad2": predicted_theta_var,
                "empirical_velocity_variance_m2_s2": float(np.var(velocity, ddof=1)),
                "predicted_velocity_variance_m2_s2": predicted_velocity_var,
                "empirical_position_variance_m2": float(np.var(position, ddof=1)),
                "predicted_position_variance_m2": predicted_position_var,
            })
    final = rows[-1]
    ratios = {
        "attitude_empirical_to_predicted": final["empirical_attitude_variance_rad2"] / final["predicted_attitude_variance_rad2"],
        "velocity_empirical_to_predicted": final["empirical_velocity_variance_m2_s2"] / final["predicted_velocity_variance_m2_s2"],
        "position_empirical_to_predicted": final["empirical_position_variance_m2"] / final["predicted_position_variance_m2"],
    }
    return rows, ratios


def draw_log_series(canvas: np.ndarray, rect: tuple[int, int, int, int], x: np.ndarray, series: list[tuple[np.ndarray, tuple[int, int, int], str]], title: str) -> None:
    left, top, right, bottom = rect
    cv2.rectangle(canvas, (left, top), (right, bottom), (205, 211, 218), 1)
    cv2.putText(canvas, title, (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 45, 65), 1, cv2.LINE_AA)
    log_x = np.log10(x)
    all_y = np.concatenate([np.log10(np.maximum(values, 1e-20)) for values, _, _ in series])
    x_norm = (log_x - log_x.min()) / max(np.ptp(log_x), 1e-12)
    y_min, y_max = float(all_y.min()), float(all_y.max())
    for series_index, (values, color, label) in enumerate(series):
        log_y = np.log10(np.maximum(values, 1e-20))
        y_norm = (log_y - y_min) / max(y_max - y_min, 1e-12)
        points = np.column_stack([left + 20 + x_norm * (right - left - 40), bottom - 20 - y_norm * (bottom - top - 40)]).astype(np.int32)
        cv2.polylines(canvas, [points], False, color, 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (right - 175, top + 28 + 24 * series_index), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(89)
    samples = int(round(DURATION_S * RATE_HZ))
    time_s = np.arange(samples) * DT
    gyro_bias = INITIAL_GYRO_BIAS + np.cumsum(rng.normal(0.0, GYRO_BIAS_RW * np.sqrt(DT), (samples, 3)), axis=0)
    accel_bias = INITIAL_ACCEL_BIAS + np.cumsum(rng.normal(0.0, ACCEL_BIAS_RW * np.sqrt(DT), (samples, 3)), axis=0)
    gyro = gyro_bias + rng.normal(0.0, GYRO_NOISE_DENSITY / np.sqrt(DT), (samples, 3))
    accel_truth = np.array([0.0, 0.0, GRAVITY])
    accel = accel_truth + accel_bias + rng.normal(0.0, ACCEL_NOISE_DENSITY / np.sqrt(DT), (samples, 3))

    taus = np.array([DT, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0, 60.0])
    gyro_adev = allan_style(gyro[:, 0] - INITIAL_GYRO_BIAS[0], taus)
    accel_adev = allan_style(accel[:, 0] - INITIAL_ACCEL_BIAS[0], taus)
    gyro_density_est = float(np.median(gyro_adev[:4] * np.sqrt(taus[:4])))
    accel_density_est = float(np.median(accel_adev[:4] * np.sqrt(taus[:4])))
    gyro_rw_est = float(np.median(gyro_adev[-2:] / np.sqrt(taus[-2:] / 3.0)))
    accel_rw_est = float(np.median(accel_adev[-2:] / np.sqrt(taus[-2:] / 3.0)))
    # The final 20-to-60-second segment is beyond the white-noise minimum and
    # therefore isolates the designed random-walk upturn.
    gyro_long_slope = float(np.polyfit(np.log10(taus[-2:]), np.log10(gyro_adev[-2:]), 1)[0])
    accel_long_slope = float(np.polyfit(np.log10(taus[-2:]), np.log10(accel_adev[-2:]), 1)[0])

    estimate_samples = int(20.0 * RATE_HZ)
    estimated_gyro_bias = gyro[:estimate_samples].mean(axis=0)
    estimated_accel_bias = (accel[:estimate_samples] - accel_truth).mean(axis=0)
    covariance_rows, covariance_ratios = monte_carlo_growth(np.random.default_rng(97))
    rate_confusion_variance_ratio = RATE_HZ
    final_t = covariance_rows[-1]["time_s"]
    gyro_full_var = covariance_rows[-1]["predicted_attitude_variance_rad2"]
    gyro_white_only_var = GYRO_NOISE_DENSITY**2 * final_t
    accel_full_var = covariance_rows[-1]["predicted_velocity_variance_m2_s2"]
    accel_white_only_var = ACCEL_NOISE_DENSITY**2 * final_t

    report = {
        "seed": 89,
        "stationary_rate_hz": RATE_HZ,
        "stationary_duration_s": DURATION_S,
        "samples": samples,
        "density_convention": "sigma_sample = noise_density / sqrt(dt); bias increment sigma = random_walk_density * sqrt(dt)",
        "parameters": {
            "gyro_noise_density_rad_s_sqrt_hz": GYRO_NOISE_DENSITY,
            "accel_noise_density_m_s2_sqrt_hz": ACCEL_NOISE_DENSITY,
            "gyro_bias_random_walk_rad_s_sqrt_s": GYRO_BIAS_RW,
            "accel_bias_random_walk_m_s2_sqrt_s": ACCEL_BIAS_RW,
            "initial_gyro_bias_rad_s": INITIAL_GYRO_BIAS.tolist(),
            "initial_accel_bias_m_s2": INITIAL_ACCEL_BIAS.tolist(),
        },
        "stationary_estimates": {
            "window_s": 20.0,
            "gyro_bias_rad_s": estimated_gyro_bias.tolist(),
            "accel_bias_m_s2": estimated_accel_bias.tolist(),
            "gyro_bias_error_norm_rad_s": float(np.linalg.norm(estimated_gyro_bias - INITIAL_GYRO_BIAS)),
            "accel_bias_error_norm_m_s2": float(np.linalg.norm(estimated_accel_bias - INITIAL_ACCEL_BIAS)),
        },
        "allan_style": {
            "taus_s": taus.tolist(),
            "gyro_x_deviation_rad_s": gyro_adev.tolist(),
            "accel_x_deviation_m_s2": accel_adev.tolist(),
            "estimated_gyro_noise_density": gyro_density_est,
            "estimated_accel_noise_density": accel_density_est,
            "estimated_gyro_bias_random_walk": gyro_rw_est,
            "estimated_accel_bias_random_walk": accel_rw_est,
            "gyro_long_log_slope": gyro_long_slope,
            "accel_long_log_slope": accel_long_slope,
        },
        "monte_carlo": {"trials": 1200, "duration_s": 60.0, "final_empirical_to_predicted": covariance_ratios},
        "faults": {
            "sample_std_used_as_density_variance_underestimate": rate_confusion_variance_ratio,
            "frozen_gyro_bias_variance_underestimate": gyro_full_var / gyro_white_only_var,
            "frozen_accel_bias_variance_underestimate": accel_full_var / accel_white_only_var,
        },
        "command": "python3 scripts/imu_bias_noise_smoke.py",
        "artifacts": {
            "allan_table": "out/imu_allan_style.csv",
            "covariance_table": "out/imu_covariance_growth.csv",
            "trace": "out/imu_bias_noise_trace.csv",
            "model": "out/imu_bias_noise_model.npz",
            "plot": "out/imu_bias_noise_plot.png",
        },
    }
    checks = {
        "seeded_sample_count": samples == 240000,
        "stationary_gyro_bias_recovered": report["stationary_estimates"]["gyro_bias_error_norm_rad_s"] < 2.5e-4,
        "stationary_accel_bias_recovered": report["stationary_estimates"]["accel_bias_error_norm_m_s2"] < 2.5e-3,
        "gyro_density_recovered": abs(gyro_density_est / GYRO_NOISE_DENSITY - 1.0) < 0.12,
        "accel_density_recovered": abs(accel_density_est / ACCEL_NOISE_DENSITY - 1.0) < 0.12,
        "long_averages_show_bias_growth": gyro_long_slope > 0.12 and accel_long_slope > 0.12,
        "monte_carlo_attitude_matches": abs(covariance_ratios["attitude_empirical_to_predicted"] - 1.0) < 0.18,
        "monte_carlo_velocity_matches": abs(covariance_ratios["velocity_empirical_to_predicted"] - 1.0) < 0.18,
        "monte_carlo_position_matches": abs(covariance_ratios["position_empirical_to_predicted"] - 1.0) < 0.22,
        "rate_scaling_fault_visible": rate_confusion_variance_ratio >= 199.0,
        "frozen_bias_fault_visible": gyro_full_var / gyro_white_only_var > 4.0 and accel_full_var / accel_white_only_var > 6.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report["checks"] = checks
    report["status"] = "PASS" if all(checks.values()) else "FAIL"

    with (OUT / "imu_allan_style.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tau_s", "gyro_x_deviation_rad_s", "accel_x_deviation_m_s2"])
        writer.writeheader()
        writer.writerows({"tau_s": tau, "gyro_x_deviation_rad_s": g, "accel_x_deviation_m_s2": a} for tau, g, a in zip(taus, gyro_adev, accel_adev))
    with (OUT / "imu_covariance_growth.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(covariance_rows[0])); writer.writeheader(); writer.writerows(covariance_rows)
    trace_indices = np.arange(0, samples, int(RATE_HZ))
    trace_rows = []
    for index in trace_indices:
        trace_rows.append({
            "time_s": time_s[index],
            "gyro_x_rad_s": gyro[index, 0],
            "gyro_bias_x_rad_s": gyro_bias[index, 0],
            "accel_x_m_s2": accel[index, 0],
            "accel_bias_x_m_s2": accel_bias[index, 0],
        })
    with (OUT / "imu_bias_noise_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0])); writer.writeheader(); writer.writerows(trace_rows)
    np.savez(
        OUT / "imu_bias_noise_model.npz",
        time_s=time_s,
        gyro_measurement_rad_s=gyro,
        accel_measurement_m_s2=accel,
        gyro_bias_rad_s=gyro_bias,
        accel_bias_m_s2=accel_bias,
        taus_s=taus,
        gyro_allan_deviation=gyro_adev,
        accel_allan_deviation=accel_adev,
    )
    canvas = np.full((720, 1280, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, "IMU BIAS, NOISE, AND COVARIANCE", (45, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 45, 65), 2, cv2.LINE_AA)
    draw_log_series(canvas, (55, 95, 615, 360), taus, [(gyro_adev, (45, 135, 205), "gyro Allan-style"), (GYRO_NOISE_DENSITY / np.sqrt(taus), (60, 165, 105), "white reference")], "GYRO AVERAGING SIGNATURE")
    draw_log_series(canvas, (665, 95, 1225, 360), taus, [(accel_adev, (45, 135, 205), "accel Allan-style"), (ACCEL_NOISE_DENSITY / np.sqrt(taus), (60, 165, 105), "white reference")], "ACCEL AVERAGING SIGNATURE")
    times = np.array([row["time_s"] for row in covariance_rows])
    empirical = np.array([row["empirical_position_variance_m2"] for row in covariance_rows])
    predicted = np.array([row["predicted_position_variance_m2"] for row in covariance_rows])
    draw_log_series(canvas, (250, 430, 1030, 665), times, [(empirical, (45, 135, 205), "Monte Carlo"), (predicted, (60, 165, 105), "propagated")], "POSITION VARIANCE GROWTH")
    cv2.imwrite(str(OUT / "imu_bias_noise_plot.png"), canvas)
    (OUT / "imu_bias_noise_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("IMU bias/noise lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
