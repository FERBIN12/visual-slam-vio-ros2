#!/usr/bin/env python3
"""Deterministic preintegration covariance, Jacobian, and Monte Carlo lab."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
STATE_NAMES = ["theta_x", "theta_y", "theta_z", "dv_x", "dv_y", "dv_z", "dp_x", "dp_y", "dp_z", "dbg_x", "dbg_y", "dbg_z", "dba_x", "dba_y", "dba_z"]


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(phi, dtype=float).reshape(3, 1))
    return rotation


def log_so3(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]) * 0.5
    if angle < 1e-7:
        return vee
    return vee * angle / np.sin(angle)


@dataclass
class State:
    rotation: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray


def initial_state(bg: np.ndarray, ba: np.ndarray) -> State:
    return State(np.eye(3), np.zeros(3), np.zeros(3), bg.copy(), ba.copy())


def plus(state: State, delta: np.ndarray) -> State:
    return State(
        state.rotation @ exp_so3(delta[:3]),
        state.velocity + delta[3:6], state.position + delta[6:9],
        state.gyro_bias + delta[9:12], state.accel_bias + delta[12:15],
    )


def minus(state: State, reference: State) -> np.ndarray:
    return np.concatenate([
        log_so3(reference.rotation.T @ state.rotation),
        state.velocity - reference.velocity, state.position - reference.position,
        state.gyro_bias - reference.gyro_bias, state.accel_bias - reference.accel_bias,
    ])


def advance(state: State, gyro: np.ndarray, accel: np.ndarray, dt: float, noise: np.ndarray | None = None) -> State:
    noise = np.zeros(12) if noise is None else noise
    omega = gyro + noise[:3] - state.gyro_bias
    force = accel + noise[3:6] - state.accel_bias
    rotation_mid = state.rotation @ exp_so3(omega * dt * 0.5)
    acceleration = rotation_mid @ force
    return State(
        state.rotation @ exp_so3(omega * dt),
        state.velocity + acceleration * dt,
        state.position + state.velocity * dt + 0.5 * acceleration * dt * dt,
        state.gyro_bias + noise[6:9], state.accel_bias + noise[9:12],
    )


def numerical_transition(state: State, gyro: np.ndarray, accel: np.ndarray, dt: float) -> tuple[State, np.ndarray, np.ndarray]:
    nominal = advance(state, gyro, accel, dt)
    f = np.zeros((15, 15)); l = np.zeros((15, 12))
    eps_state, eps_noise = 1e-6, 1e-6
    for column in range(15):
        direction = np.zeros(15); direction[column] = eps_state
        f[:, column] = (minus(advance(plus(state, direction), gyro, accel, dt), nominal) - minus(advance(plus(state, -direction), gyro, accel, dt), nominal)) / (2.0 * eps_state)
    for column in range(12):
        direction = np.zeros(12); direction[column] = eps_noise
        l[:, column] = (minus(advance(state, gyro, accel, dt, direction), nominal) - minus(advance(state, gyro, accel, dt, -direction), nominal)) / (2.0 * eps_noise)
    return nominal, f, l


def fixture(rate_hz: float = 100.0, duration_s: float = 1.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(round(rate_hz * duration_s))
    timestamps = np.linspace(0.0, duration_s, count + 1)
    time = 0.5 * (timestamps[:-1] + timestamps[1:])
    gyro = np.column_stack([0.24 + 0.08 * np.sin(1.3 * time), -0.17 + 0.06 * np.cos(0.9 * time), 0.31 + 0.07 * np.sin(0.6 * time + 0.2)])
    accel = np.column_stack([0.8 + 0.2 * np.cos(1.1 * time), -0.35 + 0.15 * np.sin(0.7 * time), 9.9 + 0.25 * np.cos(0.5 * time + 0.4)])
    return timestamps, gyro, accel


def noise_covariance(dt: float, parameters: dict, wrong_rate_scaling: bool = False) -> np.ndarray:
    gyro_var = parameters["gyro_noise_density"] ** 2 * (1.0 if wrong_rate_scaling else 1.0 / dt)
    accel_var = parameters["accel_noise_density"] ** 2 * (1.0 if wrong_rate_scaling else 1.0 / dt)
    gyro_rw_var = parameters["gyro_bias_random_walk"] ** 2 * dt
    accel_rw_var = parameters["accel_bias_random_walk"] ** 2 * dt
    return np.diag(np.repeat([gyro_var, accel_var, gyro_rw_var, accel_rw_var], 3))


def propagate(rate_hz: float, parameters: dict, wrong_rate_scaling: bool = False, transpose_fault: bool = False) -> tuple[State, np.ndarray, np.ndarray, list[dict], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    timestamps, gyro, accel = fixture(rate_hz)
    state = initial_state(parameters["reference_bg"], parameters["reference_ba"])
    covariance = np.zeros((15, 15)); phi = np.eye(15); trace_rows = []
    for index, dt in enumerate(np.diff(timestamps)):
        state, f, l = numerical_transition(state, gyro[index], accel[index], float(dt))
        transition = f.T if transpose_fault else f
        covariance = transition @ covariance @ transition.T + l @ noise_covariance(float(dt), parameters, wrong_rate_scaling) @ l.T
        covariance = 0.5 * (covariance + covariance.T)
        phi = f @ phi
        eigenvalues = np.linalg.eigvalsh(covariance)
        trace_rows.append({
            "step": index + 1, "time_s": timestamps[index + 1],
            "attitude_variance_trace": np.trace(covariance[:3, :3]),
            "velocity_variance_trace": np.trace(covariance[3:6, 3:6]),
            "position_variance_trace": np.trace(covariance[6:9, 6:9]),
            "gyro_bias_variance_trace": np.trace(covariance[9:12, 9:12]),
            "accel_bias_variance_trace": np.trace(covariance[12:15, 12:15]),
            "minimum_eigenvalue": eigenvalues[0],
            "symmetry_error": np.linalg.norm(covariance - covariance.T),
        })
    return state, covariance, phi, trace_rows, (timestamps, gyro, accel)


def reintegrate(timestamps: np.ndarray, gyro: np.ndarray, accel: np.ndarray, bg: np.ndarray, ba: np.ndarray) -> State:
    state = initial_state(bg, ba)
    for index, dt in enumerate(np.diff(timestamps)):
        state = advance(state, gyro[index], accel[index], float(dt))
    return state


def full_bias_jacobian(timestamps: np.ndarray, gyro: np.ndarray, accel: np.ndarray, bg: np.ndarray, ba: np.ndarray, base: State) -> np.ndarray:
    jacobian = np.zeros((9, 6)); epsilon = 1e-5
    for column in range(6):
        direction_g = np.zeros(3); direction_a = np.zeros(3)
        (direction_g if column < 3 else direction_a)[column % 3] = epsilon
        plus_state = reintegrate(timestamps, gyro, accel, bg + direction_g, ba + direction_a)
        minus_state = reintegrate(timestamps, gyro, accel, bg - direction_g, ba - direction_a)
        jacobian[:, column] = (minus(plus_state, base)[:9] - minus(minus_state, base)[:9]) / (2.0 * epsilon)
    return jacobian


def monte_carlo(timestamps: np.ndarray, gyro: np.ndarray, accel: np.ndarray, parameters: dict, nominal: State, trials: int = 700) -> np.ndarray:
    rng = np.random.default_rng(4405)
    errors = np.zeros((trials, 15))
    for trial in range(trials):
        state = initial_state(parameters["reference_bg"], parameters["reference_ba"])
        for index, dt in enumerate(np.diff(timestamps)):
            noise = np.concatenate([
                rng.normal(0.0, parameters["gyro_noise_density"] / np.sqrt(dt), 3),
                rng.normal(0.0, parameters["accel_noise_density"] / np.sqrt(dt), 3),
                rng.normal(0.0, parameters["gyro_bias_random_walk"] * np.sqrt(dt), 3),
                rng.normal(0.0, parameters["accel_bias_random_walk"] * np.sqrt(dt), 3),
            ])
            state = advance(state, gyro[index], accel[index], float(dt), noise)
        errors[trial] = minus(state, nominal)
    return np.cov(errors, rowvar=False, ddof=1)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    parameters = {
        "gyro_noise_density": 0.0018,
        "accel_noise_density": 0.018,
        "gyro_bias_random_walk": 0.00012,
        "accel_bias_random_walk": 0.0012,
        "reference_bg": np.array([0.012, -0.008, 0.006]),
        "reference_ba": np.array([0.045, -0.035, 0.025]),
    }
    nominal, covariance, phi, trace_rows, fixture_data = propagate(100.0, parameters)
    timestamps, gyro, accel = fixture_data
    finite_jacobian = full_bias_jacobian(timestamps, gyro, accel, parameters["reference_bg"], parameters["reference_ba"], nominal)
    propagated_jacobian = phi[:9, 9:15]
    jacobian_error = propagated_jacobian - finite_jacobian

    empirical = monte_carlo(timestamps, gyro, accel, parameters, nominal)
    predicted_diag = np.diag(covariance); empirical_diag = np.diag(empirical)
    ratios = empirical_diag / predicted_diag
    monte_rows = [{"state": name, "predicted_variance": predicted_diag[i], "empirical_variance": empirical_diag[i], "empirical_over_predicted": ratios[i]} for i, name in enumerate(STATE_NAMES)]

    rate_rows = []
    for rate in [50.0, 100.0, 200.0, 400.0]:
        _, rate_covariance, _, _, _ = propagate(rate, parameters)
        rate_rows.append({
            "rate_hz": rate,
            "attitude_variance_trace": np.trace(rate_covariance[:3, :3]),
            "velocity_variance_trace": np.trace(rate_covariance[3:6, 3:6]),
            "position_variance_trace": np.trace(rate_covariance[6:9, 6:9]),
        })
    _, wrong_covariance, _, _, _ = propagate(100.0, parameters, wrong_rate_scaling=True)
    _, transpose_covariance, _, _, _ = propagate(100.0, parameters, transpose_fault=True)
    trusted_position_trace = float(np.trace(covariance[6:9, 6:9]))
    wrong_rate_ratio = trusted_position_trace / float(np.trace(wrong_covariance[6:9, 6:9]))
    transpose_ratio = float(np.trace(transpose_covariance[6:9, 6:9])) / trusted_position_trace
    symmetry_error = float(np.linalg.norm(covariance - covariance.T))
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance)[0])
    jacobian_max_error = float(np.max(np.abs(jacobian_error)))
    central_ratios = ratios[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]]
    rate_position = np.array([row["position_variance_trace"] for row in rate_rows])
    checks = {
        "covariance_symmetric": symmetry_error < 1e-12,
        "covariance_psd": minimum_eigenvalue > -1e-12,
        "bias_jacobian_matches_finite_difference": jacobian_max_error < 2e-4,
        "monte_carlo_consistent": np.all((central_ratios > 0.78) & (central_ratios < 1.25)),
        "rate_invariant": rate_position.max() / rate_position.min() < 1.03,
        "noise_density_fault_visible": wrong_rate_ratio > 80.0,
        "transpose_fault_visible": transpose_ratio > 1.5 or transpose_ratio < 0.67,
        "state_rotation_valid": np.linalg.norm(nominal.rotation.T @ nominal.rotation - np.eye(3)) < 1e-12,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    jacobian_rows = []
    for row in range(9):
        for column in range(6):
            jacobian_rows.append({
                "output": STATE_NAMES[row], "bias": STATE_NAMES[9 + column],
                "propagated": propagated_jacobian[row, column],
                "finite_difference": finite_jacobian[row, column],
                "absolute_error": abs(jacobian_error[row, column]),
            })
    report = {
        "convention": "15-state right perturbation [theta,dv,dp,dbg,dba]; white densities scale as sigma/sqrt(dt); bias random walks as sigma*sqrt(dt)",
        "fixture": {"rate_hz": 100.0, "duration_s": timestamps[-1], "intervals": len(timestamps) - 1, "monte_carlo_trials": 700},
        "noise": {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in parameters.items()},
        "covariance": {"symmetry_error": symmetry_error, "minimum_eigenvalue": minimum_eigenvalue, "final_diagonal": predicted_diag.tolist()},
        "jacobian": {"shape": list(propagated_jacobian.shape), "max_absolute_error": jacobian_max_error, "frobenius_error": float(np.linalg.norm(jacobian_error))},
        "monte_carlo": {"minimum_ratio": float(central_ratios.min()), "maximum_ratio": float(central_ratios.max()), "median_ratio": float(np.median(central_ratios))},
        "rate_sweep": rate_rows,
        "faults": {"sample_sigma_as_density_underestimate_ratio": wrong_rate_ratio, "transposed_transition_position_trace_ratio": transpose_ratio},
        "command": "python3 scripts/imu_covariance_jacobian_smoke.py",
        "artifacts": {
            "covariance_trace": "out/preintegration_covariance_trace.csv",
            "jacobian_table": "out/preintegration_jacobian.csv",
            "monte_carlo_table": "out/preintegration_monte_carlo.csv",
            "model": "out/preintegration_covariance_model.npz",
            "plot": "out/preintegration_covariance_plot.png",
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_csv(OUT / "preintegration_covariance_trace.csv", trace_rows)
    write_csv(OUT / "preintegration_jacobian.csv", jacobian_rows)
    write_csv(OUT / "preintegration_monte_carlo.csv", monte_rows)
    write_csv(OUT / "preintegration_covariance_rate_sweep.csv", rate_rows)
    np.savez(
        OUT / "preintegration_covariance_model.npz",
        covariance=covariance, empirical_covariance=empirical, state_transition=phi,
        propagated_bias_jacobian=propagated_jacobian, finite_difference_bias_jacobian=finite_jacobian,
        timestamps_s=timestamps, gyro_body_rad_s=gyro, specific_force_body_m_s2=accel,
        final_rotation=nominal.rotation, final_velocity=nominal.velocity, final_position=nominal.position,
    )
    canvas = np.full((720, 1280, 3), 246, np.uint8); ink, blue, teal, red, grid = (58, 46, 36), (205, 132, 42), (135, 145, 25), (45, 45, 205), (214, 208, 200)
    cv2.putText(canvas, "PREINTEGRATION UNCERTAINTY AUDIT", (45, 50), cv2.FONT_HERSHEY_SIMPLEX, .78, ink, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (55, 100), (785, 635), grid, 1); cv2.putText(canvas, "PREDICTED vs MONTE CARLO", (78, 140), cv2.FONT_HERSHEY_SIMPLEX, .55, ink, 1, cv2.LINE_AA)
    chosen = [0, 3, 6, 9, 12]
    for index, state_index in enumerate(chosen):
        x = 110 + index * 132; scale = 1e6 / max(predicted_diag[chosen].max(), empirical_diag[chosen].max())
        ph = max(2, int(330 * predicted_diag[state_index] / max(predicted_diag[chosen].max(), empirical_diag[chosen].max())))
        eh = max(2, int(330 * empirical_diag[state_index] / max(predicted_diag[chosen].max(), empirical_diag[chosen].max())))
        cv2.rectangle(canvas, (x, 560 - ph), (x + 42, 560), blue, -1); cv2.rectangle(canvas, (x + 46, 560 - eh), (x + 88, 560), teal, -1)
        cv2.putText(canvas, STATE_NAMES[state_index].split("_")[0], (x, 595), cv2.FONT_HERSHEY_SIMPLEX, .38, ink, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (835, 100), (1225, 635), grid, 1); cv2.putText(canvas, "ACTIVE CONTROLS", (860, 140), cv2.FONT_HERSHEY_SIMPLEX, .55, ink, 1, cv2.LINE_AA)
    controls = [("J max error", jacobian_max_error, blue), ("rate mistake", wrong_rate_ratio, red), ("F transpose", transpose_ratio, red)]
    for index, (label, value, color) in enumerate(controls):
        y = 225 + index * 125; cv2.putText(canvas, label, (875, y), cv2.FONT_HERSHEY_SIMPLEX, .48, ink, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.4g}", (875, y + 48), cv2.FONT_HERSHEY_SIMPLEX, .72, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "preintegration_covariance_plot.png"), canvas)
    (OUT / "preintegration_covariance_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("IMU covariance and Jacobian lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
