#!/usr/bin/env python3
"""Deterministic motion-only bundle-adjustment lab with robust and L2 controls."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
K = np.array([[720.0, 0.0, 640.0], [0.0, 715.0, 360.0], [0.0, 0.0, 1.0]])
WIDTH, HEIGHT = 1280, 720
HUBER_DELTA_PX = 2.5


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(w, dtype=float).reshape(3, 1))
    return rotation


def apply_left_update(rotation: np.ndarray, translation: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    update_rotation = exp_so3(delta[:3])
    return update_rotation @ rotation, update_rotation @ translation + delta[3:]


def project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = (rotation @ points.T).T + translation
    normalized = camera[:, :2] / camera[:, 2:3]
    pixels = np.column_stack(
        [K[0, 0] * normalized[:, 0] + K[0, 2], K[1, 1] * normalized[:, 1] + K[1, 2]]
    )
    return pixels, camera


def residual_and_jacobian(points: np.ndarray, observed: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted, camera = project(points, rotation, translation)
    residual = observed - predicted
    jacobian = np.zeros((len(points), 2, 6))
    for index, (x, y, z) in enumerate(camera):
        projection_jacobian = np.array(
            [[K[0, 0] / z, 0.0, -K[0, 0] * x / (z * z)], [0.0, K[1, 1] / z, -K[1, 1] * y / (z * z)]]
        )
        predicted_jacobian = projection_jacobian @ np.hstack([-skew(camera[index]), np.eye(3)])
        jacobian[index] = -predicted_jacobian
    return residual, jacobian, predicted


def huber_cost_and_weights(residual: np.ndarray, robust: bool) -> tuple[float, np.ndarray]:
    norms = np.linalg.norm(residual, axis=1)
    if not robust:
        return float(0.5 * np.sum(norms * norms)), np.ones(len(norms))
    weights = np.ones(len(norms))
    large = norms > HUBER_DELTA_PX
    weights[large] = HUBER_DELTA_PX / np.maximum(norms[large], 1e-12)
    cost = np.where(
        large,
        HUBER_DELTA_PX * (norms - 0.5 * HUBER_DELTA_PX),
        0.5 * norms * norms,
    )
    return float(np.sum(cost)), weights


def optimize(points: np.ndarray, observed: np.ndarray, initial_rotation: np.ndarray, initial_translation: np.ndarray, robust: bool) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rotation = initial_rotation.copy()
    translation = initial_translation.copy()
    damping = 1e-3
    trace: list[dict] = []
    for iteration in range(30):
        residual, jacobian, _ = residual_and_jacobian(points, observed, rotation, translation)
        cost, weights = huber_cost_and_weights(residual, robust)
        weighted_jacobian = jacobian.reshape(-1, 6) * np.repeat(np.sqrt(weights), 2)[:, None]
        weighted_residual = residual.reshape(-1) * np.repeat(np.sqrt(weights), 2)
        information = weighted_jacobian.T @ weighted_jacobian
        gradient = weighted_jacobian.T @ weighted_residual
        accepted = False
        trial_delta = np.zeros(6)
        trial_cost = cost
        for _ in range(12):
            system = information + damping * np.diag(np.maximum(np.diag(information), 1.0))
            trial_delta = np.linalg.solve(system, -gradient)
            trial_rotation, trial_translation = apply_left_update(rotation, translation, trial_delta)
            trial_residual, _, _ = residual_and_jacobian(points, observed, trial_rotation, trial_translation)
            trial_cost, _ = huber_cost_and_weights(trial_residual, robust)
            if trial_cost < cost:
                rotation, translation = trial_rotation, trial_translation
                damping = max(damping / 3.0, 1e-9)
                accepted = True
                break
            damping = min(damping * 10.0, 1e12)
        trace.append(
            {
                "iteration": iteration,
                "cost_before": cost,
                "cost_after": trial_cost,
                "step_norm": float(np.linalg.norm(trial_delta)),
                "damping": damping,
                "downweighted_rows": int(np.sum(weights < 1.0)),
                "accepted": int(accepted),
            }
        )
        if not accepted or np.linalg.norm(trial_delta) < 1e-9:
            break
    return rotation, translation, trace


def rotation_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    delta = estimate @ truth.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return -rotation.T @ translation


def check_jacobian(point: np.ndarray, observed: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> float:
    _, analytic, _ = residual_and_jacobian(point[None], observed[None], rotation, translation)
    epsilon = 1e-7
    numeric = np.zeros((2, 6))
    for column in range(6):
        step = np.zeros(6)
        step[column] = epsilon
        plus_rotation, plus_translation = apply_left_update(rotation, translation, step)
        minus_rotation, minus_translation = apply_left_update(rotation, translation, -step)
        plus, _, _ = residual_and_jacobian(point[None], observed[None], plus_rotation, plus_translation)
        minus, _, _ = residual_and_jacobian(point[None], observed[None], minus_rotation, minus_translation)
        numeric[:, column] = ((plus - minus) / (2.0 * epsilon)).reshape(2)
    return float(np.max(np.abs(numeric - analytic.reshape(2, 6))))


def draw_overlay(observed: np.ndarray, before: np.ndarray, after: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), 246, dtype=np.uint8)
    for index in range(len(observed)):
        obs = tuple(np.round(observed[index]).astype(int))
        start = tuple(np.round(before[index]).astype(int))
        end = tuple(np.round(after[index]).astype(int))
        if labels[index] == "injected_outlier":
            color = (45, 45, 210)
        elif weights[index] < 0.999:
            color = (45, 145, 205)
        else:
            color = (170, 120, 25)
        cv2.line(canvas, start, end, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, end, obs, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, obs, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, end, 4, (45, 170, 125), 1, cv2.LINE_AA)
    cv2.putText(canvas, "dot: observed  ring: optimized  grey: pose update  red: injected", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (45, 50, 65), 2)
    return canvas


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(53)
    fit_clean_count, outlier_count, holdout_count = 180, 30, 40
    truth_rotation = exp_so3(np.radians(np.array([1.8, -5.2, 2.4])))
    truth_translation = np.array([0.32, -0.08, 0.22])
    all_clean_points = np.column_stack(
        [rng.uniform(-2.8, 2.8, fit_clean_count + holdout_count), rng.uniform(-1.7, 1.7, fit_clean_count + holdout_count), rng.uniform(4.0, 11.0, fit_clean_count + holdout_count)]
    )
    all_clean_pixels, _ = project(all_clean_points, truth_rotation, truth_translation)
    all_clean_pixels += rng.normal(0.0, 0.42, all_clean_pixels.shape)
    fit_points = all_clean_points[:fit_clean_count]
    fit_pixels = all_clean_pixels[:fit_clean_count]
    chosen = rng.choice(fit_clean_count, outlier_count, replace=False)
    outlier_pixels = fit_pixels[chosen] + rng.uniform([-130.0, -100.0], [130.0, 100.0], (outlier_count, 2))
    points = np.vstack([fit_points, fit_points[chosen]])
    observed = np.vstack([fit_pixels, outlier_pixels])
    labels = np.array(["clean"] * fit_clean_count + ["injected_outlier"] * outlier_count)

    initial_rotation = exp_so3(np.radians(np.array([3.5, -2.5, 2.0]))) @ truth_rotation
    initial_translation = truth_translation + np.array([0.20, -0.14, 0.18])
    before, _ = project(points, initial_rotation, initial_translation)
    robust_rotation, robust_translation, robust_trace = optimize(points, observed, initial_rotation, initial_translation, True)
    l2_rotation, l2_translation, l2_trace = optimize(points, observed, initial_rotation, initial_translation, False)
    after, _ = project(points, robust_rotation, robust_translation)
    residual, _, _ = residual_and_jacobian(points, observed, robust_rotation, robust_translation)
    residual_norm = np.linalg.norm(residual, axis=1)
    _, final_weights = huber_cost_and_weights(residual, True)
    clean = labels == "clean"
    outlier = ~clean
    holdout_prediction, _ = project(all_clean_points[-holdout_count:], robust_rotation, robust_translation)
    holdout_residual = np.linalg.norm(holdout_prediction - all_clean_pixels[-holdout_count:], axis=1)

    initial_rotation_error = rotation_error_deg(initial_rotation, truth_rotation)
    robust_rotation_error = rotation_error_deg(robust_rotation, truth_rotation)
    robust_center_error = float(np.linalg.norm(camera_center(robust_rotation, robust_translation) - camera_center(truth_rotation, truth_translation)))
    l2_rotation_error = rotation_error_deg(l2_rotation, truth_rotation)
    l2_center_error = float(np.linalg.norm(camera_center(l2_rotation, l2_translation) - camera_center(truth_rotation, truth_translation)))
    jacobian_error = check_jacobian(fit_points[0], fit_pixels[0], initial_rotation, initial_translation)
    robust_cost_start = robust_trace[0]["cost_before"]
    robust_cost_end = robust_trace[-1]["cost_after"]

    report = {
        "fit_rows": int(len(points)),
        "clean_rows": fit_clean_count,
        "injected_outliers": outlier_count,
        "holdout_rows": holdout_count,
        "huber_delta_px": HUBER_DELTA_PX,
        "iterations": len(robust_trace),
        "initial_rotation_error_deg": initial_rotation_error,
        "rotation_error_deg": robust_rotation_error,
        "camera_center_error_m": robust_center_error,
        "clean_median_reprojection_px": float(np.median(residual_norm[clean])),
        "clean_p95_reprojection_px": float(np.percentile(residual_norm[clean], 95)),
        "holdout_median_reprojection_px": float(np.median(holdout_residual)),
        "holdout_p95_reprojection_px": float(np.percentile(holdout_residual, 95)),
        "outliers_downweighted": int(np.sum(final_weights[outlier] < 0.25)),
        "clean_rows_full_weight": int(np.sum(final_weights[clean] == 1.0)),
        "robust_cost_start": robust_cost_start,
        "robust_cost_end": robust_cost_end,
        "jacobian_max_abs_error": jacobian_error,
        "plain_l2_control": {
            "iterations": len(l2_trace),
            "rotation_error_deg": l2_rotation_error,
            "camera_center_error_m": l2_center_error,
        },
        "convention": "left SE(3) update on x_camera = R_camera_map @ x_map + t_camera_map; residual = observed - projected",
        "command": "python3 scripts/motion_ba_smoke.py",
        "artifacts": {
            "report": "out/motion_ba_report.json",
            "trace": "out/motion_ba_trace.csv",
            "observations": "out/motion_ba_observations.csv",
            "model": "out/motion_ba_model.npz",
            "overlay": "out/motion_ba_overlay.png",
        },
    }
    checks = {
        "cost_reduced": robust_cost_end < robust_cost_start * 0.25,
        "rotation_improved": robust_rotation_error < 0.1 and robust_rotation_error < initial_rotation_error * 0.05,
        "metric_center_accuracy": robust_center_error < 0.01,
        "clean_reprojection": report["clean_p95_reprojection_px"] < 1.2,
        "holdout_generalization": report["holdout_p95_reprojection_px"] < 1.3,
        "outliers_downweighted": report["outliers_downweighted"] >= 28,
        "jacobian_verified": jacobian_error < 1e-4,
        "l2_failure_visible": l2_center_error > robust_center_error * 8.0 and l2_rotation_error > robust_rotation_error * 5.0,
    }
    report["checks"] = checks
    report["status"] = "PASS" if all(checks.values()) else "FAIL"

    with (OUT / "motion_ba_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(robust_trace[0]))
        writer.writeheader()
        writer.writerows(robust_trace)
    with (OUT / "motion_ba_observations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "x_map", "y_map", "z_map", "u_observed", "v_observed", "u_before", "v_before", "u_after", "v_after", "reprojection_px", "robust_weight"])
        for index in range(len(points)):
            writer.writerow([index, labels[index], *points[index], *observed[index], *before[index], *after[index], residual_norm[index], final_weights[index]])
    np.savez(
        OUT / "motion_ba_model.npz",
        K=K,
        rotation_camera_map=robust_rotation,
        translation_camera_map=robust_translation,
        camera_center_map=camera_center(robust_rotation, robust_translation),
        truth_rotation=truth_rotation,
        truth_translation=truth_translation,
        initial_rotation=initial_rotation,
        initial_translation=initial_translation,
        final_weights=final_weights,
    )
    cv2.imwrite(str(OUT / "motion_ba_overlay.png"), draw_overlay(observed, before, after, labels, final_weights))
    (OUT / "motion_ba_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Motion-only BA lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
