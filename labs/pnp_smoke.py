#!/usr/bin/env python3
"""Deterministic metric PnP lab with outliers and a clustered-landmark control."""
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


def project(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    pixels, _ = cv2.projectPoints(points, rvec, tvec, K, None)
    return pixels.reshape(-1, 2)


def rotation_error_deg(rvec: np.ndarray, truth: np.ndarray) -> float:
    estimate, _ = cv2.Rodrigues(rvec)
    expected, _ = cv2.Rodrigues(truth)
    delta = estimate @ expected.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def camera_center(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    return (-rotation.T @ tvec.reshape(3, 1)).reshape(3)


def coverage(pixels: np.ndarray, mask: np.ndarray) -> float:
    selected = pixels[mask]
    if not len(selected):
        return 0.0
    cols = np.clip((selected[:, 0] / WIDTH * 4).astype(int), 0, 3)
    rows = np.clip((selected[:, 1] / HEIGHT * 3).astype(int), 0, 2)
    return float(len(set(zip(cols.tolist(), rows.tolist()))) / 12.0)


def pose_condition(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> float:
    _, jacobian = cv2.projectPoints(points, rvec, tvec, K, None)
    pose_jacobian = jacobian[:, :6]
    singular = np.linalg.svd(pose_jacobian, compute_uv=False)
    return float((singular[0] / max(singular[-1], 1e-12)) ** 2)


def solve(points: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points,
        pixels,
        K,
        None,
        iterationsCount=3000,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        raise RuntimeError("PnP failed to produce a supported pose")
    accepted = np.zeros(len(points), dtype=bool)
    accepted[inliers.reshape(-1)] = True
    rvec, tvec = cv2.solvePnPRefineLM(
        points[accepted], pixels[accepted], K, None, rvec, tvec
    )
    return rvec.reshape(3), tvec.reshape(3), accepted


def overlay(points: np.ndarray, observed: np.ndarray, predicted: np.ndarray, accepted: np.ndarray, injected: np.ndarray) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), 246, dtype=np.uint8)
    for index in range(len(points)):
        obs = tuple(np.round(observed[index]).astype(int))
        pred = tuple(np.round(predicted[index]).astype(int))
        if injected[index]:
            color = (45, 45, 210)
        elif accepted[index]:
            color = (185, 115, 20)
        else:
            color = (65, 150, 180)
        cv2.line(canvas, pred, obs, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pred, 4, (45, 170, 125), 1, cv2.LINE_AA)
        cv2.circle(canvas, obs, 3, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "circle: observed   ring: projected   red: injected", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 50, 65), 2)
    return canvas


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(47)
    clean_count = 160
    outlier_count = 25
    truth_rvec = np.radians(np.array([1.2, -4.5, 2.0]))
    truth_tvec = np.array([0.28, -0.06, 0.18])
    points_clean = np.column_stack(
        [
            rng.uniform(-2.8, 2.8, clean_count),
            rng.uniform(-1.7, 1.7, clean_count),
            rng.uniform(4.0, 11.0, clean_count),
        ]
    )
    pixels_clean = project(points_clean, truth_rvec, truth_tvec)
    pixels_clean += rng.normal(0.0, 0.35, pixels_clean.shape)
    source_indices = rng.choice(clean_count, outlier_count, replace=False)
    points = np.vstack([points_clean, points_clean[source_indices]])
    pixels = np.vstack(
        [
            pixels_clean,
            np.column_stack(
                [
                    rng.uniform(80.0, WIDTH - 80.0, outlier_count),
                    rng.uniform(70.0, HEIGHT - 70.0, outlier_count),
                ]
            ),
        ]
    )
    labels = np.array(["clean"] * clean_count + ["injected_outlier"] * outlier_count)
    injected = labels == "injected_outlier"

    rvec, tvec, accepted = solve(points, pixels)
    predicted = project(points, rvec, tvec)
    residuals = np.linalg.norm(predicted - pixels, axis=1)
    clean_accepted = int(np.sum(accepted & ~injected))
    outliers_rejected = int(np.sum(~accepted & injected))
    trusted_condition = pose_condition(points[accepted], rvec, tvec)

    clustered_points = np.column_stack(
        [
            rng.uniform(-0.12, 0.12, clean_count),
            rng.uniform(-0.08, 0.08, clean_count),
            rng.uniform(5.5, 8.5, clean_count),
        ]
    )
    clustered_pixels = project(clustered_points, truth_rvec, truth_tvec)
    clustered_pixels += rng.normal(0.0, 0.65, clustered_pixels.shape)
    clustered_rvec, clustered_tvec, clustered_mask = solve(clustered_points, clustered_pixels)
    clustered_condition = pose_condition(
        clustered_points[clustered_mask], clustered_rvec, clustered_tvec
    )

    rotation_error = rotation_error_deg(rvec, truth_rvec)
    center_error = float(
        np.linalg.norm(camera_center(rvec, tvec) - camera_center(truth_rvec, truth_tvec))
    )
    clustered_rotation_error = rotation_error_deg(clustered_rvec, truth_rvec)
    clustered_center_error = float(
        np.linalg.norm(
            camera_center(clustered_rvec, clustered_tvec)
            - camera_center(truth_rvec, truth_tvec)
        )
    )
    accepted_residuals = residuals[accepted]
    report = {
        "correspondences": int(len(points)),
        "clean_correspondences": clean_count,
        "outliers_injected": outlier_count,
        "pnp_inliers": int(np.sum(accepted)),
        "clean_correspondences_accepted": clean_accepted,
        "injected_outliers_rejected": outliers_rejected,
        "rotation_error_deg": rotation_error,
        "camera_center_error_m": center_error,
        "median_reprojection_px": float(np.median(accepted_residuals)),
        "p95_reprojection_px": float(np.percentile(accepted_residuals, 95)),
        "accepted_grid_coverage": coverage(pixels, accepted),
        "pose_jtj_condition": trusted_condition,
        "clustered_landmarks": {
            "pnp_inliers": int(np.sum(clustered_mask)),
            "grid_coverage": coverage(clustered_pixels, clustered_mask),
            "rotation_error_deg": clustered_rotation_error,
            "camera_center_error_m": clustered_center_error,
            "pose_jtj_condition": clustered_condition,
        },
        "convention": "x_camera = R_camera_map @ x_map + t_camera_map; camera center in map is -R.T @ t",
        "artifacts": {
            "correspondences": "out/pnp_correspondences.csv",
            "model": "out/pnp_model.npz",
            "overlay": "out/pnp_reprojection_overlay.png",
        },
        "command": "python3 scripts/pnp_smoke.py",
    }
    checks = {
        "clean_support": clean_accepted >= 150,
        "outlier_rejection": outliers_rejected >= 23,
        "rotation_accuracy": rotation_error < 0.5,
        "metric_center_accuracy": center_error < 0.05,
        "trusted_reprojection": report["p95_reprojection_px"] < 1.2,
        "trusted_coverage": report["accepted_grid_coverage"] >= 0.75,
        "clustered_coverage_rejected": report["clustered_landmarks"]["grid_coverage"] <= 0.25,
        "clustered_condition_worse": clustered_condition > trusted_condition * 20.0,
    }
    report["checks"] = checks
    report["status"] = "PASS" if all(checks.values()) else "FAIL"

    with (OUT / "pnp_correspondences.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "label",
                "x_map",
                "y_map",
                "z_map",
                "u_observed",
                "v_observed",
                "u_projected",
                "v_projected",
                "pnp_inlier",
                "reprojection_px",
            ]
        )
        for index in range(len(points)):
            writer.writerow(
                [
                    index,
                    labels[index],
                    *points[index],
                    *pixels[index],
                    *predicted[index],
                    int(accepted[index]),
                    residuals[index],
                ]
            )
    np.savez(
        OUT / "pnp_model.npz",
        K=K,
        rvec_camera_map=rvec,
        tvec_camera_map=tvec,
        camera_center_map=camera_center(rvec, tvec),
        rvec_truth=truth_rvec,
        tvec_truth=truth_tvec,
        accepted=accepted,
        clustered_rvec=clustered_rvec,
        clustered_tvec=clustered_tvec,
    )
    cv2.imwrite(
        str(OUT / "pnp_reprojection_overlay.png"),
        overlay(points, pixels, predicted, accepted, injected),
    )
    (OUT / "pnp_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PnP lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
