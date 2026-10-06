#!/usr/bin/env python3
"""Deterministic relative-pose lab with outliers and a low-parallax control."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
K = np.array([[720.0, 0.0, 640.0], [0.0, 715.0, 360.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1280, 720)


def project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    camera = (rotation @ points.T + translation.reshape(3, 1)).T
    pixels = (K @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:3]


def rotation_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    delta = estimate @ truth.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def direction_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    a = estimate.reshape(3) / np.linalg.norm(estimate)
    b = truth.reshape(3) / np.linalg.norm(truth)
    cosine = np.clip(float(a @ b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def parallax_deg(
    pixels_1: np.ndarray, pixels_2: np.ndarray, rotation_21: np.ndarray
) -> np.ndarray:
    k_inv = np.linalg.inv(K)
    rays_1 = (k_inv @ np.c_[pixels_1, np.ones(len(pixels_1))].T).T
    rays_2 = (k_inv @ np.c_[pixels_2, np.ones(len(pixels_2))].T).T
    rays_2_in_1 = (rotation_21.T @ rays_2.T).T
    rays_1 /= np.linalg.norm(rays_1, axis=1, keepdims=True)
    rays_2_in_1 /= np.linalg.norm(rays_2_in_1, axis=1, keepdims=True)
    cosine = np.clip(np.sum(rays_1 * rays_2_in_1, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def sampson_px(essential: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    fundamental = np.linalg.inv(K).T @ essential @ np.linalg.inv(K)
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    fx1 = (fundamental @ x1.T).T
    ftx2 = (fundamental.T @ x2.T).T
    numerator = np.sum(x2 * fx1, axis=1) ** 2
    denominator = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def estimate(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    essential, ransac_mask = cv2.findEssentialMat(
        p1,
        p2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
        maxIters=4000,
    )
    if essential is None or essential.shape != (3, 3):
        raise RuntimeError(f"expected one 3x3 essential matrix, got {None if essential is None else essential.shape}")
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential, p1, p2, K, mask=ransac_mask.copy()
    )
    return essential, rotation, translation.reshape(3), ransac_mask.reshape(-1).astype(bool) & pose_mask.reshape(-1).astype(bool)


def make_overlay(
    p1: np.ndarray, p2: np.ndarray, accepted: np.ndarray, injected: np.ndarray
) -> np.ndarray:
    width, height = IMAGE_SIZE
    scale = 0.5
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, "FRAME 1", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 50, 70), 2)
    cv2.putText(canvas, "FRAME 2", (width // 2 + 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 50, 70), 2)
    cv2.line(canvas, (width // 2, 0), (width // 2, height), (190, 195, 205), 2)
    for index in range(len(p1)):
        left = tuple(np.round(p1[index] * scale).astype(int))
        right_local = np.round(p2[index] * scale).astype(int)
        right = (int(right_local[0] + width // 2), int(right_local[1]))
        if injected[index]:
            color = (45, 45, 210)
        elif accepted[index]:
            color = (185, 115, 20)
        else:
            color = (65, 150, 180)
        if index < 90 or injected[index]:
            cv2.line(canvas, left, right, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, left, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, right, 3, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "blue: accepted clean   amber: rejected clean   red: injected", (35, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (45, 50, 65), 2)
    return canvas


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(31)
    clean_count = 180
    outlier_count = 30
    points = np.column_stack(
        [
            rng.uniform(-2.2, 2.2, clean_count),
            rng.uniform(-1.25, 1.25, clean_count),
            rng.uniform(4.0, 10.0, clean_count),
        ]
    )
    rotation_truth, _ = cv2.Rodrigues(np.radians(np.array([0.8, 3.2, -0.6])))
    translation_truth = np.array([-0.34, 0.015, 0.025])
    p1_clean = project(points, np.eye(3), np.zeros(3))
    p2_clean = project(points, rotation_truth, translation_truth)
    p1_clean += rng.normal(0.0, 0.28, p1_clean.shape)
    p2_clean += rng.normal(0.0, 0.28, p2_clean.shape)

    injected_left = rng.choice(clean_count, outlier_count, replace=False)
    injected_right = np.roll(rng.permutation(clean_count), 17)[:outlier_count]
    p1 = np.vstack([p1_clean, p1_clean[injected_left]])
    p2 = np.vstack([p2_clean, p2_clean[injected_right]])
    labels = np.array(["clean"] * clean_count + ["injected_outlier"] * outlier_count)
    injected = labels == "injected_outlier"

    essential, rotation, translation, accepted = estimate(p1, p2)
    residuals = sampson_px(essential, p1, p2)
    clean_accepted = int(np.sum(accepted & ~injected))
    outliers_rejected = int(np.sum(~accepted & injected))
    trusted_parallax = parallax_deg(p1_clean, p2_clean, rotation_truth)

    tiny_translation = np.array([0.0, 0.0, -0.008])
    p2_tiny = project(points, np.eye(3), tiny_translation)
    p1_tiny = project(points, np.eye(3), np.zeros(3))
    p1_tiny += rng.normal(0.0, 0.28, p1_tiny.shape)
    p2_tiny += rng.normal(0.0, 0.28, p2_tiny.shape)
    tiny_essential, tiny_rotation, tiny_direction, tiny_accepted = estimate(p1_tiny, p2_tiny)
    tiny_parallax = parallax_deg(p1_tiny, p2_tiny, np.eye(3))

    rotation_error = rotation_error_deg(rotation, rotation_truth)
    translation_error = direction_error_deg(translation, translation_truth)
    tiny_translation_error = direction_error_deg(tiny_direction, tiny_translation)
    accepted_residuals = residuals[accepted]
    report = {
        "matches": int(len(p1)),
        "clean_matches": clean_count,
        "outliers_injected": outlier_count,
        "pose_inliers": int(np.sum(accepted)),
        "clean_matches_accepted": clean_accepted,
        "injected_outliers_rejected": outliers_rejected,
        "rotation_error_deg": rotation_error,
        "translation_direction_error_deg": translation_error,
        "median_sampson_px": float(np.median(accepted_residuals)),
        "p95_sampson_px": float(np.percentile(accepted_residuals, 95)),
        "median_parallax_deg": float(np.median(trusted_parallax)),
        "tiny_forward_motion": {
            "baseline_m": float(np.linalg.norm(tiny_translation)),
            "pose_inliers": int(np.sum(tiny_accepted)),
            "median_parallax_deg": float(np.median(tiny_parallax)),
            "translation_direction_error_deg": tiny_translation_error,
        },
        "convention": "x_camera2 = R_21 @ x_camera1 + t_21; translation is a direction until metric scale is supplied",
        "artifacts": {
            "matches": "out/relative_pose_matches.csv",
            "model": "out/relative_pose_model.npz",
            "overlay": "out/relative_pose_overlay.png",
        },
        "command": "python3 scripts/relative_pose_smoke.py",
    }

    checks = {
        "clean_support": clean_accepted >= 155,
        "outlier_rejection": outliers_rejected >= 27,
        "rotation_accuracy": rotation_error < 1.0,
        "translation_accuracy": translation_error < 4.0,
        "trusted_parallax": report["median_parallax_deg"] > 1.0,
        "tiny_parallax_rejected": report["tiny_forward_motion"]["median_parallax_deg"] < 0.15,
        "tiny_pose_support_failed": int(np.sum(tiny_accepted)) < 30,
    }
    report["checks"] = checks
    report["status"] = "PASS" if all(checks.values()) else "FAIL"

    with (OUT / "relative_pose_matches.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "u1", "v1", "u2", "v2", "pose_inlier", "sampson_px"])
        for index in range(len(p1)):
            writer.writerow(
                [
                    index,
                    labels[index],
                    *p1[index],
                    *p2[index],
                    int(accepted[index]),
                    residuals[index],
                ]
            )
    np.savez(
        OUT / "relative_pose_model.npz",
        K=K,
        E=essential,
        R_21=rotation,
        t_21_direction=translation,
        R_21_truth=rotation_truth,
        t_21_truth=translation_truth,
        tiny_E=tiny_essential,
        tiny_R=tiny_rotation,
        tiny_t_direction=tiny_direction,
    )
    cv2.imwrite(str(OUT / "relative_pose_overlay.png"), make_overlay(p1, p2, accepted, injected))
    (OUT / "relative_pose_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("relative pose lab:", report["status"])
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
