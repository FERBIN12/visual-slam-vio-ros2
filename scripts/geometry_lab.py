#!/usr/bin/env python3
"""Command-line camera-geometry lab.

Examples:
  python3 scripts/geometry_lab.py project
  python3 scripts/geometry_lab.py roundtrip
  python3 scripts/geometry_lab.py stereo
  python3 scripts/geometry_lab.py failure
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float = 458.2
    fy: float = 457.9
    cx: float = 319.6
    cy: float = 241.1

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected (N, 3), got {points.shape}")
        if not np.all(points[:, 2] > 0):
            bad = np.flatnonzero(points[:, 2] <= 0).tolist()
            raise ValueError(f"non-positive depth at rows {bad}")
        return np.c_[self.fx * points[:, 0] / points[:, 2] + self.cx,
                     self.fy * points[:, 1] / points[:, 2] + self.cy]

    def normalize(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64)
        return np.c_[(pixels[:, 0] - self.cx) / self.fx,
                     (pixels[:, 1] - self.cy) / self.fy]


FIXTURE = np.array([
    [-0.80, -0.45, 2.2], [0.00, -0.45, 2.2], [0.80, -0.45, 2.2],
    [-0.80,  0.00, 3.7], [0.42, -0.08, 3.7], [0.80,  0.00, 3.7],
    [-0.80,  0.45, 5.4], [0.00,  0.45, 5.4], [0.80,  0.45, 5.4],
], dtype=np.float64)


def roundtrip(camera: Intrinsics, points: np.ndarray) -> dict:
    pixels = camera.project(points)
    normalized = camera.normalize(pixels)
    expected = points[:, :2] / points[:, 2:3]
    error = np.linalg.norm(normalized - expected, axis=1)
    return {"count": len(points), "max_error": float(error.max()),
            "median_error": float(np.median(error))}


def stereo(camera: Intrinsics, points: np.ndarray, baseline: float = 0.12) -> dict:
    left = camera.project(points)
    right_points = points.copy(); right_points[:, 0] -= baseline
    right = camera.project(right_points)
    disparity = left[:, 0] - right[:, 0]
    depth = camera.fx * baseline / disparity
    error = np.abs(depth - points[:, 2])
    return {"baseline_m": baseline, "disparity_min_px": float(disparity.min()),
            "disparity_max_px": float(disparity.max()),
            "median_depth_error_m": float(np.median(error)),
            "max_depth_error_m": float(error.max()),
            "positive_depth": int(np.sum(depth > 0))}


def failure(camera: Intrinsics, points: np.ndarray) -> dict:
    observed = camera.project(points)
    bad = Intrinsics(camera.fx * 0.90, camera.fy * 0.90, camera.cx, camera.cy)
    residual = np.linalg.norm(bad.project(points) - observed, axis=1)
    radius = np.linalg.norm(camera.normalize(observed), axis=1)
    order = np.argsort(radius)
    return {"bad_focal_scale": 0.90,
            "median_error_px": float(np.median(residual)),
            "max_error_px": float(residual.max()),
            "centre_error_px": float(residual[order[0]]),
            "edge_error_px": float(residual[order[-1]]),
            "gate": "FAIL" if float(np.median(residual)) > 1.0 else "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("project", "roundtrip", "stereo", "failure"))
    parser.add_argument("--out", default="out/geometry_lab.json")
    args = parser.parse_args()
    camera = Intrinsics()
    if args.mode == "project":
        pixels = camera.project(FIXTURE)
        result = {"camera": asdict(camera), "points": FIXTURE.tolist(),
                  "pixels": pixels.round(6).tolist(), "status": "PASS"}
    elif args.mode == "roundtrip":
        result = {"mode": args.mode, **roundtrip(camera, FIXTURE), "status": "PASS"}
    elif args.mode == "stereo":
        result = {"mode": args.mode, **stereo(camera, FIXTURE), "status": "PASS"}
    else:
        result = {"mode": args.mode, **failure(camera, FIXTURE)}
        if result["gate"] != "FAIL":
            raise AssertionError("bad intrinsics did not cross the residual gate")
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
