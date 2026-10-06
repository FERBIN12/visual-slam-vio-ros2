#!/usr/bin/env python3
"""A calibration lab that actually calibrates (replaces calibration_smoke.py, which compared its
observations to themselves and reported hardcoded holdout numbers).

Synthetic 9x6 checkerboard, 25 mm squares, seen from N random poses by a camera with KNOWN
intrinsics and distortion. Corners get 0.3 px Gaussian noise. cv2.calibrateCamera estimates the
camera from the training views; held-out views are scored with solvePnP using the ESTIMATED camera.
Failure case: three nearly fronto-parallel views, which cannot pin down the focal length.
"""
import json
from pathlib import Path
import cv2
import numpy as np

K_TRUE = np.array([[458.2, 0, 319.6], [0, 457.9, 241.1], [0, 0, 1]])
D_TRUE = np.array([-0.19, 0.035, 0.0012, -0.0008, 0.0])
SIZE = (640, 480); SQ = 0.025; NOISE = 0.3
GRID = np.array([[x * SQ, y * SQ, 0.0] for y in range(6) for x in range(9)], np.float64)

def view(rng, tilt):
    """A random board pose in front of the camera, retried until every corner lands inside the image."""
    for _ in range(500):
        r = rng.uniform(-tilt, tilt, 3)
        R, _ = cv2.Rodrigues(r)
        t = np.array([rng.uniform(-.12, .02), rng.uniform(-.1, .0), rng.uniform(.45, .9)])
        px, _ = cv2.projectPoints(GRID, cv2.Rodrigues(R)[0], t, K_TRUE, D_TRUE)
        px = px.reshape(-1, 2)
        if px[:, 0].min() > 15 and px[:, 0].max() < SIZE[0] - 15 and px[:, 1].min() > 15 and px[:, 1].max() < SIZE[1] - 15:
            return px + rng.normal(0, NOISE, px.shape), R, t
    raise RuntimeError("could not place a board inside the image")

def calibrate(imgpts):
    objpts = [GRID.astype(np.float32)] * len(imgpts)
    K0 = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float64)          # deliberately wrong start
    rms, K, D, _, _ = cv2.calibrateCamera(objpts, [p.astype(np.float32).reshape(-1, 1, 2) for p in imgpts], SIZE, K0, None)
    return float(rms), K, D.ravel()

def holdout_error(K, D, views):
    errs = []
    for px, _, _ in views:
        ok, rvec, tvec = cv2.solvePnP(GRID, px, K, D)
        proj, _ = cv2.projectPoints(GRID, rvec, tvec, K, D)
        errs.append(np.linalg.norm(proj.reshape(-1, 2) - px, axis=1))
    return np.concatenate(errs)

def trial(seed, n_train, tilt, flags=0):
    rng = np.random.default_rng(seed)
    train = [view(rng, tilt) for _ in range(n_train)]; held = [view(rng, 0.5) for _ in range(6)]
    objp = [GRID.astype(np.float32)] * n_train
    K0 = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float64)
    rms, K, D, _, _ = cv2.calibrateCamera(objp, [v[0].astype(np.float32).reshape(-1, 1, 2) for v in train], SIZE, K0, None, flags=flags)
    he = holdout_error(K, D.ravel(), held)
    return {"rms": float(rms), "fx_pct": float(100 * (K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0]), "fy_pct": float(100 * (K[1, 1] - K_TRUE[1, 1]) / K_TRUE[1, 1]),
            "pp_px": float(np.hypot(K[0, 2] - K_TRUE[0, 2], K[1, 2] - K_TRUE[1, 2])), "hold": float(np.median(he)), "k": K, "d": D.ravel(), "train": train, "held": held}

def main():
    SEEDS = range(100, 112)
    table = {}
    for name, n, tilt in (("flat_3_views", 3, 0.02), ("5_views", 5, 0.5), ("20_views", 20, 0.5), ("60_views", 60, 0.5)):
        rs = [trial(sd, n, tilt) for sd in SEEDS]
        table[name] = {"views": n, "tilt_rad": tilt, "trials": len(rs),
                       "rms_px_median": float(np.median([r["rms"] for r in rs])),
                       "holdout_px_median": float(np.median([r["hold"] for r in rs])),
                       "abs_fx_error_pct_median": float(np.median([abs(r["fx_pct"]) for r in rs])),
                       "abs_fx_error_pct_p90": float(np.percentile([abs(r["fx_pct"]) for r in rs], 90)),
                       "principal_point_error_px_median": float(np.median([r["pp_px"] for r in rs]))}
    ex = trial(100, 20, 0.5)                          # one concrete 20-view calibration, shown in detail
    report = {"noise_px": NOISE, "board": "9x6 inner corners, 25 mm squares", "true_K": K_TRUE.tolist(), "true_dist": D_TRUE.tolist(),
              "summary": table,
              "example_20_views": {"K": ex["k"].tolist(), "dist": ex["d"].tolist(), "rms_px": ex["rms"], "holdout_median_px": ex["hold"],
                                   "fx_error_pct": ex["fx_pct"], "principal_point_error_px": ex["pp_px"]},
              "status": "PASS"}
    t = table
    # What the lab must show, stated as checks that can fail:
    assert t["20_views"]["holdout_px_median"] < 0.55, "good calibration should reproject near the noise floor"
    assert t["flat_3_views"]["holdout_px_median"] > 2 * t["20_views"]["holdout_px_median"] or t["flat_3_views"]["abs_fx_error_pct_median"] > 3 * t["20_views"]["abs_fx_error_pct_median"], "flat views must be visibly worse"
    assert t["5_views"]["abs_fx_error_pct_median"] > t["60_views"]["abs_fx_error_pct_median"], "more views must tighten the intrinsics"
    assert t["20_views"]["abs_fx_error_pct_median"] > 0.25, "the trap: intrinsics stay measurably off even when reprojection is at the noise floor"
    Path("out").mkdir(exist_ok=True)
    Path("out/calibration_real_report.json").write_text(json.dumps(report, indent=2) + "\n")
    flat = trial(100, 3, 0.02)
    json.dump({"train": [v[0].tolist() for v in ex["train"]], "held": [v[0].tolist() for v in ex["held"]], "flat": [v[0].tolist() for v in flat["train"]]}, open("out/calibration_views.json", "w"))
    print("calibration lab: PASS"); print(json.dumps({"summary": table, "example": report["example_20_views"]}, indent=2))

if __name__ == "__main__":
    main()
