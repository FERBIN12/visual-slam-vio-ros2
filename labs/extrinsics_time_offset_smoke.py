#!/usr/bin/env python3
"""Deterministic camera-IMU extrinsic and time-offset calibration lab."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def exp_so3(v: np.ndarray) -> np.ndarray:
    r, _ = cv2.Rodrigues(np.asarray(v, dtype=float).reshape(3, 1)); return r


def log_so3(r: np.ndarray) -> np.ndarray:
    v, _ = cv2.Rodrigues(np.asarray(r, dtype=float)); return v.reshape(3)


def body_state(t: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float).reshape(-1)
    rotations = np.stack([exp_so3(np.array([0.16 * np.sin(.4*x) + .03*x, .12 * np.cos(.3*x) - .12, .10 * np.sin(.7*x)])) for x in t])
    positions = np.column_stack([.4*np.sin(.5*t), .25*np.cos(.4*t), .1*np.sin(.3*t)+.02*t])
    return rotations, positions


def camera_measurements(times: np.ndarray, true_offset: float, r_bc: np.ndarray, t_bc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_body, p_body = body_state(times + true_offset)
    r_cam = np.stack([r @ r_bc for r in r_body])
    p_cam = p_body + np.einsum("nij,j->ni", r_body, t_bc)
    return r_cam, p_cam


def residual(parameters: np.ndarray, times: np.ndarray, r_cam: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    r_bc, t_bc, offset = exp_so3(parameters[:3]), parameters[3:6], parameters[6]
    r_body, p_body = body_state(times + offset)
    r_hat = np.stack([r @ r_bc.T for r in r_cam])
    p_hat = p_cam - np.einsum("nij,nj->ni", r_hat, np.broadcast_to(t_bc, p_hat_shape := (len(times), 3)))
    rot_error = np.stack([log_so3(r_hat[i] @ r_body[i].T) for i in range(len(times))])
    return np.concatenate([rot_error.reshape(-1), (p_hat - p_body).reshape(-1)])


def metrics(parameters: np.ndarray, times: np.ndarray, r_cam: np.ndarray, p_cam: np.ndarray) -> dict:
    values = residual(parameters, times, r_cam, p_cam)
    n = len(times) * 3
    return {"rotation_rmse_deg": float(np.degrees(np.sqrt(np.mean(values[:n]**2)))), "translation_rmse_m": float(np.sqrt(np.mean(values[n:]**2))), "residual_rms": float(np.sqrt(np.mean(values**2)))}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    true_offset = .037
    true_r = np.array([.11, -.07, .05]); true_t = np.array([.12, -.04, .08])
    times = np.linspace(.3, 9.7, 240)
    r_cam, p_cam = camera_measurements(times, true_offset, exp_so3(true_r), true_t)
    split = 168
    initial = np.zeros(7)
    fit = least_squares(lambda x: residual(x, times[:split], r_cam[:split], p_cam[:split]), initial, bounds=([-1,-1,-1,-.5,-.5,-.5,-.2],[1,1,1,.5,.5,.5,.2]), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=300)
    train_metrics = metrics(fit.x, times[:split], r_cam[:split], p_cam[:split]); holdout_metrics = metrics(fit.x, times[split:], r_cam[split:], p_cam[split:])
    sweep_rows = []
    for offset in np.linspace(-.12, .12, 49):
        trial = np.r_[true_r, true_t, offset]
        m = metrics(trial, times[:split], r_cam[:split], p_cam[:split]); sweep_rows.append({"offset_s": offset, "residual_rms": m["residual_rms"], "translation_rmse_m": m["translation_rmse_m"], "rotation_rmse_deg": m["rotation_rmse_deg"]})
    wrong_ext = np.r_[-true_r, true_t, true_offset]
    wrong_offset = np.r_[true_r, true_t, -.08]
    weak_times = np.linspace(.3, 9.7, 80)
    weak_r = np.eye(3)[None, :].repeat(len(weak_times), axis=0); weak_p = np.column_stack([.4*weak_times, np.zeros(len(weak_times)), np.zeros(len(weak_times))])
    weak_cam_r = np.stack([r @ exp_so3(true_r) for r in weak_r]); weak_cam_p = weak_p + np.einsum("nij,j->ni", weak_r, true_t)
    def weak_residual(parameters: np.ndarray) -> np.ndarray:
        r_h = exp_so3(parameters[:3]); t_h = parameters[3:6]
        r_body_h = np.eye(3)[None, :].repeat(len(weak_times), axis=0)
        p_body_h = np.column_stack([.4 * (weak_times + parameters[6]), np.zeros(len(weak_times)), np.zeros(len(weak_times))])
        r_est = np.stack([r @ r_h.T for r in weak_cam_r])
        p_est = weak_cam_p - t_h
        return np.concatenate([np.stack([log_so3(r_est[i] @ r_body_h[i].T) for i in range(len(weak_times))]).reshape(-1), (p_est - p_body_h).reshape(-1)])
    weak_fit = least_squares(weak_residual, initial, bounds=([-1,-1,-1,-.5,-.5,-.5,-.2],[1,1,1,.5,.5,.5,.2]), max_nfev=200)
    weak_jac = np.linalg.svd(weak_fit.jac, compute_uv=False)
    weak_condition = float(weak_jac[0] / max(weak_jac[-1], 1e-15))
    sweep_min = min(sweep_rows, key=lambda row: row["residual_rms"])
    errors = {"rotation_deg": float(np.degrees(np.linalg.norm(fit.x[:3] - true_r))), "translation_m": float(np.linalg.norm(fit.x[3:6] - true_t)), "offset_ms": float(abs(fit.x[6] - true_offset) * 1000.0)}
    checks = {
        "fit_rotation": errors["rotation_deg"] < .01,
        "fit_translation": errors["translation_m"] < 1e-4,
        "fit_time_offset": errors["offset_ms"] < .05,
        "heldout_residual": holdout_metrics["translation_rmse_m"] < 1e-4 and holdout_metrics["rotation_rmse_deg"] < .01,
        "offset_sweep_minimum": abs(sweep_min["offset_s"] - true_offset) < .006,
        "wrong_extrinsic_visible": metrics(wrong_ext, times[split:], r_cam[split:], p_cam[split:])["residual_rms"] > .05,
        "wrong_offset_visible": metrics(wrong_offset, times[split:], r_cam[split:], p_cam[split:])["translation_rmse_m"] > .005,
        "weak_motion_condition_visible": weak_condition > 1e5,
    }
    checks = {k: bool(v) for k,v in checks.items()}
    report = {"convention": "T_world_camera = T_world_body T_body_camera; camera clock t maps to body state at t + offset", "truth": {"rotation_vector_body_camera": true_r.tolist(), "translation_body_camera_m": true_t.tolist(), "offset_s": true_offset}, "fit": {"rotation_vector": fit.x[:3].tolist(), "translation_m": fit.x[3:6].tolist(), "offset_s": float(fit.x[6]), "parameter_errors": errors}, "train": train_metrics, "holdout": holdout_metrics, "offset_sweep_minimum": sweep_min, "faults": {"wrong_extrinsic": metrics(wrong_ext, times[split:], r_cam[split:], p_cam[split:]), "wrong_offset": metrics(wrong_offset, times[split:], r_cam[split:], p_cam[split:]), "weak_motion_condition": weak_condition}, "command": "python3 scripts/extrinsics_time_offset_smoke.py", "artifacts": {"calibration_report": "out/extrinsics_time_offset_report.json", "offset_sweep": "out/extrinsics_offset_sweep.csv", "holdout": "out/extrinsics_holdout.csv", "model": "out/extrinsics_time_offset_model.npz", "plot": "out/extrinsics_time_offset_plot.png"}, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
    with (OUT/"extrinsics_offset_sweep.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(sweep_rows[0])); w.writeheader(); w.writerows(sweep_rows)
    holdout_rows = []
    fitted_r = exp_so3(fit.x[:3])
    for t, measured_r in zip(times[split:], r_cam[split:]):
        truth_r = body_state(np.array([t + fit.x[6]]))[0][0]
        estimate_r = measured_r @ fitted_r.T
        holdout_rows.append({"time_s": float(t), "rotation_error_deg": float(np.degrees(np.linalg.norm(log_so3(estimate_r @ truth_r.T))))})
    with (OUT/"extrinsics_holdout.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(holdout_rows[0])); w.writeheader(); w.writerows(holdout_rows)
    np.savez(OUT/"extrinsics_time_offset_model.npz", camera_times_s=times, camera_rotation=r_cam, camera_position=p_cam, fitted_parameters=fit.x, offset_sweep=np.array([[r["offset_s"],r["residual_rms"]] for r in sweep_rows]), weak_singular_values=weak_jac)
    canvas=np.full((720,1280,3),246,np.uint8); ink=(58,46,36); blue=(205,132,42); red=(45,45,205); teal=(135,145,25)
    cv2.putText(canvas,"CAMERA–IMU EXTRINSICS + TIME OFFSET",(42,50),cv2.FONT_HERSHEY_SIMPLEX,.72,ink,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(55,100),(790,635),(214,208,200),1); cv2.putText(canvas,"OFFSET SWEEP",(80,140),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA)
    vals=np.array([r["residual_rms"] for r in sweep_rows]); xs=np.linspace(90,750,len(vals)); ys=540-300*(vals-vals.min())/max(vals.max()-vals.min(),1e-12); cv2.polylines(canvas,[np.column_stack([xs,ys]).astype(np.int32)],False,blue,4,cv2.LINE_AA); best=int(np.argmin(vals)); cv2.circle(canvas,(int(xs[best]),int(ys[best])),8,teal,-1); cv2.putText(canvas,f"minimum {sweep_rows[best]['offset_s']*1000:.1f} ms",(100,595),cv2.FONT_HERSHEY_SIMPLEX,.48,teal,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(835,100),(1225,635),(214,208,200),1); cv2.putText(canvas,"CALIBRATION GATES",(860,140),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA)
    lines=[("rotation",f"{errors['rotation_deg']:.4f} deg",teal),("translation",f"{errors['translation_m']*1000:.3f} mm",teal),("offset",f"{errors['offset_ms']:.3f} ms",teal),("weak condition",f"{weak_condition:.2g}",red)]
    for i,(label,value,color) in enumerate(lines): cv2.putText(canvas,label,(870,220+i*88),cv2.FONT_HERSHEY_SIMPLEX,.48,ink,1,cv2.LINE_AA); cv2.putText(canvas,value,(870,260+i*88),cv2.FONT_HERSHEY_SIMPLEX,.62,color,2,cv2.LINE_AA)
    cv2.imwrite(str(OUT/"extrinsics_time_offset_plot.png"),canvas); (OUT/"extrinsics_time_offset_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Extrinsics/time-offset lab:",report["status"]); print(json.dumps(report,indent=2)); return 0 if report["status"]=="PASS" else 1

if __name__=="__main__": raise SystemExit(main())
