#!/usr/bin/env python3
"""A trajectory-evaluation lab that really evaluates (replaces ate_rpe_contract.py / trajectory_alignment_contract.py,
which hardcoded the verdict).

Ground truth: a synthetic SE(3) path, 40 s, 200 Hz, about 100 m long (T_world_cam, 4x4).
Estimate: 10 Hz keyframe poses (5 % dropped) built by chaining the true relative motions with noise and a small
systematic yaw bias, so the error grows with distance ("drift"). Every other estimate is that SAME drifted estimate with
exactly one imperfection added:
  drift        the base case, nothing else
  offset       estimate expressed in another origin (constant rigid transform, 40 deg yaw, 12 m)
  scale        monocular-style scale error: positions x 0.7
  time_offset  estimate clock 0.3 s ahead of the ground-truth clock
  swapped      poses stored as T_cam_world instead of T_world_cam (inverse)
  lever_arm    a constant 0.3 m body-frame extrinsic error (T_wc * T_err): NOT a rigid world transform
  jitter       no drift at all, but independent 2 cm / 0.004 rad noise on every pose
Own code: timestamp association, Umeyama (SE3 and Sim3), ATE RMSE (none / SE3 / Sim3), RPE over a fixed delta.
Cross-checks: scipy Rotation.align_vectors (Kabsch), a plain SVD Kabsch, a scipy least_squares fit of (s, R, t), and the
`evo` package when it is importable (the report says whether it was).
Run from demo/:  python3 labs/trajectory_eval_real.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

T_END, HZ_GT, HZ_EST, DROP = 40.0, 200, 10, 0.05
TIME_OFFSET = 0.3; SCALE = 0.7; RPE_DELTA_S = 1.0
YAW_BIAS, ROT_NOISE, TR_NOISE = 0.00005, 0.0003, 0.002      # drift model, per 0.1 s step
JIT_T, JIT_R = 0.02, 0.004

# ------------------------------------------------------------------ ground truth
def gt_pose(t):
    t = np.atleast_1d(t)
    p = np.c_[30 * np.sin(0.12 * t), 20 * np.sin(0.2 * t + 0.6) - 20 * np.sin(0.6), 2 * np.sin(0.3 * t) + 0.1 * t]
    v = np.c_[30 * 0.12 * np.cos(0.12 * t), 20 * 0.2 * np.cos(0.2 * t + 0.6), 2 * 0.3 * np.cos(0.3 * t) + 0.1]
    yaw = np.arctan2(v[:, 1], v[:, 0]); roll = 0.10 * np.sin(0.5 * t); pitch = 0.08 * np.sin(0.37 * t)
    R = Rot.from_euler("ZYX", np.c_[yaw, pitch, roll]).as_matrix()
    T = np.tile(np.eye(4), (len(t), 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = p
    return T

def inv(T):
    Ti = np.empty_like(T); Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    Ti[..., :3, :3] = Rt; Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3]); Ti[..., 3, :] = [0, 0, 0, 1]
    return Ti

def drifted_estimate(G, rng):
    """Chain the true relative motions with noise + a constant yaw bias per step."""
    E = np.empty_like(G); E[0] = G[0]
    for k in range(len(G) - 1):
        rel = inv(G[k]) @ G[k + 1]
        d = np.eye(4); d[:3, :3] = Rot.from_rotvec(rng.normal(0, ROT_NOISE, 3) + [0, 0, YAW_BIAS]).as_matrix(); d[:3, 3] = rng.normal(0, TR_NOISE, 3)
        E[k + 1] = E[k] @ rel @ d
    return E

# ------------------------------------------------------------------ evaluation code under test (own implementation)
def associate(t_gt, t_est, max_diff=0.005):
    """Nearest ground-truth stamp for every estimate stamp; pairs further than max_diff are dropped (like TUM associate.py)."""
    j = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    j = np.where(np.abs(t_gt[j - 1] - t_est) < np.abs(t_gt[j] - t_est), j - 1, j)
    ok = np.abs(t_gt[j] - t_est) <= max_diff
    return np.flatnonzero(ok), j[ok]

def umeyama(src, dst, with_scale):
    """Least-squares s, R, t with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    Sg = np.eye(3); Sg[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ Sg @ Vt
    s = float(np.trace(np.diag(sig) @ Sg) / S.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s

def align(src, dst, mode):
    if mode == "none": return src.copy(), (1.0, np.eye(3), np.zeros(3))
    s, R, t = umeyama(src, dst, mode == "sim3"); return (s * (R @ src.T)).T + t, (s, R, t)

def ate_rmse(est_p, gt_p, mode):
    a, par = align(est_p, gt_p, mode)
    e = np.linalg.norm(a - gt_p, axis=1); return float(np.sqrt(np.mean(e ** 2))), e, par

def rpe(E, G, delta):
    """Translation (m) and rotation (deg) error of relative motions over `delta` index steps. No alignment needed."""
    Er = inv(E[:-delta]) @ E[delta:]; Gr = inv(G[:-delta]) @ G[delta:]
    Err = inv(Gr) @ Er
    tr = np.linalg.norm(Err[:, :3, 3], axis=1)
    ang = np.degrees(np.arccos(np.clip((np.trace(Err[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return float(np.sqrt(np.mean(tr ** 2))), float(np.sqrt(np.mean(ang ** 2))), tr

def evaluate(name, T_est, t_est_stamp, T_gt_dense, t_gt, with_rpe=True):
    ie, ig = associate(t_gt, t_est_stamp)
    E, G = T_est[ie], T_gt_dense[ig]
    out = {"name": name, "pairs": int(len(ie))}
    errs = {}
    for m in ("none", "se3", "sim3"):
        v, e, par = ate_rmse(E[:, :3, 3], G[:, :3, 3], m); out[f"ate_{m}_m"] = v; errs[m] = e
        if m == "sim3": out["sim3_scale"] = par[0]
    if with_rpe:
        n = int(round(RPE_DELTA_S * HZ_EST))                         # est stamps are 0.1 s apart except at dropped frames
        r = rpe_by_time(E, G, np.asarray(t_est_stamp)[ie], RPE_DELTA_S)
        out["rpe_trans_m"], out["rpe_rot_deg"] = r[0], r[1]; out["_rpe_series"] = r[2]
    out["_ate_series"] = errs; out["_t"] = np.asarray(t_est_stamp)[ie]
    return out

def rpe_by_time(E, G, t, delta_s, tol=0.02):
    """Pair pose i with the pose whose stamp is closest to t_i + delta_s (within tol), so dropped frames cannot corrupt the delta."""
    j = np.searchsorted(t, t + delta_s - tol); ok = (j < len(t))
    i = np.flatnonzero(ok); j = j[ok]; good = np.abs(t[j] - t[i] - delta_s) <= tol; i, j = i[good], j[good]
    Er = inv(E[i]) @ E[j]; Gr = inv(G[i]) @ G[j]; Err = inv(Gr) @ Er
    tr = np.linalg.norm(Err[:, :3, 3], axis=1)
    ang = np.degrees(np.arccos(np.clip((np.trace(Err[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return float(np.sqrt(np.mean(tr ** 2))), float(np.sqrt(np.mean(ang ** 2))), tr

# ------------------------------------------------------------------ independent computations
def kabsch_svd(src, dst):
    """Plain textbook Kabsch, written separately from umeyama(): covariance, SVD, reflection fix."""
    a, b = src - src.mean(0), dst - dst.mean(0); H = a.T @ b; U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, dst.mean(0) - R @ src.mean(0)

def scipy_kabsch(src, dst):
    R, _ = Rot.align_vectors(dst - dst.mean(0), src - src.mean(0)); R = R.as_matrix(); return R, dst.mean(0) - R @ src.mean(0)

def lsq_sim3(src, dst, with_scale):
    def fun(v):
        s = v[6] if with_scale else 1.0
        return ((s * (Rot.from_rotvec(v[:3]).as_matrix() @ src.T)).T + v[3:6] - dst).ravel()
    x0 = np.r_[Rot.from_matrix(kabsch_svd(src, dst)[0]).as_rotvec() * 0.0, np.zeros(3), 1.0]   # start from identity, not from the answer
    x0[3:6] = dst.mean(0) - src.mean(0)
    r = least_squares(fun, x0, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return (r.x[6] if with_scale else 1.0), Rot.from_rotvec(r.x[:3]).as_matrix(), r.x[3:6], float(np.sqrt(np.mean(r.fun ** 2) * 3))

def try_evo(G_pos, E_pos, ts_g, ts_e):
    try:
        from evo.core.trajectory import PosePath3D, PoseTrajectory3D
        from evo.core import metrics, sync
    except Exception as ex:
        return {"available": False, "reason": f"{type(ex).__name__}: {ex}"}
    return {"available": True, "reason": "module present"}

def main():
    out = Path("out"); out.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    t_gt = np.arange(0, T_END + 1e-9, 1 / HZ_GT); T_gt = gt_pose(t_gt)
    t_full = np.arange(0, T_END + 1e-9, 1 / HZ_EST); G10 = gt_pose(t_full)
    seg = np.linalg.norm(np.diff(T_gt[:, :3, 3], axis=0), axis=1)
    path_len = float(seg.sum()); speed = seg * HZ_GT
    base = drifted_estimate(G10, rng)
    keep = np.flatnonzero(rng.random(len(t_full)) > DROP); keep[0] = 0
    stamps = t_full + rng.normal(0, 0.0005, len(t_full))

    # ---- variants (each = the drifted base + one imperfection), all subsampled with the same dropped frames
    off = np.eye(4); off[:3, :3] = Rot.from_euler("ZYX", [np.radians(40), np.radians(5), np.radians(-3)]).as_matrix(); off[:3, 3] = [12, -7, 3]
    scale = base.copy(); scale[:, :3, 3] *= SCALE
    lever = np.eye(4); lever[:3, 3] = [0.3, 0.0, 0.1]
    jit = G10.copy()
    for k in range(len(jit)):
        d = np.eye(4); d[:3, :3] = Rot.from_rotvec(rng.normal(0, JIT_R, 3)).as_matrix(); d[:3, 3] = rng.normal(0, JIT_T, 3); jit[k] = G10[k] @ d
    V = {"drift": (base, stamps), "offset": (off @ base, stamps), "scale": (scale, stamps), "time_offset": (base, stamps + TIME_OFFSET),
         "swapped": (inv(base), stamps), "lever_arm": (base @ lever, stamps), "jitter": (jit, stamps)}
    R_ = {k: evaluate(k, v[0][keep], v[1][keep], T_gt, t_gt) for k, v in V.items()}
    # the fixes: invert back; subtract the known offset
    R_["swapped_fixed"] = evaluate("swapped_fixed", inv(V["swapped"][0])[keep], stamps[keep], T_gt, t_gt)
    R_["time_offset_fixed"] = evaluate("time_offset_fixed", base[keep], stamps[keep] + TIME_OFFSET - TIME_OFFSET, T_gt, t_gt)

    # ---- scan of the assumed time offset (what an offset search does): ATE(SE3) vs assumed offset
    scan = []
    for off_s in np.round(np.arange(-0.40, 0.4001, 0.01), 3):
        r = evaluate("scan", base[keep], stamps[keep] + TIME_OFFSET - off_s, T_gt, t_gt, with_rpe=False)
        scan.append([off_s, r["ate_se3_m"], r["pairs"]])
    scan = np.array(scan)
    best_off = float(scan[np.argmin(scan[:, 1]), 0])

    # ---- cross-checks of the Umeyama / ATE code on the clean-association pairs of the drift case and of the scale case
    chk = {}
    for nm in ("drift", "scale"):
        ie, ig = associate(t_gt, V[nm][1][keep]); P = V[nm][0][keep][ie][:, :3, 3]; Q = T_gt[ig][:, :3, 3]
        s1, R1, t1 = umeyama(P, Q, False); Rk, tk = kabsch_svd(P, Q); Rs, ts = scipy_kabsch(P, Q)
        ls_s, ls_R, ls_t, ls_rm = lsq_sim3(P, Q, False)
        s3, R3, t3 = umeyama(P, Q, True); l3s, l3R, l3t, l3rm = lsq_sim3(P, Q, True)
        chk[nm] = {"se3_R_vs_svd_kabsch_max_abs": float(np.abs(R1 - Rk).max()), "se3_R_vs_scipy_align_vectors_max_abs": float(np.abs(R1 - Rs).max()),
                   "se3_t_vs_svd_kabsch_max_abs": float(np.abs(t1 - tk).max()),
                   "se3_ate_own": R_[nm]["ate_se3_m"], "se3_ate_lsq": ls_rm, "se3_ate_rel_gap": abs(R_[nm]["ate_se3_m"] - ls_rm) / ls_rm,
                   "sim3_scale_own": s3, "sim3_scale_lsq": l3s, "sim3_ate_own": R_[nm]["ate_sim3_m"], "sim3_ate_lsq": l3rm,
                   "sim3_ate_rel_gap": abs(R_[nm]["ate_sim3_m"] - l3rm) / l3rm}
    # control: known transform, no noise, recovers exactly (tests the code, not the data)
    Pk = rng.normal(size=(50, 3)) * 5; Rt = Rot.from_euler("ZYX", [0.7, -0.3, 0.2]).as_matrix(); st, tt = 1.7, np.array([3., -2, 1])
    s_, R_k, t_k = umeyama(Pk, (st * (Rt @ Pk.T)).T + tt, True)
    known = {"scale_err": abs(s_ - st), "R_err": float(np.abs(R_k - Rt).max()), "t_err": float(np.abs(t_k - tt).max())}
    # control: Umeyama must NOT be fooled by a reflection (det must be +1)
    s_r, R_r, _ = umeyama(Pk, Pk * np.array([1, 1, -1]), False)
    known["reflection_det"] = float(np.linalg.det(R_r))

    # ---- evo, if available
    ev = try_evo(None, None, None, None)
    if ev["available"]:
        from evo.core.trajectory import PosePath3D
        from evo.core import metrics
        ie, ig = associate(t_gt, V["drift"][1][keep]); E_ = V["drift"][0][keep][ie]; G_ = T_gt[ig]
        pe, pg = PosePath3D(poses_se3=list(E_)), PosePath3D(poses_se3=list(G_))
        pe.align(pg, correct_scale=False)
        m = metrics.APE(metrics.PoseRelation.translation_part); m.process_data((pg, pe))
        ev["ate_se3_evo"] = float(m.get_statistic(metrics.StatisticsType.rmse)); ev["ate_se3_own"] = R_["drift"]["ate_se3_m"]
        pe2 = PosePath3D(poses_se3=list(V["scale"][0][keep][ie])); pe2.align(pg, correct_scale=True)
        m2 = metrics.APE(metrics.PoseRelation.translation_part); m2.process_data((pg, pe2))
        ev["ate_sim3_evo_scale_case"] = float(m2.get_statistic(metrics.StatisticsType.rmse)); ev["ate_sim3_own_scale_case"] = R_["scale"]["ate_sim3_m"]
    # ---- evidence files
    names = list(R_.keys())
    with open(out / "trajectory_eval_table.csv", "w") as f:
        f.write("variant,pairs,ate_none_m,ate_se3_m,ate_sim3_m,sim3_scale,rpe_trans_m,rpe_rot_deg\n")
        for n in names:
            r = R_[n]; f.write(f"{n},{r['pairs']},{r['ate_none_m']:.6f},{r['ate_se3_m']:.6f},{r['ate_sim3_m']:.6f},{r['sim3_scale']:.6f},{r['rpe_trans_m']:.6f},{r['rpe_rot_deg']:.6f}\n")
    np.savetxt(out / "trajectory_eval_offset_scan.csv", scan, delimiter=",", fmt="%.6f", header="assumed_offset_s,ate_se3_m,pairs", comments="")
    ie, ig = associate(t_gt, stamps[keep]); tt_ = stamps[keep][ie]
    cols = [tt_, T_gt[ig][:, :3, 3]]; hdr = ["t", "gt_x", "gt_y", "gt_z"]
    for n in ("drift", "offset", "scale", "swapped", "jitter", "lever_arm"):
        cols.append(V[n][0][keep][ie][:, :3, 3]); hdr += [f"{n}_x", f"{n}_y", f"{n}_z"]
    np.savetxt(out / "trajectory_eval_positions.csv", np.column_stack(cols), delimiter=",", fmt="%.5f", header=",".join(hdr), comments="")
    # per-pose ATE(SE3) error and per-pair RPE series for the drift-versus-jitter comparison
    ser = [R_["drift"]["_t"], R_["drift"]["_ate_series"]["se3"], R_["jitter"]["_ate_series"]["se3"]]
    np.savetxt(out / "trajectory_eval_ate_series.csv", np.column_stack(ser), delimiter=",", fmt="%.6f", header="t,drift_ate_se3_err_m,jitter_ate_se3_err_m", comments="")
    np.savetxt(out / "trajectory_eval_speed.csv", np.c_[t_gt[1:], speed], delimiter=",", fmt="%.5f", header="t,speed_m_s", comments="")

    summary = {n: {k: v for k, v in R_[n].items() if not k.startswith("_") and k != "name"} for n in names}
    report = {"ground_truth": {"duration_s": T_END, "rate_hz": HZ_GT, "path_length_m": path_len, "speed_mean_m_s": float(speed.mean()), "speed_max_m_s": float(speed.max())},
              "estimate": {"rate_hz": HZ_EST, "frames_kept": int(len(keep)), "frames_dropped": int(len(t_full) - len(keep)), "yaw_bias_rad_per_step": YAW_BIAS,
                           "rot_noise_rad": ROT_NOISE, "trans_noise_m": TR_NOISE},
              "imperfections": {"time_offset_s": TIME_OFFSET, "scale": SCALE, "frame_offset": "yaw 40 deg, pitch 5, roll -3, translation [12,-7,3] m",
                                "lever_arm_m": [0.3, 0, 0.1], "jitter_m_rad": [JIT_T, JIT_R], "rpe_delta_s": RPE_DELTA_S},
              "results": summary, "offset_scan_best_s": best_off, "cross_checks": chk, "known_transform_control": known, "evo": ev, "status": "PASS"}

    # ---- assertions that can fail
    for nm, c in chk.items():
        assert c["se3_R_vs_svd_kabsch_max_abs"] < 1e-9 and c["se3_R_vs_scipy_align_vectors_max_abs"] < 1e-6, f"{nm}: own Umeyama rotation must match Kabsch/scipy"
        assert c["se3_ate_rel_gap"] < 1e-6 and c["sim3_ate_rel_gap"] < 1e-6, f"{nm}: own ATE must match an independent least-squares fit"
        assert abs(c["sim3_scale_own"] - c["sim3_scale_lsq"]) < 1e-6, f"{nm}: own Sim3 scale must match the optimiser"
    assert known["scale_err"] < 1e-10 and known["R_err"] < 1e-10 and known["t_err"] < 1e-9, "Umeyama must recover a known noiseless Sim3 exactly"
    assert abs(known["reflection_det"] - 1) < 1e-9, "alignment must return a proper rotation"
    if ev["available"]:
        assert abs(ev["ate_se3_evo"] - ev["ate_se3_own"]) < 1e-6 and abs(ev["ate_sim3_evo_scale_case"] - ev["ate_sim3_own_scale_case"]) < 1e-6, "own ATE must match evo"
    s = summary
    d = s["drift"]
    assert 0.03 < d["ate_se3_m"] < 0.6, "the base drift must be small but measurable"
    assert s["offset"]["ate_none_m"] > 20 * d["ate_se3_m"], "an unaligned offset must dominate ATE"
    assert abs(s["offset"]["ate_se3_m"] - d["ate_se3_m"]) < 1e-6 * max(1, d["ate_se3_m"]) + 1e-9, "SE3 alignment must absorb the constant offset exactly"
    assert abs(s["offset"]["rpe_trans_m"] - d["rpe_trans_m"]) < 1e-9, "RPE must be blind to a constant offset"
    assert s["scale"]["ate_se3_m"] > 10 * d["ate_se3_m"], "SE3 alignment must expose a scale error"
    assert abs(s["scale"]["ate_sim3_m"] - d["ate_sim3_m"]) < 1e-6 * max(1, d["ate_sim3_m"]) + 1e-9, "Sim3 alignment must absorb the scale error"
    assert abs(s["scale"]["sim3_scale"] * SCALE - 1) < 0.05, "Sim3 must recover the inverse of the injected scale"
    assert s["time_offset"]["ate_se3_m"] > 3 * d["ate_se3_m"], "a time offset must not be hidden by SE3 alignment"
    assert s["time_offset"]["ate_sim3_m"] > 3 * d["ate_sim3_m"], "a time offset must not be hidden by Sim3 alignment"
    assert abs(best_off - TIME_OFFSET) < 0.05, "an offset scan must land near the injected offset (drift biases it slightly)"
    assert abs(s["time_offset_fixed"]["ate_se3_m"] - d["ate_se3_m"]) < 1e-6, "removing the offset must restore the drift-only ATE"
    assert s["swapped"]["ate_se3_m"] > 10 * d["ate_se3_m"] and s["swapped"]["ate_sim3_m"] > 10 * d["ate_sim3_m"], "a swapped convention must not be hidden by either alignment"
    assert abs(s["swapped_fixed"]["ate_se3_m"] - d["ate_se3_m"]) < 1e-9, "inverting the poses back must restore the drift-only ATE"
    assert s["lever_arm"]["ate_se3_m"] > 1.5 * d["ate_se3_m"], "a body-frame extrinsic error is not a rigid world transform and must survive alignment"
    assert d["ate_se3_m"] > s["jitter"]["ate_se3_m"] and s["jitter"]["rpe_trans_m"] > 3 * d["rpe_trans_m"], "ATE ranks drift worse, RPE ranks jitter worse"
    (out / "trajectory_eval_real_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("trajectory eval lab: PASS"); print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
