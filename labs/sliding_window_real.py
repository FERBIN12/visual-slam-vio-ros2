#!/usr/bin/env python3
"""A sliding-window lab that really marginalizes (replaces sliding_window_smoke.py, which had no solver).

Planar SE(2) keyframe trajectory (x, y, heading), 30 keyframes, about 1 m apart. DESIGN CHOICE: SE(2), not SE(3).
Factors
  * odometry: relative pose between consecutive keyframes, expressed in the earlier body frame. A stand-in for
    preintegrated IMU / VO, with noise AND a systematic bias (1.5 % too long, 0.006 rad extra turn per step) so the chain drifts.
  * landmark: a KNOWN map landmark seen in the body frame (2-D vector, 0.12 m noise), only if it is within 2.2 m.
    A single landmark pins 2 of the 3 degrees of freedom of a pose; they are sparse (see report), so anchoring is weak.
  * start prior: pose 0 known to 1 cm / 0.003 rad.
Solvers
  (a) FULL BATCH over all keyframes (own Gauss-Newton; cross-checked against scipy least_squares).
  (b) WINDOW of 8 keyframes + PRIOR: before the oldest pose leaves, it is marginalized with a Gauss-Newton Schur complement
      (H_rr - H_rm H_mm^-1 H_mr, b_r - H_rm H_mm^-1 b_m, evaluated at the current estimate) and the result is a prior factor
      on the new oldest pose, centred at the linearization point.
  (c) WINDOW with the prior DROPPED: the oldest pose is simply forgotten.
Output trajectory of a window method = each pose's estimate at the moment it leaves the window (what an online system would
have produced), plus the final window.
Verification: with factors made LINEAR (rotations frozen at the true values) the window+prior solution on the retained poses
must equal the batch solution on those poses (assert, 1e-8). Control: the dropped-prior window must NOT match.
Run from demo/:  python3 labs/sliding_window_real.py
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

N, W = 30, 8
SIG_T, SIG_R = 0.03, 0.01           # odometry noise (m, rad)
BIAS_SCALE, BIAS_TURN = 1.015, 0.006
SIG_LM, LM_RANGE = 0.12, 2.2
SIG_P0 = (0.01, 0.01, 0.003)

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def rot(th): c, s = np.cos(th), np.sin(th); return np.array([[c, -s], [s, c]])

# ------------------------------------------------------------------ problem generation
def make_problem(seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((N, 3)); th = 0.0
    for k in range(1, N):
        th += 0.10 * np.sin(k / 4.0) + 0.02                      # gently weaving path
        X[k] = X[k - 1] + 1.0 * np.array([np.cos(th), np.sin(th), 0]) + np.array([0, 0, 0])
        X[k, 2] = th
    # sparse known landmarks scattered along the path
    lms = []
    for k in range(2, N, 3):
        side = rng.choice([-1, 1]); off = rng.uniform(0.8, 1.8)
        lms.append(X[k, :2] + side * off * np.array([-np.sin(X[k, 2]), np.cos(X[k, 2])]) + rng.normal(0, .2, 2))
    lms = np.array(lms)
    F = []                                                         # factor list
    F.append(dict(k="p0", i=0, z=X[0] + rng.normal(0, SIG_P0), w=1 / np.array(SIG_P0)))
    for i in range(N - 1):
        d = rot(X[i, 2]).T @ (X[i + 1, :2] - X[i, :2]); dth = wrap(X[i + 1, 2] - X[i, 2])
        z = np.r_[d * BIAS_SCALE + rng.normal(0, SIG_T, 2), dth + BIAS_TURN + rng.normal(0, SIG_R)]
        F.append(dict(k="odo", i=i, j=i + 1, z=z, w=1 / np.array([SIG_T, SIG_T, SIG_R])))
    n_obs = 0
    for i in range(N):
        for li, L in enumerate(lms):
            v = rot(X[i, 2]).T @ (L - X[i, :2])
            if np.linalg.norm(v) < LM_RANGE:
                F.append(dict(k="lm", i=i, L=L, z=v + rng.normal(0, SIG_LM, 2), w=np.full(2, 1 / SIG_LM))); n_obs += 1
    return X, lms, F, n_obs

# ------------------------------------------------------------------ factor residuals + analytic Jacobians
def res_jac(f, x, Rfix=None):
    """Residual and Jacobian blocks {pose_index: (r_dim x 3)} of one factor at state x (N x 3).
    Rfix: if given (N rotation angles), rotations used inside the position factors are frozen (=> linear factors)."""
    k = f["k"]
    if k == "p0":
        return (x[f["i"]] - f["z"]) * f["w"], {f["i"]: np.diag(f["w"])}
    ang = lambda i: x[i, 2] if Rfix is None else Rfix[i]
    if k == "lm":
        i = f["i"]; R = rot(ang(i)); dR = np.array([[-np.sin(ang(i)), np.cos(ang(i))], [-np.cos(ang(i)), -np.sin(ang(i))]])   # d(R^T)/dth
        v = f["L"] - x[i, :2]
        J = np.zeros((2, 3)); J[:, :2] = -R.T
        if Rfix is None: J[:, 2] = dR @ v
        return (R.T @ v - f["z"]) * f["w"], {i: J * f["w"][:, None]}
    i, j = f["i"], f["j"]; R = rot(ang(i)); dR = np.array([[-np.sin(ang(i)), np.cos(ang(i))], [-np.cos(ang(i)), -np.sin(ang(i))]])
    v = x[j, :2] - x[i, :2]
    r = np.r_[R.T @ v - f["z"][:2], wrap(x[j, 2] - x[i, 2] - f["z"][2])] * f["w"]
    Ji, Jj = np.zeros((3, 3)), np.zeros((3, 3))
    Ji[:2, :2] = -R.T; Jj[:2, :2] = R.T
    if Rfix is None: Ji[:2, 2] = dR @ v
    Ji[2, 2] = -1; Jj[2, 2] = 1
    return r, {i: Ji * f["w"][:, None], j: Jj * f["w"][:, None]}

def prior_res_jac(P, x, idx):
    """Prior on pose idx: energy 0.5 d^T Hp d - bp^T d with d = x_idx - x_lin. Written as 0.5||L^T d - c||^2 with Hp = L L^T, bp = L c."""
    d = x[idx] - P["xlin"]
    return P["Lt"] @ d - P["c"], {idx: P["Lt"]}

def normal_eq(indices, factors, x, prior=None, Rfix=None):
    """H, b (b = -J^T r) over the poses in `indices` (ordered list), only factors fully inside the set."""
    pos = {p: n for n, p in enumerate(indices)}; D = 3 * len(indices)
    H = np.zeros((D, D)); b = np.zeros(D); cost = 0.0
    items = [res_jac(f, x, Rfix) for f in factors if all(p in pos for p in ([f["i"]] + ([f["j"]] if f["k"] == "odo" else [])))]
    if prior is not None: items.append(prior_res_jac(prior, x, indices[0]))
    for r, Js in items:
        cost += 0.5 * r @ r
        for p, Jp in Js.items():
            b[3 * pos[p]:3 * pos[p] + 3] -= Jp.T @ r
            for q, Jq in Js.items():
                H[3 * pos[p]:3 * pos[p] + 3, 3 * pos[q]:3 * pos[q] + 3] += Jp.T @ Jq
    return H, b, cost

def gauss_newton(indices, factors, x, prior=None, Rfix=None, iters=15):
    x = x.copy()
    for _ in range(iters):
        H, b, _ = normal_eq(indices, factors, x, prior, Rfix)
        dx = np.linalg.solve(H + 1e-9 * np.eye(len(b)), b).reshape(-1, 3)
        x[indices] += dx
        if np.abs(dx).max() < 1e-10: break
    return x

# ------------------------------------------------------------------ marginalization
def marginalize_oldest(indices, factors, x, prior, Rfix=None):
    """Schur-complement the oldest pose out of the window; returns the prior on the new oldest pose.
    Only factors touching the oldest pose are used (odometry to the next pose, its landmarks, the old prior)."""
    m, r = indices[0], indices[1]
    touch = [f for f in factors if (f["k"] != "odo" and f["i"] == m) or (f["k"] == "odo" and f["i"] == m and f["j"] == r)]
    H, b, _ = normal_eq([m, r], touch, x, prior, Rfix)        # old prior is attached to indices[0] == m
    Hmm, Hmr, Hrr = H[:3, :3], H[:3, 3:], H[3:, 3:]
    Hp = Hrr - Hmr.T @ np.linalg.solve(Hmm, Hmr)
    bp = b[3:] - Hmr.T @ np.linalg.solve(Hmm, b[:3])
    Hp = 0.5 * (Hp + Hp.T)
    w, V = np.linalg.eigh(Hp); w = np.clip(w, 0, None)
    L = V * np.sqrt(w)                                           # Hp = L L^T
    c = np.linalg.pinv(L) @ bp                                   # L c = bp (Hp full rank here)
    return dict(xlin=x[r].copy(), Lt=L.T, c=c, Hp=Hp, bp=bp)

def run_window(factors, x0, keep_prior, Rfix=None, w=W):
    """Online sliding window. x0: initial guess for all poses (only used for first pose; later poses chain from estimates)."""
    x = x0.copy(); out = np.full_like(x0, np.nan); prior = None; idx = [0]; t_solves = []
    for k in range(N):
        if k > 0:                                                # new pose enters, initialised by chaining odometry on the current estimate
            f = next(f for f in factors if f["k"] == "odo" and f["j"] == k)
            i = k - 1; x[k, :2] = x[i, :2] + rot(x[i, 2]) @ f["z"][:2]; x[k, 2] = x[i, 2] + f["z"][2]
            idx.append(k)
        t0 = time.perf_counter()
        # without a prior the window floats: add a negligible gauge anchor so the normal equations are solvable
        gauge = None
        if not keep_prior and idx[0] > 0:
            gauge = dict(xlin=x[idx[0]].copy(), Lt=np.eye(3) * 1e-4, c=np.zeros(3))
        x = gauss_newton(idx, factors, x, prior if keep_prior else gauge, Rfix)
        t_solves.append(time.perf_counter() - t0)
        if len(idx) == w and k < N - 1:                          # about to exceed the window: retire the oldest pose
            out[idx[0]] = x[idx[0]]
            if keep_prior: prior = marginalize_oldest(idx, factors, x, prior, Rfix)
            idx = idx[1:]
    out[idx] = x[idx]
    return out, x, idx, sum(t_solves), (prior if keep_prior else None)

def ate(est, gt):                                                # no alignment: the start is anchored, frames agree by construction
    return float(np.sqrt(np.mean(np.sum((est[:, :2] - gt[:, :2]) ** 2, axis=1))))

def full_batch(factors, x0, Rfix=None):
    t0 = time.perf_counter(); x = gauss_newton(list(range(N)), factors, x0, None, Rfix, iters=30); return x, time.perf_counter() - t0

def odometry_init(factors):
    x = np.zeros((N, 3))
    for f in factors:
        if f["k"] == "p0": x[0] = f["z"]
    for f in sorted([f for f in factors if f["k"] == "odo"], key=lambda f: f["i"]):
        i = f["i"]; x[i + 1, :2] = x[i, :2] + rot(x[i, 2]) @ f["z"][:2]; x[i + 1, 2] = x[i, 2] + f["z"][2]
    return x

# ------------------------------------------------------------------ cross-checks
def scipy_batch(factors, x0, Rfix=None):
    def fun(v):
        x = v.reshape(N, 3); return np.concatenate([res_jac(f, x, Rfix)[0] for f in factors])
    return least_squares(fun, x0.ravel(), method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14).x.reshape(N, 3)

def jac_check(factors, x):
    """Analytic Jacobians vs central finite differences, worst absolute error over all factors."""
    worst = 0.0
    for f in factors:
        r, Js = res_jac(f, x)
        for p, J in Js.items():
            for c in range(3):
                e = np.zeros_like(x); e[p, c] = 1e-6
                Jn = (res_jac(f, x + e)[0] - res_jac(f, x - e)[0]) / 2e-6
                worst = max(worst, float(np.abs(Jn - J[:, c]).max()))
    return worst

def main():
    out = Path("out"); out.mkdir(exist_ok=True)
    SEED = 7
    gt, lms, F, n_obs = make_problem(SEED)
    x0 = odometry_init(F)
    n_lm_frames = len({f["i"] for f in F if f["k"] == "lm"})

    # ---- sanity: Jacobians, own batch vs scipy batch
    jerr = jac_check(F, x0)
    xb, t_batch = full_batch(F, x0)
    xs = scipy_batch(F, x0)                      # independent solver, started from the odometry chain, not from our answer
    batch_vs_scipy = float(np.abs(xb - xs).max())

    # ---- (a) (b) (c) on the nonlinear problem
    xp, xp_final, idx_p, t_p, prior_p = run_window(F, x0, True)
    xd, xd_final, idx_d, t_d, _ = run_window(F, x0, False)
    # independent check of the carried prior: re-solve the final window with scipy, prior as an explicit residual
    # (same prior object, same factors) and compare to our Gauss-Newton window solution
    fin = idx_p
    def fun_final(v):
        xx = xp_final.copy(); xx[fin] = v.reshape(-1, 3)
        rs = [res_jac(f, xx)[0] for f in F if all(p in fin for p in ([f["i"]] + ([f["j"]] if f["k"] == "odo" else [])))]
        rs.append(prior_res_jac(prior_p, xx, fin[0])[0]); return np.concatenate(rs)
    xw_sc = least_squares(fun_final, x0[fin].ravel(), method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14).x.reshape(-1, 3)
    window_vs_scipy = float(np.abs(xw_sc - xp_final[fin]).max())

    def row(est, t):
        return {"ate_vs_truth_m": ate(est, gt), "dist_to_batch_m": ate(est, xb), "final_pose_err_m": float(np.linalg.norm(est[-1, :2] - gt[-1, :2])), "solve_time_s": t}
    odo = {"ate_vs_truth_m": ate(x0, gt), "dist_to_batch_m": ate(x0, xb), "final_pose_err_m": float(np.linalg.norm(x0[-1, :2] - gt[-1, :2])), "solve_time_s": 0.0}
    main_rows = {"odometry_only": odo, "full_batch": row(xb, t_batch), "window_with_prior": row(xp, t_p), "window_prior_dropped": row(xd, t_d)}
    # final-window agreement with batch (retained poses only)
    main_rows["window_with_prior"]["final_window_dist_to_batch_m"] = float(np.sqrt(np.mean(np.sum((xp_final[idx_p, :2] - xb[idx_p, :2]) ** 2, axis=1))))
    main_rows["window_prior_dropped"]["final_window_dist_to_batch_m"] = float(np.sqrt(np.mean(np.sum((xd_final[idx_d, :2] - xb[idx_d, :2]) ** 2, axis=1))))
    main_rows["full_batch"]["ate_to_truth_note"] = "global, uses all factors incl. future ones"

    # ---- LINEAR verification (rotations frozen at the true values => every factor is linear in the state)
    Rfix = gt[:, 2].copy()
    xl_b, _ = full_batch(F, x0, Rfix)
    xl_p, xl_pf, il_p, _, _ = run_window(F, x0, True, Rfix)
    xl_d, xl_df, il_d, _, _ = run_window(F, x0, False, Rfix)
    xl_s = scipy_batch(F, x0, Rfix)                                # independent batch solution of the linear problem
    lin_prior_gap = float(np.abs(xl_pf[il_p] - xl_s[il_p]).max())  # window+prior vs scipy batch
    lin_prior_gap_own = float(np.abs(xl_pf[il_p] - xl_b[il_p]).max())
    lin_drop_gap = float(np.abs(xl_df[il_d] - xl_s[il_d]).max())
    # same check, window ends at every position, not just the end: use the retained poses of each retired window state? (end only is the contract)
    # nonlinear gap for context
    nl_gap = float(np.abs(xp_final[idx_p] - xb[idx_p]).max())

    # ---- many seeds: is the ranking robust?
    seeds = list(range(100, 130)); table = []
    for sd in seeds:
        g, _, Fs, _ = make_problem(sd); xi = odometry_init(Fs); b, _ = full_batch(Fs, xi)
        p = run_window(Fs, xi, True)[0]; d = run_window(Fs, xi, False)[0]
        table.append([ate(xi, g), ate(b, g), ate(p, g), ate(d, g), ate(p, b), ate(d, b)])
    T = np.array(table)
    names = ["odometry_ate", "batch_ate", "window_prior_ate", "window_drop_ate", "window_prior_to_batch", "window_drop_to_batch"]
    multi = {n: {"median": float(np.median(T[:, i])), "p90": float(np.percentile(T[:, i], 90))} for i, n in enumerate(names)}
    multi["trials"] = len(seeds)
    multi["drop_worse_than_prior_in"] = int(np.sum(T[:, 3] > T[:, 2]))

    # ---- evidence files
    np.savetxt(out / "sliding_window_traj.csv", np.c_[np.arange(N), gt, x0, xb, xp, xd], delimiter=",", fmt="%.6f",
               header="k,gt_x,gt_y,gt_th,odo_x,odo_y,odo_th,batch_x,batch_y,batch_th,win_prior_x,win_prior_y,win_prior_th,win_drop_x,win_drop_y,win_drop_th", comments="")
    np.savetxt(out / "sliding_window_landmarks.csv", lms, delimiter=",", fmt="%.6f", header="x,y", comments="")
    obs = np.array([[f["i"]] for f in F if f["k"] == "lm"]).ravel()
    np.savetxt(out / "sliding_window_obs_per_keyframe.csv", np.c_[np.arange(N), np.bincount(obs, minlength=N)], delimiter=",", fmt="%d", header="k,landmark_obs", comments="")
    report = {"model": "planar SE(2)", "keyframes": N, "window": W, "seed": SEED,
              "odometry": {"sigma_t_m": SIG_T, "sigma_rot_rad": SIG_R, "scale_bias": BIAS_SCALE, "turn_bias_rad_per_step": BIAS_TURN},
              "landmark": {"count": int(len(lms)), "observations": n_obs, "keyframes_with_a_landmark": n_lm_frames, "range_m": LM_RANGE, "sigma_m": SIG_LM, "map": "known"},
              "main_run": main_rows,
              "checks": {"jacobian_vs_finite_difference_max_abs": jerr, "own_batch_vs_scipy_max_abs": batch_vs_scipy, "window_gn_vs_scipy_with_same_prior_max_abs": window_vs_scipy,
                         "linear_window_prior_vs_scipy_batch_max_abs_m": lin_prior_gap, "linear_window_prior_vs_own_batch_max_abs_m": lin_prior_gap_own, "linear_window_dropped_vs_scipy_batch_max_abs_m": lin_drop_gap,
                         "nonlinear_window_prior_vs_batch_max_abs": nl_gap},
              "thirty_seeds": multi, "status": "PASS"}
    # ---- assertions that can fail
    assert jerr < 1e-6, "analytic Jacobians must match finite differences"
    assert batch_vs_scipy < 1e-6, "own Gauss-Newton batch must match scipy least_squares"
    assert window_vs_scipy < 1e-6, "own window solve with the carried prior must match scipy on the same prior"
    assert lin_prior_gap < 1e-8 and lin_prior_gap_own < 1e-8, "linear factors: marginalized-prior window must equal the batch on the retained poses"
    assert lin_drop_gap > 1e-2, "control: dropping the prior must visibly differ from the batch on the same retained poses"
    assert odo["ate_vs_truth_m"] > 3 * main_rows["full_batch"]["ate_vs_truth_m"], "odometry must drift: batch must be much better than the raw chain"
    assert main_rows["window_with_prior"]["dist_to_batch_m"] < 0.5 * main_rows["window_prior_dropped"]["dist_to_batch_m"], "main run: prior must keep the window closer to the batch"
    assert multi["window_prior_to_batch"]["median"] < 0.5 * multi["window_drop_to_batch"]["median"], "over 30 seeds the prior must matter"
    assert multi["drop_worse_than_prior_in"] >= 24, "the ranking must hold in nearly every seed, not on average only"
    (out / "sliding_window_real_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("sliding window lab: PASS"); print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
