#!/usr/bin/env python3
"""Run every lab in this repository from a clean checkout and print what each one MEASURED.

    python3 run_all.py

Each lab is a deterministic experiment on synthetic data with a known answer and a deliberately broken
control. A lab passes only if its own assertions hold, including that the broken control looks broken.
Nothing here uses a real dataset, and nothing here runs ORB-SLAM3 or any other external SLAM system.
"""
import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABS = [  # (script, args, report file, [(label, key path)])
    ("geometry_lab.py", ["roundtrip"], None, []),
    ("projection_smoke.py", [], "projection_report.json", [("median error with a 10% focal fault, px", "median_bad_intrinsics_error_px")]),
    ("distortion_smoke.py", [], "rectification_report.json", [("undistort round trip, worst", "round_trip_max"), ("stereo row error after rectification, px", "rectified_row_max_px")]),
    ("features_smoke.py", [], "features_report.json", [("keypoints on the test image", "keypoints"), ("keypoints after blur", "weak_texture_keypoints")]),
    ("matching_smoke.py", [], "matches_report.json", [("inlier ratio, good image", "geometry_inlier_ratio"), ("inlier ratio, repeated texture", "repeated_texture.geometry_inlier_ratio")]),
    ("epipolar_smoke.py", [], "epipolar_report.json", [("median Sampson error, px", "median_sampson_px"), ("same, wrong model, px", "wrong_model_median_sampson_px")]),
    ("relative_pose_smoke.py", [], "relative_pose_report.json", [("rotation error, deg", "rotation_error_deg"), ("pose inliers with 8 mm motion", "tiny_forward_motion.pose_inliers")]),
    ("triangulation_smoke.py", [], "point_cloud_report.json", [("depth error, 18 cm baseline", "trusted_median_relative_depth_error"), ("depth error, 2 mm baseline", "tiny_baseline.median_relative_depth_error")]),
    ("calibration_real.py", [], "calibration_real_report.json", [("focal error, 20 views, median %", "summary.20_views.abs_fx_error_pct_median"), ("focal error, 3 flat views, median %", "summary.flat_3_views.abs_fx_error_pct_median")]),
    ("extrinsics_time_offset_smoke.py", [], "extrinsics_time_offset_report.json", [("offset error, ms (noise-free data)", "fit.parameter_errors.offset_ms"), ("conditioning with no excitation", "faults.weak_motion_condition")]),
    ("imu_measurement_smoke.py", [], "imu_measurement_report.json", []),
    ("motion_ba_smoke.py", [], "motion_ba_report.json", [("rotation error, robust, deg", "rotation_error_deg"), ("rotation error, plain least squares, deg", "plain_l2_control.rotation_error_deg")]),
    ("pnp_smoke.py", [], "pnp_report.json", [("PnP rotation error, deg", "rotation_error_deg"), ("same, clustered landmarks, deg", "clustered_landmarks.rotation_error_deg")]),
    ("keyframe_policy_smoke.py", [], "keyframe_policy_report.json", [("keyframes, full policy", "full_policy.keyframes"), ("keyframes, rotation-only ablation", "ablations.rotation_only.keyframes")]),
    ("landmark_lifecycle_smoke.py", [], "landmark_lifecycle_report.json", [("landmarks mature", "full_policy.mature"), ("landmarks culled", "full_policy.culled")]),
    ("vo_frontend_replay.py", [], "vo_failure_report.json", [("fault-free control ATE, m", "fault_free_control.ate_m")]),
    ("imu_discrete_integration_smoke.py", [], "imu_integration_report.json", [("position error with a stale dt, m", "faults.stale_dt_position_error_m"), ("naive Euler orthogonality error", "faults.naive_euler_orthogonality_error")]),
    ("imu_bias_noise_smoke.py", [], "imu_bias_noise_report.json", [("gyro noise density, estimated", "allan_style.estimated_gyro_noise_density"), ("position error, empirical / predicted", "monte_carlo.final_empirical_to_predicted.position_empirical_to_predicted")]),
    ("imu_preintegration_smoke.py", [], "preintegration_report.json", [("preintegration vs direct, position m", "direct_parity.position_m"), ("dropped-boundary fault, position m", "boundary_fault.position_error_m")]),
    ("imu_covariance_jacobian_smoke.py", [], "preintegration_covariance_report.json", [("Jacobian max abs error", "jacobian.max_absolute_error"), ("sample-sigma-as-density underestimate ratio", "faults.sample_sigma_as_density_underestimate_ratio")]),
    ("imu_readiness_smoke.py", [], None, []),
    ("vio_initialization_smoke.py", [], "vio_initialization_report.json", [("scale error", "errors.scale"), ("condition number, exciting motion", "exciting.condition_number")]),
    ("marginalization_smoke.py", [], "marginalization_report.json", [("reduced vs full solution, max error", "replay_max_error")]),
]

def dig(d, path):
    for k in path.split("."):
        d = d[k]
    return d

def main():
    (HERE / "out").mkdir(exist_ok=True)
    failed, rows, measured = [], [], []
    for script, args, report, keys in LABS:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / "labs" / script), *args], cwd=HERE, capture_output=True, text=True)
        dt = time.time() - t0
        ok = r.returncode == 0
        if not ok:
            failed.append(script)
        rows.append((script, ok, dt))
        print(f"{'PASS' if ok else 'FAIL'}  {script:34s} {dt:5.1f}s")
        if not ok:
            print("      " + (r.stderr.strip().splitlines() or ["(no stderr)"])[-1][:200])
            continue
        if report and (HERE / "out" / report).exists():
            data = json.loads((HERE / "out" / report).read_text())
            for label, path in keys:
                try:
                    v = dig(data, path)
                    measured.append({"lab": script, "label": label, "value": v})
                    print(f"        {label:44s} {v:.4g}" if isinstance(v, (int, float)) else f"        {label:44s} {v}")
                except Exception:
                    print(f"        {label:44s} (not in report)")
    print(f"\n{len(LABS) - len(failed)} of {len(LABS)} labs passed")
    summary = {"labs": [{"script": r[0], "passed": r[1]} for r in rows], "passed": len(LABS) - len(failed), "total": len(LABS), "measured": measured}
    (HERE / "out" / "run_all_summary.json").write_text(json.dumps(summary, indent=2))
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
