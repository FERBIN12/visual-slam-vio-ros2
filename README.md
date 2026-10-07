# Visual SLAM and visual-inertial estimation: small experiments with known answers

Twenty-five deterministic experiments that reproduce the numbers shown in the Visual SLAM course
(camera geometry, feature-based visual odometry, IMU models, visual-inertial estimation and trajectory evaluation).

Every experiment builds **synthetic data where the right answer is known**, runs a method, and then runs a
**deliberately broken control** (wrong focal length, flipped sign, blurred image, two millimetre baseline, plain
least squares, a stale time step, and so on). An experiment passes only if the correct case lands where it should
**and** the broken control looks clearly broken. Every random source is seeded, so the numbers are identical on every run.

## Run it

    pip install -r requirements.txt
    python3 run_all.py

`run_all.py` runs every lab from this folder, prints PASS or FAIL for each, and prints the key numbers each one
measured. Evidence files (csv, json, png) are written to `out/`. It needs only NumPy, SciPy and OpenCV:
no ROS, no GPU, no dataset download.

## What is here

| area | labs | what they measure |
|---|---|---|
| camera geometry | `geometry_lab`, `projection_smoke`, `distortion_smoke` | pinhole projection and its inverse, the cost of a 10% focal error, lens distortion and rectification, a stereo row check |
| features and matching | `features_smoke`, `matching_smoke` | corner detection count, coverage and repeatability; descriptor matching with ratio and mutual tests; a repeated-texture picture that fools every filter except geometry |
| two views | `epipolar_smoke`, `relative_pose_smoke`, `triangulation_smoke` | essential matrix with RANSAC and Sampson error, pose recovery, triangulation, and the small-baseline trap where reprojection looks fine while depth is wrong |
| calibration | `calibration_real` | a real checkerboard calibration with held-out views: reprojection error stays at the noise floor while the focal length is still off by a measurable amount; three flat views give a wildly wrong camera with the same score |
| frames and time | `imu_measurement_smoke`, `extrinsics_time_offset_smoke` | accelerometer specific force and frame conventions with sign, direction, axis and unit faults; camera-IMU extrinsics and time offset (noise-free data) |
| robust estimation | `motion_ba_smoke` | motion-only bundle adjustment with a Huber kernel against plain least squares on data with outliers |
| visual odometry | `pnp_smoke`, `keyframe_policy_smoke`, `landmark_lifecycle_smoke`, `vo_frontend_replay` | PnP with outliers and poor landmark geometry; keyframe insertion policy and its ablations; landmark promotion, culling and fusion; a front end state machine with fault injection and recovery |
| IMU | `imu_discrete_integration_smoke`, `imu_bias_noise_smoke`, `imu_preintegration_smoke`, `imu_covariance_jacobian_smoke`, `imu_readiness_smoke` | midpoint integration against Euler, noise density and bias random walk with a Monte Carlo check, preintegration parity and bias correction, covariance and Jacobian propagation checked by sampling, data-quality checks on generated IMU streams |
| initialization and marginalization | `vio_initialization_smoke`, `marginalization_smoke` | solving scale, gravity and accelerometer bias linearly, and the Schur-complement identity |
| sliding window | `sliding_window_real` | a keyframe window that marginalizes old poses with a Schur-complement prior, against the full batch and against a window that drops the prior (30 seeds); cross-checked against scipy |
| trajectory evaluation | `trajectory_eval_real` | ATE and RPE with SE(3) and Sim(3) alignment on a synthetic trajectory, and how scale, constant offsets, time offsets, swapped pose conventions and lever arms change the numbers; cross-checked against an SVD fit, scipy and, when installed, the `evo` package |

## What this repository does not contain

- **No real dataset.** Everything is synthetic. The results show how methods behave under conditions you control,
  not how they perform on your robot, lens or building.
- **No ORB-SLAM3 and no other external SLAM system is run here.**
- **No ROS 2 nodes, Gazebo world or public-dataset benchmark.** An earlier version of this repository listed "contract"
  scripts for those topics. They only checked hand-written configuration dictionaries and measured nothing, so they
  have been removed rather than left to suggest results that were never produced.
- Some labs use noise-free data on purpose (for example the time-offset lab). Its tiny errors confirm that the model and the
  solver agree, nothing more.

## Using it

Open a lab and read its assertion lines: they are the contract. Change a constant (more noise, a shorter baseline,
fewer views), run the lab again, and see where the contract starts to fail. Then write a failure case of your own.

## License

MIT
