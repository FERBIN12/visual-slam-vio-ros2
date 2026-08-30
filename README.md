# Visual SLAM & VIO Estimation Stack

A from-scratch implementation of the estimation pipeline behind a monocular/stereo-inertial
SLAM system, plus a project-owned ROS 2 integration layer around upstream **ORB-SLAM3**,
running in a Gazebo Fortress simulation with EuRoC/TUM-VI dataset evaluation.

Every stage of the pipeline — camera geometry, feature-based visual odometry, IMU
preintegration, sliding-window optimization, and trajectory evaluation — is implemented as
a standalone, deterministic, seeded Python lab with its own pass/fail contract and a
generated evidence artifact (JSON report, CSV trace, or diagnostic plot). Nothing is
asserted without a reproducible numeric check, including the failure modes.

## What's implemented

**Camera geometry and calibration**
- Pinhole projection, round-trip, and stereo geometry (`geometry_lab.py`, `projection_smoke.py`)
- Radial distortion and rectification (`distortion_smoke.py`)
- Pinhole calibration contract with synthetic checkerboard data (`calibration_smoke.py`)
- Camera-IMU extrinsic and time-offset calibration (`extrinsics_time_offset_smoke.py`)
- Calibration/timing sensitivity sweeps (`calibration_timing_sweep_contract.py`)

**Feature-based visual odometry front end**
- Feature detection and image-coverage diagnostics (`features_smoke.py`)
- Descriptor matching with ratio test, mutual check, and a false-match gate (`matching_smoke.py`)
- Essential-matrix estimation and Sampson-residual epipolar geometry (`epipolar_smoke.py`)
- Relative pose recovery with an injected-outlier and low-parallax control (`relative_pose_smoke.py`)
- Two-view triangulation with cheirality and reprojection checks (`triangulation_smoke.py`)
- Metric PnP with RANSAC-style outlier rejection (`pnp_smoke.py`)
- Multi-signal keyframe selection policy with hysteresis (`keyframe_policy_smoke.py`)
- Landmark lifecycle / local-mapping state machine (`landmark_lifecycle_smoke.py`)
- VO frontend state machine, trajectory gate, and fault replay (`vo_frontend_replay.py`)

**IMU modeling and preintegration**
- Discrete SO(3) midpoint IMU integration vs. Euler control (`imu_discrete_integration_smoke.py`)
- On-manifold keyframe IMU preintegration with parity checks (`imu_preintegration_smoke.py`)
- Preintegration covariance propagation and Jacobians, verified against Monte Carlo
  sampling (`imu_covariance_jacobian_smoke.py`)
- IMU measurement model with sign/unit/frame fault injection (`imu_measurement_smoke.py`)
- Bias and noise modeling with Allan-style and covariance-growth controls (`imu_bias_noise_smoke.py`)
- Sensor readiness: timing, saturation, and excitation audits (`imu_readiness_smoke.py`)

**VIO / sliding-window optimization**
- VIO initialization: scale, gravity direction, and accelerometer bias recovery from a
  linear least-squares solve, with an excitation-quality control case (`vio_initialization_smoke.py`)
- Visual/IMU residuals, Jacobians, whitening, and outlier handling (`residual_factor_smoke.py`)
- Residual whitening and innovation consistency checks (`residual_consistency_contract.py`)
- Bounded nonlinear sliding-window bundle adjustment (`sliding_window_smoke.py`)
- Motion-only bundle adjustment with robust-kernel vs. L2 comparison (`motion_ba_smoke.py`)
- Schur-complement marginalization with a full-vs-reduced-system replay check
  (`marginalization_smoke.py`)
- VIO observability and fault-containment replay (`vio_observability_replay_smoke.py`)
- Executable residual unit test for the optimizer core (`reprojection_factor_test.py`)

**ROS 2 sensor pipeline and simulation**
- Gazebo Fortress world publishing synchronized camera + IMU data
  (`prototype/worlds/estimation_lab.sdf`) bridged into ROS 2 via `ros_gz_bridge`
  (`prototype/bridge.yaml`), smoke-tested end-to-end in `scripts/prototype_smoke.sh`
- ROS 2 Image/CameraInfo/Imu message contract (`ros_sensor_contract_smoke.py`)
- Camera/IMU TF-tree transform audit (`tf_tree_camera_imu_smoke.py`)
- QoS, synchronization, and bounded-queue backpressure handling (`qos_sync_backpressure_smoke.py`)
- Camera-IMU ROS 2 pipeline contract (`ros2_pipeline_contract.py`)
- rosbag2-style manifest, integrity, and replay-parity audit (`rosbag_replay_clock_smoke.py`)
- Gazebo camera/IMU world sensor contract (`gazebo_world_contract.py`)
- Sensor fault injection: noise, bias, dropout, timing faults (`sensor_fault_contract.py`)

**ORB-SLAM3 integration**
- ROS 2 adapter contract for ORB-SLAM3 ingress messages (`ros_adapter_contract.py`)
- Estimator-bridge lifecycle, diagnostics, and integration audit (`estimator_bridge_smoke.py`)
- ORB-SLAM3 settings/mode/provenance preflight (`orbslam3_settings_preflight.py`)
- Stereo-inertial run contract and runtime failure audit
  (`orbslam3_run_contract.py`, `orbslam3_runtime_smoke.py`)
- Tracking state, reset, and relocalization contract (`tracking_state_contract.py`)
- Tracking-loss and recovery state machine (`tracking_recovery_contract.py`)
- Pipeline preflight, fault injection, and recovery audit (`pipeline_preflight_fault_smoke.py`)

**Dataset, ground truth, and evaluation**
- EuRoC / TUM-VI dataset contract and provenance gate (`dataset_contract.py`)
- Ground-truth association and coordinate-frame alignment (`ground_truth_alignment_contract.py`)
- Trajectory export and SE(3) frame-alignment contract (`trajectory_alignment_contract.py`)
- ATE/RPE trajectory evaluation against aligned ground truth (`ate_rpe_contract.py`)
- Simulation-vs-dataset contract parity check (`sim_dataset_parity_contract.py`)
- Trajectory/runtime evidence scorecard (`trajectory_runtime_contract.py`)
- Real-time budget and latency contract (`realtime_budget_contract.py`)
- Failure taxonomy and regression matrix (`failure_regression_contract.py`)
- Deterministic experiment manifest validation (`experiment_manifest_contract.py`)

**End-to-end reproducible run**
- End-to-end ORB-SLAM3 run manifest and release gate (`reproducible_run_contract.py`)
- Fault injection and recovery across the full pipeline (`capstone_fault_recovery_contract.py`)
- Reproducible packaging contract (`capstone_package_contract.py`)
- Final acceptance gate tying every contract above into one pass/fail (`acceptance_contract.py`)

## Design principles

- **Deterministic and seeded.** Every lab uses a fixed RNG seed and asserts numeric
  tolerances against known ground truth — no lab merely "looks right."
- **Failure cases are first-class.** Most labs run a matched pair: a clean/well-conditioned
  case and a degraded control (outliers, low parallax, constant-velocity ambiguity, sensor
  dropout, timing faults) to prove the check actually discriminates good from bad, not just
  that it runs.
- **Contracts, not just demos.** Scripts named `*_contract.py` encode an explicit
  accept/reject specification (required fields, tolerances, state-machine transitions) so
  the pipeline's assumptions are checked mechanically, not asserted in prose.
- **ORB-SLAM3 is a dependency, not a fork.** Upstream ORB-SLAM3 is built from source in the
  container; this repo owns only the ROS 2 bridge, contracts, and evaluation harness around
  it — see [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (GPLv3) for the estimator
  itself.

## Stack

- Ubuntu 22.04, ROS 2 Humble, Gazebo Fortress (`ros_gz_bridge` for sim <-> ROS 2 transport)
- Upstream ORB-SLAM3 (stereo-inertial), built against Pangolin 0.9.2, OpenCV, Eigen3
- Python labs: NumPy, OpenCV, SciPy (`scipy.optimize.least_squares` for the sliding-window
  and motion-only BA solvers)
- Evaluation: `evo` for trajectory alignment/ATE/RPE tooling, rosbag2 for deterministic replay
- Reproducible container defined in `env/Dockerfile` / `env/compose.yaml`

## Layout

```
scripts/      Algorithm labs and pipeline contracts (see above)
prototype/    Gazebo Fortress world + ros_gz_bridge config for the camera/IMU rig
env/          Dockerfile + compose.yaml for the pinned ROS 2 Humble / Gazebo Fortress image
```

## Running a lab

Each script is standalone and writes its evidence artifact under `out/`:

```bash
python3 scripts/pnp_smoke.py
python3 scripts/imu_preintegration_smoke.py
python3 scripts/marginalization_smoke.py
```

The simulation + ROS 2 bridge smoke test runs inside the container:

```bash
docker compose -f env/compose.yaml build
scripts/prototype_smoke.sh
```

## Dependencies

```bash
pip install -r requirements.txt
```

ORB-SLAM3 itself is not vendored here; build it from
[upstream](https://github.com/UZ-SLAMLab/ORB_SLAM3) inside the provided container image
(`env/Dockerfile` installs its build dependencies: Eigen3, OpenCV, Boost, Pangolin 0.9.2).
