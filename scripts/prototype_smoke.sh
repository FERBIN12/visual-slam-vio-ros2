#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="visual-slam-vio:humble-fortress"

docker run --rm -v "$ROOT:/workspace/repo" "$IMAGE" '
  set -eo pipefail
  source /opt/ros/humble/setup.bash
  set -u
  ign gazebo -r -s /workspace/repo/prototype/worlds/estimation_lab.sdf >/tmp/gz.log 2>&1 &
  sim=$!
  ros2 run ros_gz_bridge parameter_bridge \
    --ros-args -p config_file:=/workspace/repo/prototype/bridge.yaml >/tmp/bridge.log 2>&1 &
  bridge=$!
  cleanup() { kill "$bridge" "$sim" 2>/dev/null || true; }
  trap cleanup EXIT
  sleep 5
  ros2 topic list | grep -Fx /estimator/camera/image_raw
  ros2 topic list | grep -Fx /estimator/camera/camera_info
  ros2 topic list | grep -Fx /estimator/imu/data
  timeout 3s ros2 topic echo /estimator/imu/data --once >/tmp/imu.txt
  grep -q "angular_velocity:" /tmp/imu.txt
  echo "prototype smoke test: PASS"
'
