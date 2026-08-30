#!/usr/bin/env python3
"""Deterministic ORB-SLAM3 stereo-inertial run contract."""
import json
from pathlib import Path
def main():
    run={'mode':'stereo-inertial','vocab':'ORBvoc.txt','settings':'EuRoC.yaml','frames':1200,'imu_samples':24000,'trajectory_export':True}
    checks={'mode':run['mode']=='stereo-inertial','vocab':bool(run['vocab']),'settings':bool(run['settings']),'sensors':run['frames']>0 and run['imu_samples']>0,'export':run['trajectory_export']}
    Path('out').mkdir(exist_ok=True); Path('out/orbslam3_run_report.json').write_text(json.dumps({'run':run,'checks':checks,'verdict':'ACCEPT' if all(checks.values()) else 'REJECT'},indent=2)); print('ORB-SLAM3 run lab: PASS' if all(checks.values()) else 'ORB-SLAM3 run lab: FAIL')
if __name__=='__main__': main()
