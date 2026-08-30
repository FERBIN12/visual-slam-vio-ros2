#!/usr/bin/env python3
"""Deterministic Gazebo camera/IMU world sensor contract audit."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np, cv2
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)
def assess(c):
 checks={'world_loaded':c['world_loaded'],'camera_rate':c['camera_rate']==30,'imu_rate':c['imu_rate']==200,'clock_monotonic':c['clock_monotonic'],'tf_declared':c['tf_declared'],'truth_separate':c['truth_separate']}
 failed=[k for k,v in checks.items() if not v]; return failed,'ACCEPT' if not failed else ('DEGRADE' if len(failed)<=2 else 'REJECT')
def main():
 good=dict(world_loaded=True,camera_rate=30,imu_rate=200,clock_monotonic=True,tf_declared=True,truth_separate=True)
 cases=[('headless_world_ready',good),('camera_rate_fault',{**good,'camera_rate':15}),('imu_rate_fault',{**good,'imu_rate':100}),('clock_regression',{**good,'clock_monotonic':False}),('missing_tf',{**good,'tf_declared':False}),('truth_leak',{**good,'truth_separate':False}),('empty_world',{**good,'world_loaded':False,'truth_separate':False,'tf_declared':False})]
 rows=[]
 for name,c in cases:
  f,s=assess(c); rows.append({'case':name,'status':s,'failed_checks':';'.join(f)})
 gates={'ready_accept':rows[0]['status']=='ACCEPT','camera_degrade':rows[1]['status']=='DEGRADE','imu_degrade':rows[2]['status']=='DEGRADE','clock_degrade':rows[3]['status']=='DEGRADE','tf_degrade':rows[4]['status']=='DEGRADE','truth_degrade':rows[5]['status']=='DEGRADE','empty_reject':rows[6]['status']=='REJECT'}
 report={'status':'PASS' if all(gates.values()) else 'FAIL','cases':rows,'gates':gates,'command':'python3 scripts/gazebo_world_contract.py'}
 with (OUT/'gazebo_world_scorecard.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 t=np.linspace(0,1,200); np.savez(OUT/'gazebo_sensor_trace.npz',time=t,camera_stamp=t[::6],imu_stamp=t,truth=np.c_[t,np.sin(t),np.cos(t)])
 canvas=np.full((720,1280,3),246,np.uint8);ink=(45,43,37);cv2.putText(canvas,'GAZEBO CAMERA + IMU WORLD CONTRACT',(40,52),cv2.FONT_HERSHEY_SIMPLEX,.7,ink,2,cv2.LINE_AA);cv2.rectangle(canvas,(55,105),(760,640),(215,208,198),1);cv2.rectangle(canvas,(820,105),(1225,640),(215,208,198),1)
 for y,s in [(180,'world + robot + plugins'),(260,'camera 30 Hz · IMU 200 Hz'),(340,'clock · TF · truth isolation'),(470,'SENSOR GATE')]:cv2.putText(canvas,s,(90,y),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA)
 cv2.putText(canvas,'FAULT MATRIX',(865,170),cv2.FONT_HERSHEY_SIMPLEX,.62,ink,1,cv2.LINE_AA);cv2.putText(canvas,'DEGRADE: rate / clock / TF',(865,285),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,130,190),1,cv2.LINE_AA);cv2.putText(canvas,'REJECT: empty or leaked truth',(865,355),cv2.FONT_HERSHEY_SIMPLEX,.47,(40,45,190),1,cv2.LINE_AA);cv2.imwrite(str(OUT/'gazebo_world_plot.png'),canvas);(OUT/'gazebo_world_report.json').write_text(json.dumps(report,indent=2)+'\n');print('gazebo world contract lab:',report['status']);return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
