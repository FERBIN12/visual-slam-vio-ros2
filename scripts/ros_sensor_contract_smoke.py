#!/usr/bin/env python3
"""Deterministic ROS 2-style Image/CameraInfo/Imu contract audit."""
from __future__ import annotations
import json,csv
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)
def audit(m):
    checks={'encoding':m['encoding']=='mono8','dimensions':m['width']==640 and m['height']==480,'frame':m['frame']=='camera_optical_frame','timestamp':m['stamp_s']>0,'covariance':m['imu_covariance']>0,'calibration_hash':m['calibration_hash']=='calib-r3'}; return checks,'ACCEPT' if all(checks.values()) else 'REJECT'
def main():
    good={'encoding':'mono8','width':640,'height':480,'frame':'camera_optical_frame','stamp_s':1.25,'imu_covariance':.01,'calibration_hash':'calib-r3'}; cases=[('trusted',good),('wrong_encoding',{**good,'encoding':'bgr8'}),('wrong_frame',{**good,'frame':'base_link'}),('zero_covariance',{**good,'imu_covariance':0.}),('stale_calibration',{**good,'calibration_hash':'calib-r1'})]; rows=[]
    for name,msg in cases: c,s=audit(msg); rows.append({'case':name,'status':s,'failed_checks':';'.join(k for k,v in c.items() if not v)})
    checks={'trusted_accept':rows[0]['status']=='ACCEPT','four_rejects':sum(r['status']=='REJECT' for r in rows[1:])==4,'distinct_reasons':len({r['failed_checks'] for r in rows[1:]})==4}; report={'status':'PASS' if all(checks.values()) else 'FAIL','cases':rows,'checks':checks,'command':'python3 scripts/ros_sensor_contract_smoke.py','artifacts':{'report':'out/ros_sensor_contract_report.json','scorecard':'out/ros_sensor_contract_scorecard.csv','model':'out/ros_sensor_contract_model.npz','plot':'out/ros_sensor_contract_plot.png'}}
    with (OUT/'ros_sensor_contract_scorecard.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    np.savez(OUT/'ros_sensor_contract_model.npz',image=np.zeros((480,640),np.uint8),camera_matrix=np.eye(3),statuses=np.array([r['status'] for r in rows]))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); teal=(115,140,25); red=(40,45,190); cv2.putText(canvas,'ROS 2 SENSOR CONTRACTS',(42,52),cv2.FONT_HERSHEY_SIMPLEX,.78,ink,2,cv2.LINE_AA); cv2.rectangle(canvas,(60,110),(760,635),(215,208,198),1); cv2.putText(canvas,'TRUSTED MESSAGE',(90,150),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA); cv2.putText(canvas,'mono8 · 640×480 · optical frame',(100,255),cv2.FONT_HERSHEY_SIMPLEX,.56,ink,1,cv2.LINE_AA); cv2.putText(canvas,'ACCEPT',(100,320),cv2.FONT_HERSHEY_SIMPLEX,.78,teal,2,cv2.LINE_AA); cv2.rectangle(canvas,(810,110),(1220,635),(215,208,198),1); cv2.putText(canvas,'REJECTION MATRIX',(850,150),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA); cv2.putText(canvas,'encoding / frame / covariance / hash',(850,270),cv2.FONT_HERSHEY_SIMPLEX,.5,ink,1,cv2.LINE_AA); cv2.putText(canvas,'4 distinct reasons',(850,340),cv2.FONT_HERSHEY_SIMPLEX,.65,red,2,cv2.LINE_AA); cv2.imwrite(str(OUT/'ros_sensor_contract_plot.png'),canvas); (OUT/'ros_sensor_contract_report.json').write_text(json.dumps(report,indent=2)+'\n'); print('ROS sensor contract lab:',report['status']); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
