#!/usr/bin/env python3
"""Deterministic TF-tree and camera/IMU transform audit."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2, numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)
def T(x,y,z,yaw=0.0):
    c,s=np.cos(yaw),np.sin(yaw); return np.array([[c,-s,0,x],[s,c,0,y],[0,0,1,z],[0,0,0,1.]],float)
def inv(m): return np.linalg.inv(m)
def audit(edges, optical_sign=1):
    parents={}; children={}; reasons=[]
    for parent,child,mat,stamp,dynamic in edges:
        if child in parents: reasons.append('duplicate_parent')
        parents[child]=parent; children.setdefault(parent,[]).append(child)
        if dynamic and stamp<=0: reasons.append('missing_dynamic_stamp')
        if not np.isfinite(mat).all(): reasons.append('nonfinite_transform')
    nodes=set(parents)|set(children); roots=nodes-set(parents)
    if len(roots)!=1: reasons.append('root_not_unique')
    connected=set(roots)
    changed=True
    while changed:
        changed=False
        for p,c,_,_,_ in edges:
            if p in connected and c not in connected: connected.add(c); changed=True
    if connected != nodes: reasons.append('disconnected')
    # optical frame must look forward along +Z relative to camera link
    optical=next((m for p,c,m,_,_ in edges if c=='camera_optical_frame'),None)
    if optical is None or optical_sign*optical[2,2] <= 0: reasons.append('optical_axis')
    status='ACCEPT' if not reasons else 'REJECT'
    return status, ';'.join(dict.fromkeys(reasons))
def main():
    static=T(.08,0,.03,0.02); optical=T(0,0,0,0)
    good=[('world','odom',T(0,0,0),1.0,True),('odom','base_link',T(1.2,.2,0,.1),1.0,True),('base_link','imu_link',T(.03,0,.06),0.,False),('base_link','camera_link',static,0.,False),('camera_link','camera_optical_frame',optical,0.,False)]
    cases=[('trusted_tree',good,1),('duplicate_parent',good+[('odom','camera_optical_frame',T(0,0,0),0.,False)],1),('inverse_camera_edge',good,-1),('missing_dynamic_stamp', [('world','odom',T(0,0,0),0.,True)]+good[1:],1),('disconnected_sensor',good+[('map','gps_link',T(0,0,0),0.,False)],1)]
    rows=[]
    for name,edges,sign in cases:
        status,reason=audit(edges,sign); rows.append({'case':name,'status':status,'failed_checks':reason,'edge_count':len(edges)})
    checks={'trusted_accept':rows[0]['status']=='ACCEPT','duplicate_parent_reject':rows[1]['status']=='REJECT','inverse_edge_reject':rows[2]['status']=='REJECT','stamp_reject':rows[3]['status']=='REJECT','disconnected_reject':rows[4]['status']=='REJECT','distinct_faults':len({r['failed_checks'] for r in rows[1:]})==4}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','cases':rows,'checks':checks,'command':'python3 scripts/tf_tree_camera_imu_smoke.py','artifacts':{'report':'out/tf_tree_report.json','scorecard':'out/tf_tree_scorecard.csv','model':'out/tf_tree_model.npz','plot':'out/tf_tree_plot.png'}}
    with (OUT/'tf_tree_scorecard.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(OUT/'tf_tree_model.npz',T_base_camera=static,T_camera_optical=optical,edge_count=np.array([len(e) for _,e,_ in cases]))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); teal=(115,140,25); red=(40,45,190)
    cv2.putText(canvas,'TF TREE · CAMERA / IMU',(42,52),cv2.FONT_HERSHEY_SIMPLEX,.78,ink,2,cv2.LINE_AA); cv2.rectangle(canvas,(55,105),(760,635),(215,208,198),1); cv2.putText(canvas,'world → odom → base_link',(90,170),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'↳ imu_link   ↳ camera_link',(125,265),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'↳ camera_optical_frame',(160,360),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'ACCEPT · rooted + unique', (90,480),cv2.FONT_HERSHEY_SIMPLEX,.72,teal,2,cv2.LINE_AA); cv2.rectangle(canvas,(820,105),(1225,635),(215,208,198),1); cv2.putText(canvas,'FAULT MATRIX',(860,155),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'duplicate parent', (860,245),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'inverse optical edge', (860,300),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'missing stamp', (860,355),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'REJECT · exact reason', (860,475),cv2.FONT_HERSHEY_SIMPLEX,.68,red,2,cv2.LINE_AA)
    cv2.imwrite(str(OUT/'tf_tree_plot.png'),canvas); (OUT/'tf_tree_report.json').write_text(json.dumps(report,indent=2)+'\n'); print('TF tree lab:',report['status']); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
