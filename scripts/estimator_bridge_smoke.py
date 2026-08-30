#!/usr/bin/env python3
"""Deterministic estimator-bridge lifecycle, diagnostics, and integration audit."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)
def audit(c):
    checks={'lifecycle':c['lifecycle']=='active','frames':c['input_frame']=='camera_optical_frame' and c['output_frame']=='odom','bounded_queue':c['queue_depth']<=10,'diagnostics':c['diagnostics'],'timestamp':c['output_stamp']>0,'integration':c['integration_pass']}
    critical=not checks['frames'] or not checks['lifecycle'] or not checks['integration']; return checks,'ACCEPT' if all(checks.values()) else ('REJECT' if critical else 'DEGRADE')
def main():
    good={'lifecycle':'active','input_frame':'camera_optical_frame','output_frame':'odom','queue_depth':4,'diagnostics':True,'output_stamp':1.25,'integration_pass':True}
    cases=[('trusted_bridge',good),('inactive_lifecycle',{**good,'lifecycle':'inactive'}),('frame_mismatch',{**good,'output_frame':'map'}),('queue_overrun',{**good,'queue_depth':32,'diagnostics':False}),('integration_exception',{**good,'integration_pass':False})]
    rows=[]
    for name,c in cases:
        ch,s=audit(c); rows.append({'case':name,'status':s,'failed_checks':';'.join(k for k,v in ch.items() if not v),'queue_depth':c['queue_depth']})
    checks={'trusted_accept':rows[0]['status']=='ACCEPT','inactive_reject':rows[1]['status']=='REJECT','frame_reject':rows[2]['status']=='REJECT','queue_degrade':rows[3]['status']=='DEGRADE','integration_reject':rows[4]['status']=='REJECT','distinct_faults':len({r['failed_checks'] for r in rows[1:]})==4}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','cases':rows,'checks':checks,'command':'python3 scripts/estimator_bridge_smoke.py','artifacts':{'report':'out/estimator_bridge_report.json','scorecard':'out/estimator_bridge_scorecard.csv','model':'out/estimator_bridge_model.npz','plot':'out/estimator_bridge_plot.png'}}
    with (OUT/'estimator_bridge_scorecard.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(OUT/'estimator_bridge_model.npz',queue_depth=np.array([r['queue_depth'] for r in rows]),pose=np.eye(4),diagnostic_flags=np.array([1,1,0,0,1]))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); teal=(115,140,25); red=(40,45,190); amber=(20,130,190)
    cv2.putText(canvas,'ESTIMATOR BRIDGE NODE',(42,52),cv2.FONT_HERSHEY_SIMPLEX,.78,ink,2,cv2.LINE_AA); cv2.rectangle(canvas,(55,105),(760,635),(215,208,198),1); cv2.putText(canvas,'VALIDATE → BUFFER → CALL → PUBLISH',(90,170),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'lifecycle active · queue 4 · diagnostics OK',(90,270),cv2.FONT_HERSHEY_SIMPLEX,.54,ink,1,cv2.LINE_AA); cv2.putText(canvas,'ACCEPT', (90,405),cv2.FONT_HERSHEY_SIMPLEX,.8,teal,2,cv2.LINE_AA); cv2.rectangle(canvas,(820,105),(1225,635),(215,208,198),1); cv2.putText(canvas,'CONTROL MATRIX',(855,155),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'lifecycle / frame / queue',(855,245),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'exception / integration',(855,300),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'DEGRADE', (855,420),cv2.FONT_HERSHEY_SIMPLEX,.68,amber,2,cv2.LINE_AA); cv2.putText(canvas,'REJECT', (855,490),cv2.FONT_HERSHEY_SIMPLEX,.68,red,2,cv2.LINE_AA); cv2.imwrite(str(OUT/'estimator_bridge_plot.png'),canvas); (OUT/'estimator_bridge_report.json').write_text(json.dumps(report,indent=2)+'\n'); print('estimator bridge lab:',report['status']); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
