#!/usr/bin/env python3
"""Deterministic pipeline preflight, fault injection, and recovery audit."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)
def audit(c):
    checks={'topics_ready':c['topics_ready'],'clock_ready':c['clock_ready'],'tf_ready':c['tf_ready'],'diagnostics_ready':c['diagnostics_ready'],'fault_contained':c['fault_contained'],'state_reset':c['state_reset'],'continuity':c['continuity']}
    return checks,'ACCEPT' if all(checks.values()) else ('DEGRADE' if checks['fault_contained'] and checks['state_reset'] else 'REJECT')
def main():
    good={'topics_ready':True,'clock_ready':True,'tf_ready':True,'diagnostics_ready':True,'fault_contained':True,'state_reset':True,'continuity':True}
    cases=[('clean_start',good),('delayed_camera',{**good,'clock_ready':False}),('dropped_imu',{**good,'diagnostics_ready':False}),('misframed_output',{**good,'tf_ready':False,'fault_contained':False}),('restart_without_reset',{**good,'state_reset':False,'continuity':False}),('stale_buffer',{**good,'continuity':False,'state_reset':False})]
    rows=[]
    for name,c in cases:
        ch,s=audit(c); rows.append({'case':name,'status':s,'failed_checks':';'.join(k for k,v in ch.items() if not v)})
    checks={'clean_accept':rows[0]['status']=='ACCEPT','delay_degrade':rows[1]['status']=='DEGRADE','drop_degrade':rows[2]['status']=='DEGRADE','misframe_reject':rows[3]['status']=='REJECT','restart_reject':rows[4]['status']=='REJECT','stale_reject':rows[5]['status']=='REJECT','fault_matrix_complete':len(rows)==6}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','cases':rows,'checks':checks,'command':'python3 scripts/pipeline_preflight_fault_smoke.py','artifacts':{'report':'out/pipeline_preflight_report.json','scorecard':'out/pipeline_preflight_scorecard.csv','model':'out/pipeline_preflight_model.npz','plot':'out/pipeline_preflight_plot.png'}}
    with (OUT/'pipeline_preflight_scorecard.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(OUT/'pipeline_preflight_model.npz',readiness=np.array([1,1,1,1]),fault_count=np.array([len(rows)]),continuity=np.array([int(c['continuity']) for _,c in cases]))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); teal=(115,140,25); red=(40,45,190); amber=(20,130,190)
    cv2.putText(canvas,'PIPELINE PREFLIGHT · FAULT INJECTION',(42,52),cv2.FONT_HERSHEY_SIMPLEX,.75,ink,2,cv2.LINE_AA); cv2.rectangle(canvas,(55,105),(760,635),(215,208,198),1); cv2.putText(canvas,'TOPICS · CLOCK · TF · DIAGNOSTICS',(90,170),cv2.FONT_HERSHEY_SIMPLEX,.56,ink,1,cv2.LINE_AA); cv2.putText(canvas,'state reset · continuity invariant',(90,260),cv2.FONT_HERSHEY_SIMPLEX,.56,ink,1,cv2.LINE_AA); cv2.putText(canvas,'ACCEPT / RECOVER', (90,410),cv2.FONT_HERSHEY_SIMPLEX,.75,teal,2,cv2.LINE_AA); cv2.rectangle(canvas,(820,105),(1225,635),(215,208,198),1); cv2.putText(canvas,'FAULT MATRIX',(855,155),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'delay · drop · misframe',(855,245),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'restart · stale buffer',(855,300),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'DEGRADE', (855,420),cv2.FONT_HERSHEY_SIMPLEX,.68,amber,2,cv2.LINE_AA); cv2.putText(canvas,'REJECT', (855,490),cv2.FONT_HERSHEY_SIMPLEX,.68,red,2,cv2.LINE_AA); cv2.imwrite(str(OUT/'pipeline_preflight_plot.png'),canvas); (OUT/'pipeline_preflight_report.json').write_text(json.dumps(report,indent=2)+'\n'); print('pipeline preflight lab:',report['status']); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
