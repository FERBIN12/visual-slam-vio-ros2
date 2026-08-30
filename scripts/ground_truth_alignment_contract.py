#!/usr/bin/env python3
"""Deterministic ground-truth association and coordinate-alignment audit."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'out';OUT.mkdir(exist_ok=True)
def assess(c):
 checks={'association_count':c['association_count']>=8,'time_skew':c['time_skew']<=0.01,'body_point_declared':bool(c['body_point']),'transform_direction':c['transform_direction'] in {'T_est_truth','T_truth_est'},'scale_policy':c['scale_policy'] in {'SE3','Sim3'},'identity_fixture':c['identity_fixture']}
 f=[k for k,v in checks.items() if not v];return f,'ACCEPT' if not f else ('DEGRADE' if len(f)<=2 else 'REJECT')
def main():
 good=dict(association_count=10,time_skew=0.002,body_point='base_link',transform_direction='T_est_truth',scale_policy='SE3',identity_fixture=True)
 cases=[('aligned_ready',good),('few_associations',{**good,'association_count':3}),('time_skew',{**good,'time_skew':0.08}),('body_point_missing',{**good,'body_point':''}),('direction_missing',{**good,'transform_direction':''}),('sim3_policy',{**good,'scale_policy':'Sim3'}),('malformed_pair',{**good,'association_count':1,'time_skew':0.2,'body_point':'','transform_direction':'','identity_fixture':False})]
 rows=[]
 for n,c in cases:f,s=assess(c);rows.append({'case':n,'status':s,'failed_checks':';'.join(f)})
 gates={'ready_accept':rows[0]['status']=='ACCEPT','count_degrade':rows[1]['status']=='DEGRADE','skew_degrade':rows[2]['status']=='DEGRADE','body_degrade':rows[3]['status']=='DEGRADE','direction_degrade':rows[4]['status']=='DEGRADE','sim3_accept':rows[5]['status']=='ACCEPT','malformed_reject':rows[6]['status']=='REJECT'}
 report={'status':'PASS' if all(gates.values()) else 'FAIL','cases':rows,'gates':gates,'command':'python3 scripts/ground_truth_alignment_contract.py'}
 with (OUT/'ground_truth_alignment_scorecard.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 t=np.linspace(0,9,10);truth=np.c_[t,np.sin(t),np.cos(t)];est=truth+np.array([0.02,-0.01,0.01]);np.savez(OUT/'ground_truth_alignment_trace.npz',timestamp=t,truth=truth,estimate=est,association=np.arange(10))
 canvas=np.full((720,1280,3),246,np.uint8);ink=(45,43,37);cv2.putText(canvas,'GROUND TRUTH + COORDINATE ALIGNMENT',(40,52),cv2.FONT_HERSHEY_SIMPLEX,.72,ink,2,cv2.LINE_AA);cv2.rectangle(canvas,(55,105),(760,640),(215,208,198),1);cv2.rectangle(canvas,(820,105),(1225,640),(215,208,198),1)
 for y,s in [(180,'timestamps + associations'),(260,'body point + TF direction'),(340,'SE3 / Sim3 policy'),(470,'ALIGNMENT GATE')]:cv2.putText(canvas,s,(90,y),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA)
 cv2.putText(canvas,'DEGRADE: skew / sparse pairs',(865,285),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,130,190),1,cv2.LINE_AA);cv2.putText(canvas,'REJECT: malformed convention',(865,355),cv2.FONT_HERSHEY_SIMPLEX,.47,(40,45,190),1,cv2.LINE_AA);cv2.imwrite(str(OUT/'ground_truth_alignment_plot.png'),canvas);(OUT/'ground_truth_alignment_report.json').write_text(json.dumps(report,indent=2)+'\n');print('ground truth alignment lab:',report['status']);return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
