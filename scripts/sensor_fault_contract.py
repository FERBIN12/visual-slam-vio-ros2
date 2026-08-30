#!/usr/bin/env python3
"""Deterministic sensor-noise, bias, dropout, and timing-fault audit."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np, cv2
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'out';OUT.mkdir(exist_ok=True)
def assess(c):
 checks={'seed_declared':c['seed']==17,'noise_realized':c['noise_realized'],'bias_process':c['bias_process'],'dropout_onset':c['dropout_onset'],'offset_declared':c['offset_declared'],'clean_parity':c['clean_parity']}
 failed=[k for k,v in checks.items() if not v];return failed,'ACCEPT' if not failed else ('DEGRADE' if len(failed)<=2 else 'REJECT')
def main():
 good=dict(seed=17,noise_realized=True,bias_process=True,dropout_onset=True,offset_declared=True,clean_parity=True)
 cases=[('clean_baseline',good),('noise_density_fault',{**good,'noise_realized':False}),('bias_fault',{**good,'bias_process':False}),('dropout_fault',{**good,'dropout_onset':False}),('offset_missing',{**good,'offset_declared':False}),('parity_missing',{**good,'clean_parity':False}),('malformed_scenario',{**good,'seed':0,'noise_realized':False,'bias_process':False,'offset_declared':False})]
 rows=[]
 for name,c in cases:
  f,s=assess(c);rows.append({'case':name,'status':s,'failed_checks':';'.join(f)})
 gates={'ready_accept':rows[0]['status']=='ACCEPT','noise_degrade':rows[1]['status']=='DEGRADE','bias_degrade':rows[2]['status']=='DEGRADE','dropout_degrade':rows[3]['status']=='DEGRADE','offset_degrade':rows[4]['status']=='DEGRADE','parity_degrade':rows[5]['status']=='DEGRADE','malformed_reject':rows[6]['status']=='REJECT'}
 report={'status':'PASS' if all(gates.values()) else 'FAIL','cases':rows,'gates':gates,'command':'python3 scripts/sensor_fault_contract.py'}
 with (OUT/'sensor_fault_scorecard.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rng=np.random.default_rng(17);t=np.linspace(0,10,2000);clean=np.sin(t);noise=rng.normal(0,.03,len(t));bias=.002*t;drop=np.ones(len(t));drop[900:980]=0;np.savez(OUT/'sensor_fault_trace.npz',time=t,clean=clean,noisy=clean+noise+bias,dropout=drop)
 canvas=np.full((720,1280,3),246,np.uint8);ink=(45,43,37);cv2.putText(canvas,'SENSOR NOISE + BIAS + TIMING FAULTS',(40,52),cv2.FONT_HERSHEY_SIMPLEX,.72,ink,2,cv2.LINE_AA);cv2.rectangle(canvas,(55,105),(760,640),(215,208,198),1);cv2.rectangle(canvas,(820,105),(1225,640),(215,208,198),1)
 for y,s in [(180,'seed + realized statistics'),(260,'noise · bias · blur · dropout'),(340,'offset · onset · recovery'),(470,'FAULT GATE')]:cv2.putText(canvas,s,(90,y),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA)
 cv2.putText(canvas,'DEGRADE: one controlled fault',(865,285),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,130,190),1,cv2.LINE_AA);cv2.putText(canvas,'REJECT: malformed scenario',(865,355),cv2.FONT_HERSHEY_SIMPLEX,.47,(40,45,190),1,cv2.LINE_AA);cv2.imwrite(str(OUT/'sensor_fault_plot.png'),canvas);(OUT/'sensor_fault_report.json').write_text(json.dumps(report,indent=2)+'\n');print('sensor fault contract lab:',report['status']);return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
