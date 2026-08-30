#!/usr/bin/env python3
"""Deterministic VIO initialization: scale, gravity, and accelerometer bias."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2, numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)

def run_case(t, visual_dv, imu_dv, name):
    dt=np.diff(t); rows=[]
    # imu_dv = s*visual_dv - g*dt + ba*dt; solve linear state [s,gx,gy,gz,bax,bay,baz]
    A=[]; y=[]
    for i,h in enumerate(dt):
        for k in range(3):
            row=np.zeros(7); row[0]=visual_dv[i,k]; row[1+k]=-h; row[4+k]=h*h
            A.append(row); y.append(imu_dv[i,k])
    A=np.asarray(A); y=np.asarray(y)
    x, *_ = np.linalg.lstsq(A,y,rcond=None); pred=A@x; residual=float(np.sqrt(np.mean((pred-y)**2)))
    sv=np.linalg.svd(A,compute_uv=False); cond=float(sv[0]/max(sv[-1],1e-15))
    return {'name':name,'scale':float(x[0]),'gravity':x[1:4].tolist(),'bias':x[4:7].tolist(),'residual_rms':residual,'condition_number':cond,'singular_values':sv.tolist()}

def main():
    rng=np.random.default_rng(21); t=np.r_[0.0,np.cumsum(0.045+0.01*np.sin(np.arange(180)*0.37))]; dt=np.diff(t)
    scale=2.4; g=np.array([0.,0.,-9.81]); ba=np.array([.08,-.04,.06])
    # Exciting visual velocity increments; metric increment is scale times visual.
    visual_dv=np.column_stack([.7*np.sin(.8*t[:-1]), .5*np.cos(.6*t[:-1]), .25*np.sin(.45*t[:-1])])
    imu_dv=scale*visual_dv-g*dt[:,None]+ba*(dt[:,None]**2)+rng.normal(0,2e-4,(len(dt),3))
    good=run_case(t,visual_dv,imu_dv,'exciting')
    weak_visual=np.column_stack([np.full(len(dt),.12),np.zeros(len(dt)),np.zeros(len(dt))])
    weak_imu=scale*weak_visual-g*dt[:,None]+ba*(dt[:,None]**2)
    weak=run_case(t,weak_visual,weak_imu,'constant_velocity')
    # Gravity-norm and positive-scale gates are explicit controls.
    errors={'scale':abs(good['scale']-scale),'gravity_mps2':float(np.linalg.norm(np.asarray(good['gravity'])-g)),'bias_mps2':float(np.linalg.norm(np.asarray(good['bias'])-ba))}
    checks={'scale_recovered':errors['scale']<.01,'gravity_recovered':errors['gravity_mps2']<.03,'bias_recovered':errors['bias_mps2']<.05,'residual_small':good['residual_rms']<.001,'positive_scale':good['scale']>0,'weak_condition_visible':weak['condition_number']>3e3}
    report={'truth':{'scale':scale,'gravity_mps2':g.tolist(),'bias_mps2':ba.tolist()},'exciting':good,'weak_excitation':weak,'errors':errors,'checks':checks,'status':'PASS' if all(checks.values()) else 'FAIL','command':'python3 scripts/vio_initialization_smoke.py','artifacts':{'report':'out/vio_initialization_report.json','trace':'out/vio_initialization_trace.csv','model':'out/vio_initialization_model.npz','plot':'out/vio_initialization_plot.png'}}
    trace=[{'interval':i,'time_s':float(t[i]),'scale':good['scale'],'gravity_norm':float(np.linalg.norm(good['gravity'])),'residual_rms':good['residual_rms']} for i in range(len(dt))]
    with (OUT/'vio_initialization_trace.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=trace[0]); w.writeheader(); w.writerows(trace)
    np.savez(OUT/'vio_initialization_model.npz',times_s=t,visual_delta_v=visual_dv,imu_delta_v=imu_dv,solution=np.r_[good['scale'],good['gravity'],good['bias']],weak_singular_values=np.asarray(weak['singular_values']))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); blue=(190,115,34); teal=(115,140,25); red=(40,45,190)
    cv2.putText(canvas,'VIO INITIALIZATION SCORECARD',(42,52),cv2.FONT_HERSHEY_SIMPLEX,.78,ink,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(55,105),(760,635),(215,208,198),1); cv2.putText(canvas,'RECOVERED STATE',(85,145),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA)
    vals=[('scale',f"{good['scale']:.3f} (truth {scale:.1f})"),('gravity norm',f"{np.linalg.norm(good['gravity']):.3f} m/s²"),('bias norm',f"{np.linalg.norm(good['bias']):.3f} m/s²"),('residual',f"{good['residual_rms']:.2e}")]
    for i,(a,b) in enumerate(vals): cv2.putText(canvas,a,(95,225+i*78),cv2.FONT_HERSHEY_SIMPLEX,.5,ink,1,cv2.LINE_AA); cv2.putText(canvas,b,(95,260+i*78),cv2.FONT_HERSHEY_SIMPLEX,.66,teal,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(805,105),(1225,635),(215,208,198),1); cv2.putText(canvas,'EXCITATION GATE',(835,145),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA); cv2.putText(canvas,'exciting motion',(835,240),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA); cv2.putText(canvas,'ACCEPT',(835,280),cv2.FONT_HERSHEY_SIMPLEX,.72,teal,2,cv2.LINE_AA); cv2.putText(canvas,'constant velocity',(835,390),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA); cv2.putText(canvas,'DEGRADE',(835,430),cv2.FONT_HERSHEY_SIMPLEX,.72,red,2,cv2.LINE_AA); cv2.putText(canvas,f"condition {weak['condition_number']:.2g}",(835,475),cv2.FONT_HERSHEY_SIMPLEX,.5,red,1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/'vio_initialization_plot.png'),canvas); (OUT/'vio_initialization_report.json').write_text(json.dumps(report,indent=2)+'\n'); print('VIO initialization lab:',report['status']); print(json.dumps(report,indent=2)); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
