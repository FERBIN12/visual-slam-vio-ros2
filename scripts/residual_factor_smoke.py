#!/usr/bin/env python3
"""Deterministic visual/IMU residual, Jacobian, whitening, and outlier lab."""
from __future__ import annotations
import json
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)

def project(x,K):
    return np.array([K[0,0]*x[0]/x[2]+K[0,2],K[1,1]*x[1]/x[2]+K[1,2]])
def main():
    K=np.array([[420.,0,320],[0,420,240],[0,0,1.]])
    point=np.array([.35,-.12,4.2]); obs=project(point,K); state=np.array([0.,0.,0.,.04,-.01,.02])
    def residual(s):
        p=point+s[:3]; return np.r_[project(p,K)-obs, s[3:]-np.array([.04,-.01,.02])]
    r0=residual(state); eps=1e-6; J=np.column_stack([(residual(state+eps*np.eye(6)[i])-residual(state-eps*np.eye(6)[i]))/(2*eps) for i in range(6)])
    cov=np.diag([4.,4., .01,.01,.01]); W=np.diag(1/np.sqrt(np.diag(cov))); whiten=W@r0; whiten_norm=float(np.linalg.norm(whiten)); q=point+state[:3]; expected=np.array([[K[0,0]/q[2],0,-K[0,0]*q[0]/q[2]**2],[0,K[1,1]/q[2],-K[1,1]*q[1]/q[2]**2]]); fd=float(np.max(np.abs(J[:2,:3]-expected)))
    outlier=np.r_[r0[:2]+np.array([90.,-70.]),r0[2:]]; plain=float(np.linalg.norm(outlier)); huber_delta=10.; robust=float(np.sqrt(np.sum(np.minimum(outlier**2,huber_delta**2))))
    checks={k:bool(v) for k,v in {'zero_residual':whiten_norm<.1,'finite_difference_jacobian':fd<1e-4,'whitening_finite':np.isfinite(whiten).all(),'outlier_visible':plain>50 and robust<plain}.items()}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','truth':{'observation_px':obs.tolist()},'residual':r0.tolist(),'jacobian_max_error':fd,'whitened_norm':whiten_norm,'outlier':{'plain_norm':plain,'robust_norm':robust},'checks':checks,'command':'python3 scripts/residual_factor_smoke.py','artifacts':{'report':'out/residual_factor_report.json','model':'out/residual_factor_model.npz','plot':'out/residual_factor_plot.png'}}
    np.savez(OUT/'residual_factor_model.npz',K=K,point=point,residual=r0,jacobian=J,whitened=whiten)
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); teal=(115,140,25); red=(40,45,190)
    cv2.putText(canvas,'VISUAL + IMU RESIDUAL SCORECARD',(42,52),cv2.FONT_HERSHEY_SIMPLEX,.76,ink,2,cv2.LINE_AA); cv2.rectangle(canvas,(60,110),(760,635),(215,208,198),1); cv2.putText(canvas,'FACTOR CHECKS',(90,150),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA)
    vals=[('zero residual',f'{whiten_norm:.2e}'),('FD Jacobian',f'{fd:.2e}'),('plain outlier',f'{plain:.1f}'),('robust outlier',f'{robust:.1f}')]
    for i,(a,b) in enumerate(vals): cv2.putText(canvas,a,(100,225+i*78),cv2.FONT_HERSHEY_SIMPLEX,.5,ink,1,cv2.LINE_AA); cv2.putText(canvas,b,(100,260+i*78),cv2.FONT_HERSHEY_SIMPLEX,.66,teal if i<2 else red,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(810,110),(1220,635),(215,208,198),1); cv2.putText(canvas,'AUTHORITY',(850,150),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA); cv2.putText(canvas,'WHITENED FACTOR',(850,255),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'PASS',(850,300),cv2.FONT_HERSHEY_SIMPLEX,.78,teal,2,cv2.LINE_AA); cv2.putText(canvas,'ROBUST OUTLIER',(850,420),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'VISIBLE',(850,465),cv2.FONT_HERSHEY_SIMPLEX,.78,red,2,cv2.LINE_AA)
    cv2.imwrite(str(OUT/'residual_factor_plot.png'),canvas); (OUT/'residual_factor_report.json').write_text(json.dumps(report,indent=2)+'\n'); print('Residual factor lab:',report['status']); print(json.dumps(report,indent=2)); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
