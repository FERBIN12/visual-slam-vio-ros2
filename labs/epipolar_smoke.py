#!/usr/bin/env python3
"""Essential-matrix and Sampson-residual lab."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

K=np.array([[458.2,0,319.6],[0,457.9,241.1],[0,0,1]],np.float64)

def skew(t):
    return np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]],np.float64)

def sampson_distance(F,a,b):
    x1=np.c_[a,np.ones(len(a))]; x2=np.c_[b,np.ones(len(b))]
    Fx1=(F@x1.T).T; Ftx2=(F.T@x2.T).T
    numerator=np.sum(x2*Fx1,axis=1)**2
    denominator=Fx1[:,0]**2+Fx1[:,1]**2+Ftx2[:,0]**2+Ftx2[:,1]**2+1e-12
    return np.sqrt(numerator/denominator)

def median_parallax_deg(points,R,t):
    first=points/np.linalg.norm(points,axis=1)[:,None]
    second_points=(R@points.T+t[:,None]).T
    second=second_points/np.linalg.norm(second_points,axis=1)[:,None]
    second_in_first=(R.T@second.T).T
    angles=np.arccos(np.clip(np.sum(first*second_in_first,axis=1),-1.0,1.0))
    return float(np.degrees(np.median(angles)))

def overlay(F,a,b,inliers,path):
    canvas=np.full((480,640,3),245,np.uint8)
    indices=np.r_[np.flatnonzero(inliers)[:24],np.flatnonzero(~inliers)[:15]]
    for index in indices:
        line=F@np.r_[a[index],1.0]; color=(40,170,60) if inliers[index] else (40,40,220)
        if abs(line[1]) > 1e-9:
            p0=(0,round(float(-line[2]/line[1]))); p1=(639,round(float(-(line[2]+line[0]*639)/line[1])))
        else:
            x=round(float(-line[2]/line[0])); p0=(x,0); p1=(x,479)
        cv2.line(canvas,p0,p1,color,1,cv2.LINE_AA)
        cv2.circle(canvas,tuple(round(float(v)) for v in b[index]),4,color,-1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),canvas): raise RuntimeError(f'failed to write {path}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='out/epipolar_report.json')
    ap.add_argument('--matches-out',default='out/epipolar_matches.csv'); ap.add_argument('--model-out',default='out/epipolar_model.npz')
    ap.add_argument('--overlay-out',default='out/epipolar_overlay.png'); ap.add_argument('--failure-overlay-out',default='out/epipolar_wrong_model.png'); args=ap.parse_args()
    rng=np.random.default_rng(12); points=np.c_[rng.uniform(-1.2,1.2,120),rng.uniform(-.8,.8,120),rng.uniform(2.5,8.0,120)]
    rvec=np.array([0.01,-0.04,0.02]); R,_=cv2.Rodrigues(rvec); t=np.array([.16,.01,.02])
    def project(P): return (K@(P.T/P[:,2])).T[:,:2]
    a=project(points); b=project((R@points.T+t[:,None]).T)
    a+=rng.normal(0,.25,a.shape); b+=rng.normal(0,.25,b.shape); b[-15:]=rng.uniform([0,0],[639,479],(15,2))
    E,mask=cv2.findEssentialMat(a,b,K,method=cv2.RANSAC,prob=0.999,threshold=1.0)
    if E is None: raise AssertionError('essential matrix estimation failed')
    if E.shape!=(3,3): E=E[:3,:3]
    F=np.linalg.inv(K).T@E@np.linalg.inv(K); sampson=sampson_distance(F,a,b)
    inliers=sampson<1.0; inlier_errors=sampson[inliers]; ransac_mask=mask.ravel().astype(bool)
    wrong_E=skew(np.array([0.0,.16,.02]))@R; wrong_F=np.linalg.inv(K).T@wrong_E@np.linalg.inv(K)
    wrong_sampson=sampson_distance(wrong_F,a,b); clean=np.arange(len(a))<len(a)-15
    forward_points=np.c_[rng.uniform(-.18,.18,105),rng.uniform(-.12,.12,105),rng.uniform(4.0,8.0,105)]
    trusted_parallax=median_parallax_deg(points[:-15],R,t)
    forward_parallax=median_parallax_deg(forward_points,np.eye(3),np.array([0.0,0.0,.16]))
    matches_path=Path(args.matches_out); matches_path.parent.mkdir(parents=True,exist_ok=True)
    with matches_path.open('w',newline='') as fh:
        writer=csv.writer(fh); writer.writerow(['x1_px','y1_px','x2_px','y2_px','sampson_px','ransac_inlier','sampson_inlier','injected_outlier'])
        for i,(p1,p2,error) in enumerate(zip(a,b,sampson)):
            writer.writerow([*p1,*p2,error,int(ransac_mask[i]),int(inliers[i]),int(not clean[i])])
    model_path=Path(args.model_out); model_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(model_path,K=K,E=E,F=F,R=R,t=t,ransac_mask=ransac_mask,sampson_inliers=inliers)
    overlay_path=Path(args.overlay_out); failure_path=Path(args.failure_overlay_out)
    overlay_path.parent.mkdir(parents=True,exist_ok=True); failure_path.parent.mkdir(parents=True,exist_ok=True)
    overlay(F,a,b,inliers,overlay_path); overlay(wrong_F,a,b,wrong_sampson<1.0,failure_path)
    injected_rejected=int(np.sum(~inliers[-15:])); clean_accepted=int(np.sum(inliers[:-15]))
    wrong_median=float(np.median(wrong_sampson[clean]))
    assert int(inliers.sum()) > 85 and float(np.median(inlier_errors)) < .4
    assert injected_rejected>=13 and clean_accepted>90 and wrong_median>5.0 and forward_parallax<.1
    report={'matches':len(a),'ransac_inliers':int(ransac_mask.sum()),'sampson_inliers':int(inliers.sum()),
            'median_sampson_px':float(np.median(inlier_errors)), 'p95_sampson_px':float(np.percentile(inlier_errors,95)),
            'outliers_injected':15,'injected_outliers_rejected':injected_rejected,'clean_matches_accepted':clean_accepted,
            'wrong_model_median_sampson_px':wrong_median,
            'trusted_median_parallax_deg':trusted_parallax,'forward_motion_median_parallax_deg':forward_parallax,
            'threshold_px':1.0,
            'artifacts':{'matches':str(matches_path),'model':str(model_path),'overlay':str(overlay_path),'failure_overlay':str(failure_path)},
            'command':'python3 scripts/epipolar_smoke.py','status':'PASS'}
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2)+'\n')
    print('epipolar lab: PASS'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
