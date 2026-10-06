#!/usr/bin/env python3
"""Two-view triangulation, cheirality, and reprojection lab."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

K=np.array([[458.2,0,319.6],[0,457.9,241.1],[0,0,1]],np.float64)

def project(P, X):
    q=(P@np.c_[X,np.ones(len(X))].T).T; return q[:,:2]/q[:,2:3]

def triangulate(P1,P2,u1,u2):
    homogeneous=cv2.triangulatePoints(P1,P2,u1.T,u2.T).T
    return homogeneous[:,:3]/homogeneous[:,3:4]

def write_ply(path,points):
    path.write_text('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n'%len(points)+''.join(f'{x:.6f} {y:.6f} {z:.6f}\n' for x,y,z in points))

def overlay(u1,accepted,path):
    canvas=np.full((480,640,3),245,np.uint8)
    for point,good in zip(u1,accepted):
        color=(40,170,60) if good else (40,40,220)
        cv2.circle(canvas,tuple(round(float(v)) for v in point),4,color,-1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),canvas): raise RuntimeError(f'failed to write {path}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='out/point_cloud_report.json')
    ap.add_argument('--points-out',default='out/triangulated_points.csv'); ap.add_argument('--model-out',default='out/triangulation_model.npz')
    ap.add_argument('--cloud-out',default='out/cloud.ply'); ap.add_argument('--overlay-out',default='out/triangulation_overlay.png'); args=ap.parse_args()
    rng=np.random.default_rng(21); X=np.c_[rng.uniform(-1,1,100),rng.uniform(-.7,.7,100),rng.uniform(2.0,8.0,100)]
    R=np.eye(3); t=np.array([.18,0,0.]); P1=K@np.c_[np.eye(3),np.zeros(3)]; P2=K@np.c_[R,t]
    clean_u1=project(P1,X)+rng.normal(0,.08,(len(X),2)); clean_u2=project(P2,X)+rng.normal(0,.08,(len(X),2))
    fault_rng=np.random.default_rng(99); injected_u2=fault_rng.uniform([0,0],[639,479],(10,2))
    u1=np.r_[clean_u1,clean_u1[:10]]; u2=np.r_[clean_u2,injected_u2]
    labels=np.array(['clean']*100+['injected_outlier']*10)
    Xr=triangulate(P1,P2,u1,u2)
    reproj1=np.linalg.norm(project(P1,Xr)-u1,axis=1); reproj2=np.linalg.norm(project(P2,Xr)-u2,axis=1)
    reproj_max=np.maximum(reproj1,reproj2)
    depth2=(R@Xr.T+t[:,None]).T[:,2]; positive=(Xr[:,2]>0)&(depth2>0)
    parallax=np.degrees(np.arctan2(abs(t[0]),np.abs(Xr[:,2])))
    accepted=positive&(reproj_max<.5)&(parallax>.5)
    # Tiny baseline: reprojection can remain small while metric depth becomes unstable.
    tiny_t=np.array([.002,0,0.]); tiny_P2=K@np.c_[R,tiny_t]
    tiny_u1=project(P1,X)+rng.normal(0,.08,(len(X),2)); tiny_u2=project(tiny_P2,X)+rng.normal(0,.08,(len(X),2))
    tiny_X=triangulate(P1,tiny_P2,tiny_u1,tiny_u2)
    tiny_reproj=np.maximum(np.linalg.norm(project(P1,tiny_X)-tiny_u1,axis=1),
                           np.linalg.norm(project(tiny_P2,tiny_X)-tiny_u2,axis=1))
    trusted_depth_error=np.abs(Xr[:100,2]-X[:,2])/X[:,2]
    tiny_depth_error=np.abs(tiny_X[:,2]-X[:,2])/X[:,2]
    tiny_parallax=np.degrees(np.arctan2(abs(tiny_t[0]),np.abs(tiny_X[:,2])))
    points_path=Path(args.points_out); points_path.parent.mkdir(parents=True,exist_ok=True)
    with points_path.open('w',newline='') as fh:
        writer=csv.writer(fh); writer.writerow(['x1_px','y1_px','x2_px','y2_px','X_m','Y_m','Z1_m','Z2_m','reprojection1_px','reprojection2_px','parallax_deg','positive_depth','label','accepted'])
        for p1,p2,p3,z2,r1,r2,angle,pos,label,good in zip(u1,u2,Xr,depth2,reproj1,reproj2,parallax,positive,labels,accepted):
            writer.writerow([*p1,*p2,*p3,z2,r1,r2,angle,int(pos),label,int(good)])
    model_path=Path(args.model_out); model_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(model_path,K=K,P1=P1,P2=P2,R=R,t=t,points=Xr,accepted=accepted,
                        tiny_P2=tiny_P2,tiny_points=tiny_X)
    ply=Path(args.cloud_out); ply.parent.mkdir(parents=True,exist_ok=True); write_ply(ply,Xr[accepted])
    overlay_path=Path(args.overlay_out); overlay_path.parent.mkdir(parents=True,exist_ok=True); overlay(u1,accepted,overlay_path)
    clean_accepted=int(np.sum(accepted[:100])); outliers_rejected=int(np.sum(~accepted[100:]))
    accepted_reprojection=reproj_max[accepted]
    assert clean_accepted==100 and outliers_rejected==10 and float(np.median(accepted_reprojection))<.3
    assert float(np.median(tiny_depth_error))>.2 and float(np.median(tiny_parallax))<.1
    report={'matches':len(u1),'clean_matches':100,'outliers_injected':10,'accepted_points':int(accepted.sum()),
            'clean_points_accepted':clean_accepted,'injected_outliers_rejected':outliers_rejected,
            'positive_depth':int(np.sum(positive)),'depth_range_m':[float(Xr[accepted,2].min()),float(Xr[accepted,2].max())],
            'median_reprojection_px':float(np.median(accepted_reprojection)),'p95_reprojection_px':float(np.percentile(accepted_reprojection,95)),
            'parallax_p50_deg':float(np.percentile(parallax[accepted],50)),
            'trusted_median_relative_depth_error':float(np.median(trusted_depth_error)),
            'outlier_median_reprojection_px':float(np.median(reproj_max[100:])),
            'tiny_baseline':{'baseline_m':float(tiny_t[0]),'median_reprojection_px':float(np.median(tiny_reproj)),
                             'median_relative_depth_error':float(np.median(tiny_depth_error)),
                             'parallax_p50_deg':float(np.median(tiny_parallax))},
            'artifacts':{'point_cloud':str(ply),'points':str(points_path),'model':str(model_path),'overlay':str(overlay_path)},
            'command':'python3 scripts/triangulation_smoke.py','status':'PASS'}
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2)+'\n')
    print('triangulation lab: PASS'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
