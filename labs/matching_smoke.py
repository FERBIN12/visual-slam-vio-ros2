#!/usr/bin/env python3
"""Descriptor matching lab: ratio test, mutual check, and false-match gate."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

def image(seed=7):
    rng=np.random.default_rng(seed); base=(rng.random((420,640))*255).astype(np.uint8)
    base=cv2.GaussianBlur(base,(5,5),0); base[::35,:]=255; base[:,::43]=255
    return base

def repeated_image(seed=3):
    rng=np.random.default_rng(seed); patch=(rng.random((60,64))*255).astype(np.uint8)
    patch=cv2.GaussianBlur(patch,(3,3),0)
    return np.tile(patch,(7,10))[:420,:640]

def match(left,right,expected_shift=8.0,ratio_threshold=.78):
    orb=cv2.ORB_create(nfeatures=600,edgeThreshold=15)
    kp1,d1=orb.detectAndCompute(left,None); kp2,d2=orb.detectAndCompute(right,None)
    if d1 is None or d2 is None: raise RuntimeError('descriptor extraction returned no features')
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    forward=matcher.knnMatch(d1,d2,k=2); reverse=matcher.knnMatch(d2,d1,k=2)
    ratio=[pair[0] for pair in forward if len(pair)==2 and pair[0].distance < ratio_threshold*pair[1].distance]
    reverse_best={pair[0].queryIdx:pair[0].trainIdx for pair in reverse
                  if len(pair)==2 and pair[0].distance < ratio_threshold*pair[1].distance}
    mutual=[candidate for candidate in ratio if reverse_best.get(candidate.trainIdx)==candidate.queryIdx]
    distances=np.array([candidate.distance for candidate in mutual],dtype=np.float64)
    geometry=np.array([np.linalg.norm((np.array(kp2[m.trainIdx].pt)-np.array(kp1[m.queryIdx].pt))
                                      -np.array([expected_shift,0.0])) for m in mutual])
    inliers=geometry < 3.0
    return kp1,kp2,forward,ratio,mutual,distances,geometry,inliers

def overlay(left,right,kp1,kp2,matches,inliers,path):
    canvas=np.concatenate([cv2.cvtColor(left,cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(right,cv2.COLOR_GRAY2BGR)],axis=1)
    for candidate,good in zip(matches,inliers):
        a=tuple(round(v) for v in kp1[candidate.queryIdx].pt)
        b=(round(kp2[candidate.trainIdx].pt[0])+left.shape[1],round(kp2[candidate.trainIdx].pt[1]))
        color=(40,210,80) if good else (40,40,230)
        cv2.circle(canvas,a,2,color,-1,cv2.LINE_AA); cv2.circle(canvas,b,2,color,-1,cv2.LINE_AA)
        cv2.line(canvas,a,b,color,1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),canvas): raise RuntimeError(f'failed to write {path}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='out/matches_report.json')
    ap.add_argument('--matches-out',default='out/matches.csv'); ap.add_argument('--overlay-out',default='out/matches_overlay.png')
    ap.add_argument('--failure-overlay-out',default='out/matches_repeated_texture.png'); args=ap.parse_args()
    left=image(); right=np.roll(left,8,axis=1)
    kp1,kp2,forward,ratio,mutual,distances,geometry,inliers=match(left,right)
    repeated=repeated_image(); repeated_right=np.roll(repeated,8,axis=1)
    rkp1,rkp2,rforward,rratio,rmutual,rdistances,rgeometry,rinliers=match(repeated,repeated_right)
    matches_path=Path(args.matches_out); matches_path.parent.mkdir(parents=True,exist_ok=True)
    with matches_path.open('w',newline='') as fh:
        writer=csv.writer(fh); writer.writerow(['query_x_px','query_y_px','train_x_px','train_y_px','hamming','geometry_error_px','inlier'])
        for candidate,error,good in zip(mutual,geometry,inliers):
            writer.writerow([*kp1[candidate.queryIdx].pt,*kp2[candidate.trainIdx].pt,candidate.distance,error,int(good)])
    overlay_path=Path(args.overlay_out); failure_path=Path(args.failure_overlay_out)
    overlay_path.parent.mkdir(parents=True,exist_ok=True); failure_path.parent.mkdir(parents=True,exist_ok=True)
    overlay(left,right,kp1,kp2,mutual,inliers,overlay_path)
    overlay(repeated,repeated_right,rkp1,rkp2,rmutual,rinliers,failure_path)
    trusted_inlier=float(np.mean(inliers)); repeated_inlier=float(np.mean(rinliers))
    assert len(kp1)>100 and len(mutual)>30 and float(np.median(distances)) < 45
    assert trusted_inlier > .95 and len(rmutual)>30 and repeated_inlier < .2
    report={'detector':'ORB', 'keypoints_left':len(kp1), 'keypoints_right':len(kp2),
            'raw_candidates':len(forward), 'ratio_survivors':len(ratio),
            'mutual_survivors':len(mutual), 'median_hamming':float(np.median(distances)),
            'geometry_inlier_ratio':trusted_inlier, 'geometry_error_median_px':float(np.median(geometry)),
            'repeated_texture':{'raw_candidates':len(rforward),'ratio_survivors':len(rratio),
                                'mutual_survivors':len(rmutual),'median_hamming':float(np.median(rdistances)),
                                'geometry_inlier_ratio':repeated_inlier,
                                'geometry_error_median_px':float(np.median(rgeometry))},
            'ratio_threshold':.78,'expected_shift_px':[8.0,0.0],
            'artifacts':{'matches':str(matches_path),'overlay':str(overlay_path),'failure_overlay':str(failure_path)},
            'command':'python3 scripts/matching_smoke.py','status':'PASS'}
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2)+'\n')
    print('matching lab: PASS'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
