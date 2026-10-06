#!/usr/bin/env python3
"""Deterministic feature-detection and coverage lab."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

def fixture(size=(480, 640)):
    image = np.zeros(size, np.uint8)
    for y in range(40, size[0]-30, 55):
        for x in range(45, size[1]-30, 70):
            cv2.rectangle(image, (x, y), (x+22, y+22), 210, 2)
    cv2.line(image, (30, 450), (610, 30), 255, 2)
    return image

def detect(image, threshold=20):
    detector = cv2.ORB_create(nfeatures=500, fastThreshold=threshold, edgeThreshold=15)
    keypoints = detector.detect(image, None)
    rows = [[*kp.pt, kp.response, kp.octave, kp.size, kp.angle] for kp in keypoints]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)

def coverage(points, shape, grid=(8, 6)):
    if len(points) == 0: return 0.0
    w, h = shape[1] / grid[0], shape[0] / grid[1]
    cells = {(min(grid[0]-1, int(row[0]/w)), min(grid[1]-1, int(row[1]/h))) for row in points}
    return len(cells) / float(grid[0]*grid[1])

def nearest_neighbor_median(points):
    if len(points) < 2: return 0.0
    xy = points[:, :2]
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.min(distances, axis=1)))

def overlay(image, points, path, grid=(8, 6)):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y, *_ in points:
        cv2.circle(canvas, (round(float(x)), round(float(y))), 3, (255, 180, 30), 1, cv2.LINE_AA)
    for column in range(1, grid[0]):
        x = round(column * image.shape[1] / grid[0]); cv2.line(canvas, (x, 0), (x, image.shape[0]), (70, 70, 70), 1)
    for row in range(1, grid[1]):
        y = round(row * image.shape[0] / grid[1]); cv2.line(canvas, (0, y), (image.shape[1], y), (70, 70, 70), 1)
    if not cv2.imwrite(str(path), canvas): raise RuntimeError(f'failed to write {path}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', default='out/features_report.json')
    ap.add_argument('--points-out', default='out/keypoints.csv')
    ap.add_argument('--overlay-out', default='out/keypoints_overlay.png')
    ap.add_argument('--failure-overlay-out', default='out/keypoints_weak_texture.png'); args=ap.parse_args()
    image=fixture(); shifted=np.roll(image, 2, axis=1); weak=cv2.GaussianBlur(image,(51,51),0)
    points=detect(image); points_shift=detect(shifted); points_weak=detect(weak)
    translated = points[:, :2] + np.array([2.0, 0.0], np.float32)
    if len(translated) and len(points_shift):
        distances = np.linalg.norm(translated[:,None,:]-points_shift[None,:,:2], axis=2)
        repeatability=float(np.mean(np.min(distances,axis=1)<5.0))
    else: repeatability=0.0
    cov=coverage(points,image.shape); weak_cov=coverage(points_weak,image.shape)
    threshold_sweep = {str(value): {'keypoints': int(len(result := detect(image, value))),
                                    'coverage': coverage(result, image.shape)} for value in (20, 150, 200)}
    points_path=Path(args.points_out); points_path.parent.mkdir(parents=True,exist_ok=True)
    with points_path.open('w',newline='') as fh:
        writer=csv.writer(fh); writer.writerow(['x_px','y_px','response','octave','size_px','angle_deg']); writer.writerows(points)
    overlay_path=Path(args.overlay_out); failure_path=Path(args.failure_overlay_out)
    overlay_path.parent.mkdir(parents=True,exist_ok=True); failure_path.parent.mkdir(parents=True,exist_ok=True)
    overlay(image,points,overlay_path); overlay(weak,points_weak,failure_path)
    assert len(points) > 50 and cov > .35 and repeatability > .75
    assert len(points_weak) < len(points) * .1 and weak_cov < .2
    report={'detector':'ORB/FAST', 'keypoints':int(len(points)), 'coverage':cov,
            'repeatability_shift_2px':repeatability, 'nearest_neighbor_median_px':nearest_neighbor_median(points),
            'weak_texture_keypoints':int(len(points_weak)), 'weak_texture_coverage':weak_cov,
            'threshold_sweep':threshold_sweep,
            'artifacts':{'coordinates':str(points_path),'overlay':str(overlay_path),'failure_overlay':str(failure_path)},
            'command':'python3 scripts/features_smoke.py', 'status':'PASS'}
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2)+'\n')
    print('features lab: PASS'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
