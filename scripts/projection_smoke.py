#!/usr/bin/env python3
"""Production-style pinhole projection lab."""
from __future__ import annotations
import argparse, csv, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Camera:
    fx: float; fy: float; cx: float; cy: float

CAMERA = Camera(458.2, 457.9, 319.6, 241.1)
FIXTURE = np.array([
    [-.80,-.45,2.2],[0,-.45,2.2],[.80,-.45,2.2],
    [-.80,0,3.7],[.42,-.08,3.7],[.80,0,3.7],
    [-.80,.45,5.4],[0,.45,5.4],[.80,.45,5.4]], dtype=np.float64)

def project(points: np.ndarray, camera: Camera = CAMERA) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3: raise ValueError("points must have shape (N, 3)")
    if np.any(points[:,2] <= 0): raise ValueError("projection requires positive depth")
    z = points[:,2]
    return np.column_stack((camera.fx*points[:,0]/z+camera.cx, camera.fy*points[:,1]/z+camera.cy))

def normalize(pixels: np.ndarray, camera: Camera = CAMERA) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    return np.column_stack(((pixels[:,0]-camera.cx)/camera.fx, (pixels[:,1]-camera.cy)/camera.fy))

def reprojection_error(points, pixels, camera=CAMERA):
    return np.linalg.norm(project(points, camera)-pixels, axis=1)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', default='out'); args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pixels = project(FIXTURE); normalized = normalize(pixels)
    expected = FIXTURE[:,:2]/FIXTURE[:,2:3]
    normalized_error = np.linalg.norm(normalized-expected, axis=1); assert float(normalized_error.max()) < 1e-12
    bad_camera = Camera(CAMERA.fx*.90, CAMERA.fy*.90, CAMERA.cx, CAMERA.cy)
    bad_error = reprojection_error(FIXTURE, pixels, bad_camera); assert float(np.median(bad_error)) > 1.0
    table = out/'projection_points.csv'
    with table.open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['X','Y','Z','u','v','xn','yn','bad_K_error_px'])
        for point,pixel,norm,err in zip(FIXTURE,pixels,normalized,bad_error):
            writer.writerow([*(round(float(x),6) for x in point), *(round(float(x),6) for x in pixel), *(round(float(x),9) for x in norm), round(float(err),6)])
    report = {'camera': vars(CAMERA), 'fixture_points': int(len(FIXTURE)),
              'max_normalized_roundtrip_error': float(normalized_error.max()),
              'median_bad_intrinsics_error_px': float(np.median(bad_error)),
              'max_good_reprojection_error_px': float(reprojection_error(FIXTURE,pixels).max()),
              'table': str(table), 'status': 'PASS'}
    (out/'projection_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('projection lab: PASS'); print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
