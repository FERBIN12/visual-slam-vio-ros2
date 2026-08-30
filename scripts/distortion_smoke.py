#!/usr/bin/env python3
"""Deterministic radial-distortion and rectification lab."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

K = np.array([[458.2, 0.0, 319.6], [0.0, 457.9, 241.1], [0.0, 0.0, 1.0]])
D = np.array([-0.19, 0.035, 0.0012, -0.0008, 0.0])

def distort(normalized, coeff=D):
    x, y = normalized[:, 0], normalized[:, 1]
    k1, k2, p1, p2, k3 = coeff
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    return np.c_[x*radial + 2*p1*x*y + p2*(r2 + 2*x*x),
                 y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y]

def undistort(observed, coeff=D, iterations=25):
    estimate = observed.copy()
    for _ in range(iterations):
        estimate += observed - distort(estimate, coeff)
    return estimate

def pixels(normalized):
    return normalized * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])

def dense_remap(width=640, height=480):
    """Return destination-to-source maps for an ideal rectified pixel grid."""
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    ideal = np.c_[(uu.ravel() - K[0, 2]) / K[0, 0],
                  (vv.ravel() - K[1, 2]) / K[1, 1]]
    source = pixels(distort(ideal)).reshape(height, width, 2)
    map_x, map_y = source[..., 0].astype(np.float32), source[..., 1].astype(np.float32)
    valid = (map_x >= 0) & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1)
    return map_x, map_y, valid

def stereo_row_check():
    """Rectified horizontal stereo must preserve corresponding image rows."""
    xyz = np.array([(x, y, z) for z in (2.0, 3.5, 6.0)
                    for y in (-0.45, 0.0, 0.4) for x in (-0.7, 0.1, 0.8)])
    baseline = 0.12
    left = xyz[:, :2] / xyz[:, 2, None]
    right_xyz = xyz - np.array([baseline, 0.0, 0.0])
    right = right_xyz[:, :2] / right_xyz[:, 2, None]
    left_rect = pixels(undistort(distort(left)))
    right_rect = pixels(undistort(distort(right)))
    row_error = np.abs(left_rect[:, 1] - right_rect[:, 1])
    crop_bug_error = np.abs(left_rect[:, 1] - (right_rect[:, 1] + 4.0))
    return baseline, row_error, crop_bug_error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/rectification_report.json')
    ap.add_argument('--map-out', default='out/rectification_maps.npz')
    ap.add_argument('--table-out', default='out/distortion_samples.csv')
    args = ap.parse_args()
    u = np.linspace(-0.72, 0.72, 25); v = np.linspace(-0.52, 0.52, 19)
    ideal = np.array([(x, y) for y in v for x in u], dtype=np.float64)
    observed = distort(ideal); recovered = undistort(observed)
    error = np.linalg.norm(recovered - ideal, axis=1)
    bad_coeff = D.copy(); bad_coeff[0] *= -1.0
    bad = undistort(observed, bad_coeff)
    bad_error = np.linalg.norm(bad - ideal, axis=1)
    map_x, map_y, valid = dense_remap()
    baseline, row_error, crop_bug_error = stereo_row_check()
    assert float(error.max()) < 1e-8
    assert float(np.median(bad_error)) > 1e-3
    assert float(row_error.max()) < 1e-8
    assert float(np.median(crop_bug_error)) > 3.9
    map_path = Path(args.map_out); map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(map_path, map_x=map_x, map_y=map_y, valid=valid, K=K, D=D)
    table_path = Path(args.table_out); table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['x_ideal','y_ideal','x_observed','y_observed','round_trip_error','sign_flip_error','radius'])
        for p, q, good_e, bad_e in zip(ideal, observed, error, bad_error):
            writer.writerow([*p, *q, good_e, bad_e, np.linalg.norm(p)])
    report = {'image_size':[640,480], 'samples':len(ideal), 'coefficients':D.tolist(),
              'round_trip_max':float(error.max()), 'round_trip_median':float(np.median(error)),
              'valid_pixel_ratio':float(np.mean(valid)), 'wrong_coeff_median':float(np.median(bad_error)),
              'stereo_baseline_m':baseline, 'rectified_row_max_px':float(row_error.max()),
              'crop_fault_row_median_px':float(np.median(crop_bug_error)),
              'artifacts':{'maps':str(map_path),'samples':str(table_path)},
              'command':'python3 scripts/distortion_smoke.py',
              'status':'PASS'}
    path=Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report,indent=2)+'\n')
    print('distortion lab: PASS'); print(json.dumps(report, indent=2))

if __name__=='__main__': main()
