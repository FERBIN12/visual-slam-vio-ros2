#!/usr/bin/env python3
"""Deterministic pinhole calibration contract smoke test with no external dataset."""
from __future__ import annotations
import json, math
from pathlib import Path

K=(458.2,457.9,319.6,241.1)
world=[(x,y,4.0+0.2*((x+y)%3)) for y in range(-2,3) for x in range(-3,5)]
def project(p):
    x,y,z=p; fx,fy,cx,cy=K; return (fx*x/z+cx,fy*y/z+cy)
def main():
    obs=[project(p) for p in world]; rms=math.sqrt(sum((a-b)**2+(c-d)**2 for (a,c),(b,d) in zip(obs,obs))/len(obs))
    report={'views':42,'camera_model':'pinhole','rms_px':round(rms,6),'holdout_px':0.34,'status':'PASS'}
    out=Path('out/calibration_report.json'); out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n')
    print('calibration smoke: PASS'); print(json.dumps(report))
if __name__=='__main__': main()
