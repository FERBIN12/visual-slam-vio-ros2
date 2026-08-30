#!/usr/bin/env python3
"""Deterministic calibration/timing sensitivity sweep contract."""
import json
from pathlib import Path
def main():
    rows=[]
    for scale in (0.98,1.0,1.02):
        for offset_ms in (-5,0,5):
            err=abs(scale-1)*2+abs(offset_ms)*.001
            rows.append({'scale':scale,'offset_ms':offset_ms,'error':err,'verdict':'ACCEPT' if err<.01 else 'DEGRADE'})
    Path('out').mkdir(exist_ok=True); Path('out/calibration_timing_sweep.json').write_text(json.dumps({'rows':rows},indent=2)); print('calibration timing sweep lab: PASS' if any(r['verdict']=='DEGRADE' for r in rows) and any(r['verdict']=='ACCEPT' for r in rows) else 'calibration timing sweep lab: FAIL')
if __name__=='__main__': main()
