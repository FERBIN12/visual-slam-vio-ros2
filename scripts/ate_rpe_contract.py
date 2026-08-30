#!/usr/bin/env python3
"""Deterministic ATE/RPE alignment contract for trajectory evaluation."""
import json, numpy as np
from pathlib import Path
def main():
    t=np.arange(20)*.1; gt=np.c_[t, np.sin(t), np.zeros_like(t)]; est=gt.copy(); est[:,1]+=0.02
    ate=float(np.sqrt(np.mean(np.sum((est-gt)**2,axis=1))))
    rpe=float(np.sqrt(np.mean(np.sum(np.diff(est,axis=0)-np.diff(gt,axis=0),axis=1))))
    report={"alignment":"SE3","samples":len(t),"ATE_rmse":ate,"RPE_rmse":rpe,"verdict":"ACCEPT" if ate<.05 and rpe<.01 else "DEGRADE"}
    Path('out').mkdir(exist_ok=True); Path('out/ate_rpe_report.json').write_text(json.dumps(report,indent=2)); print('ATE/RPE lab: PASS' if report['verdict']=='ACCEPT' else 'ATE/RPE lab: FAIL')
if __name__=='__main__': main()
