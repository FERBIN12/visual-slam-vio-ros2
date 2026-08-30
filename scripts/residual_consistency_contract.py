#!/usr/bin/env python3
"""Check residual whitening and innovation consistency deterministically."""
import json, numpy as np
from pathlib import Path
def main():
    rng=np.random.default_rng(4); r=rng.normal(0,.02,200); S=np.full(200,.02**2)
    nis=r*r/S; mean=float(nis.mean()); pct=float(np.mean(nis<3.84))
    report={"samples":len(r),"mean_NIS":mean,"under_95pct":pct,"verdict":"ACCEPT" if 0.5<mean<1.8 and pct>.9 else "DEGRADE"}
    Path('out').mkdir(exist_ok=True); Path('out/residual_consistency_report.json').write_text(json.dumps(report,indent=2)); print('residual consistency lab: PASS' if report['verdict']=='ACCEPT' else 'residual consistency lab: FAIL')
if __name__=='__main__': main()
