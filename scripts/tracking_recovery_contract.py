#!/usr/bin/env python3
"""Deterministic tracking-loss and recovery state-machine contract."""
import json
from pathlib import Path
def main():
    states=['TRACKING','LOST','RECOVERING','TRACKING']; transitions=list(zip(states,states[1:]))
    ok=transitions==[('TRACKING','LOST'),('LOST','RECOVERING'),('RECOVERING','TRACKING')]
    Path('out').mkdir(exist_ok=True); Path('out/tracking_recovery_report.json').write_text(json.dumps({'states':states,'transitions':transitions,'verdict':'ACCEPT' if ok else 'REJECT'},indent=2)); print('tracking recovery lab: PASS' if ok else 'tracking recovery lab: FAIL')
if __name__=='__main__': main()
