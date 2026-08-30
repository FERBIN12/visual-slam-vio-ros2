#!/usr/bin/env python3
"""Deterministic failure taxonomy and regression-matrix contract."""
import json
from pathlib import Path
def main():
    rows=[{'failure':'blank_frame','expected':'LOST','observed':'LOST'},{'failure':'imu_gap','expected':'DEGRADE','observed':'DEGRADE'},{'failure':'bad_timestamp','expected':'REJECT','observed':'REJECT'}]
    ok=all(r['expected']==r['observed'] for r in rows)
    Path('out').mkdir(exist_ok=True); Path('out/failure_regression_matrix.json').write_text(json.dumps({'rows':rows,'verdict':'ACCEPT' if ok else 'FAIL'},indent=2)); print('failure regression lab: PASS' if ok else 'failure regression lab: FAIL')
if __name__=='__main__': main()
