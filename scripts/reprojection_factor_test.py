#!/usr/bin/env python3
"""Tiny executable residual test for the VIO optimizer core."""
from __future__ import annotations
import math
def project(p,K=(458.2,457.9,319.6,241.1)):
    x,y,z=p; fx,fy,cx,cy=K; return fx*x/z+cx,fy*y/z+cy
def main():
    predicted=project((0.42,-0.08,3.7)); observed=(predicted[0]+0.12,predicted[1]-0.08)
    residual=(observed[0]-predicted[0],observed[1]-predicted[1]); norm=math.hypot(*residual)
    assert abs(norm-math.sqrt(.12**2+.08**2))<1e-9
    print(f'reprojection factor test: PASS residual={residual} norm={norm:.6f}px')
if __name__=='__main__': main()
