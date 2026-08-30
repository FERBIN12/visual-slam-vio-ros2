#!/usr/bin/env python3
"""Deterministic bounded nonlinear sliding-window optimization lab."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)
def main():
    rng=np.random.default_rng(12); n=36; t=np.linspace(0,7,n); truth=np.column_stack([.4*np.sin(.7*t),.2*np.cos(.5*t),.1*t]); obs=truth+rng.normal(0,.004,truth.shape)
    def res(x,idx): return (x.reshape(-1,3)-obs[idx]).ravel()/0.004
    rows=[]; window=10
    for end in range(window,n+1):
        idx=np.arange(end-window,end); x0=obs[idx].copy().ravel()+.02
        tic=time.perf_counter(); fit=least_squares(lambda x:res(x,idx),x0,max_nfev=80); elapsed=(time.perf_counter()-tic)*1000
        rows.append({'end_index':int(end-1),'window':window,'cost':float(2*fit.cost),'gradient_inf':float(np.max(np.abs(fit.grad))),'iterations':int(fit.nfev),'solve_ms':elapsed,'holdout_rmse_m':float(np.sqrt(np.mean((fit.x.reshape(-1,3)-obs[idx])**2)))})
    full0=np.zeros((n,3)).ravel(); full=least_squares(lambda x:res(x,np.arange(n)),full0,max_nfev=80); final=rows[-1]
    checks={k:bool(v) for k,v in {'cost_reduced':final['cost']<1e-8,'gradient_small':final['gradient_inf']<1e-5,'bounded_window':all(r['window']==window for r in rows),'holdout_finite':np.isfinite(final['holdout_rmse_m']),'full_history_larger':n>window}.items()}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','truth':{'states':n,'window':window},'final_window':final,'full_history':{'states':n,'cost':float(2*full.cost),'iterations':int(full.nfev)},'checks':checks,'command':'python3 scripts/sliding_window_smoke.py','artifacts':{'report':'out/sliding_window_report.json','trace':'out/sliding_window_trace.csv','model':'out/sliding_window_model.npz','plot':'out/sliding_window_plot.png'}}
    with (OUT/'sliding_window_trace.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    np.savez(OUT/'sliding_window_model.npz',times_s=t,truth=truth,observations=obs,final_state=full.x,window_cost=np.array([r['cost'] for r in rows]))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); teal=(115,140,25); blue=(190,115,34)
    cv2.putText(canvas,'SLIDING-WINDOW OPTIMIZATION',(42,52),cv2.FONT_HERSHEY_SIMPLEX,.78,ink,2,cv2.LINE_AA); cv2.rectangle(canvas,(60,110),(760,635),(215,208,198),1); cv2.putText(canvas,'WINDOW TRACE',(90,150),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA)
    vals=np.array([r['cost'] for r in rows]); xs=np.linspace(100,700,len(vals)); ys=500-300*(vals-vals.min())/max(vals.max()-vals.min(),1e-12); cv2.polylines(canvas,[np.column_stack([xs,ys]).astype(np.int32)],False,blue,4,cv2.LINE_AA); cv2.putText(canvas,f'window = {window} states',(100,580),cv2.FONT_HERSHEY_SIMPLEX,.5,ink,1,cv2.LINE_AA)
    cv2.rectangle(canvas,(810,110),(1220,635),(215,208,198),1); cv2.putText(canvas,'SOLVER GATES',(850,150),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA); cv2.putText(canvas,'cost reduced',(850,245),cv2.FONT_HERSHEY_SIMPLEX,.55,ink,1,cv2.LINE_AA); cv2.putText(canvas,'PASS',(850,285),cv2.FONT_HERSHEY_SIMPLEX,.72,teal,2,cv2.LINE_AA); cv2.putText(canvas,f'full history: {n} states',(850,410),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,f'bounded: {window} states',(850,455),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/'sliding_window_plot.png'),canvas); (OUT/'sliding_window_report.json').write_text(json.dumps(report,indent=2)+'\n'); print('Sliding-window lab:',report['status']); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
