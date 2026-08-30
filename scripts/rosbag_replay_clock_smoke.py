#!/usr/bin/env python3
"""Deterministic rosbag2-style manifest, integrity, and replay-parity audit."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'out'; OUT.mkdir(exist_ok=True)
TOPICS=['/camera/image','/camera/camera_info','/imu/data','/tf','/ground_truth/pose']
def digest(stamps,counts,clock):
    return hashlib.sha256(json.dumps({'stamps':stamps,'counts':counts,'clock':clock},sort_keys=True).encode()).hexdigest()[:12]
def audit(c):
    checks={'topics_complete':set(c['topics'])==set(TOPICS),'counts_positive':all(v>0 for v in c['counts'].values()),'span_s':c['span_s']>=9.9,'monotonic':c['monotonic'],'hash_present':len(c['hash'])==12,'sim_clock':c['clock']=='sim_time','replay_parity':c['replay_hash']==c['hash']}
    return checks,'ACCEPT' if all(checks.values()) else 'REJECT'
def main():
    base={'topics':TOPICS,'counts':{'/camera/image':300,'/camera/camera_info':1,'/imu/data':2000,'/tf':300,'/ground_truth/pose':300},'span_s':10.0,'monotonic':True,'clock':'sim_time'}; good={**base}; good['hash']=digest([0.,10.],good['counts'],good['clock']); good['replay_hash']=good['hash']
    cases=[('trusted_bag',good),('missing_camera_info',{**good,'topics':['/camera/image','/imu/data','/tf','/ground_truth/pose']}),('wall_time_playback',{**good,'clock':'wall_time'}),('reordered_stamps',{**good,'monotonic':False}),('replay_mismatch',{**good,'replay_hash':'deadbeefcafe'})]
    rows=[]
    for name,c in cases:
        ch,s=audit(c); rows.append({'case':name,'status':s,'failed_checks':';'.join(k for k,v in ch.items() if not v),'message_total':sum(c['counts'].values())})
    checks={'trusted_accept':rows[0]['status']=='ACCEPT','missing_topic_reject':rows[1]['status']=='REJECT','wall_clock_reject':rows[2]['status']=='REJECT','nonmonotonic_reject':rows[3]['status']=='REJECT','parity_reject':rows[4]['status']=='REJECT','distinct_faults':len({r['failed_checks'] for r in rows[1:]})==4}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','cases':rows,'checks':checks,'command':'python3 scripts/rosbag_replay_clock_smoke.py','artifacts':{'report':'out/rosbag_replay_report.json','scorecard':'out/rosbag_replay_scorecard.csv','model':'out/rosbag_replay_model.npz','plot':'out/rosbag_replay_plot.png'}}
    with (OUT/'rosbag_replay_scorecard.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(OUT/'rosbag_replay_model.npz',topic_counts=np.array(list(good['counts'].values())),span=np.array([good['span_s']]),replay_hash=np.array([good['replay_hash']]))
    canvas=np.full((720,1280,3),246,np.uint8); ink=(45,43,37); teal=(115,140,25); red=(40,45,190)
    cv2.putText(canvas,'ROSBAG2 · REPLAY / CLOCK CONTROL',(42,52),cv2.FONT_HERSHEY_SIMPLEX,.78,ink,2,cv2.LINE_AA); cv2.rectangle(canvas,(55,105),(760,635),(215,208,198),1); cv2.putText(canvas,'5 TOPICS · 2,901 MESSAGES',(90,170),cv2.FONT_HERSHEY_SIMPLEX,.6,ink,1,cv2.LINE_AA); cv2.putText(canvas,'10.0 s · monotonic · sim_time',(90,260),cv2.FONT_HERSHEY_SIMPLEX,.58,ink,1,cv2.LINE_AA); cv2.putText(canvas,'REPLAY HASH MATCH', (90,365),cv2.FONT_HERSHEY_SIMPLEX,.72,teal,2,cv2.LINE_AA); cv2.putText(canvas,'ACCEPT', (90,465),cv2.FONT_HERSHEY_SIMPLEX,.8,teal,2,cv2.LINE_AA); cv2.rectangle(canvas,(820,105),(1225,635),(215,208,198),1); cv2.putText(canvas,'INTEGRITY FAILURES',(855,155),cv2.FONT_HERSHEY_SIMPLEX,.56,ink,1,cv2.LINE_AA); cv2.putText(canvas,'missing topic', (855,245),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'wall clock', (855,300),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'reordered stamps', (855,355),cv2.FONT_HERSHEY_SIMPLEX,.52,ink,1,cv2.LINE_AA); cv2.putText(canvas,'REJECT · no parity', (855,475),cv2.FONT_HERSHEY_SIMPLEX,.68,red,2,cv2.LINE_AA); cv2.imwrite(str(OUT/'rosbag_replay_plot.png'),canvas); (OUT/'rosbag_replay_report.json').write_text(json.dumps(report,indent=2)+'\n'); print('rosbag replay lab:',report['status']); return 0 if report['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
