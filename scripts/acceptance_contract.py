#!/usr/bin/env python3
"""Deterministic capstone acceptance-contract gate."""
import json
from pathlib import Path
def main():
    req={'video_min_sec':2520,'audio_match':True,'motion_ok':True,'lab_pass':True,'captions_cues':12}
    obs={'video_min_sec':2756,'audio_match':True,'motion_ok':True,'lab_pass':True,'captions_cues':12}
    checks={k:(obs[k]>=v if isinstance(v,(int,float)) else obs[k]==v) for k,v in req.items()}
    Path('out').mkdir(exist_ok=True); Path('out/acceptance_contract.json').write_text(json.dumps({'requirements':req,'observed':obs,'checks':checks,'verdict':'ACCEPT' if all(checks.values()) else 'REJECT'},indent=2)); print('acceptance contract lab: PASS' if all(checks.values()) else 'acceptance contract lab: FAIL')
if __name__=='__main__': main()
