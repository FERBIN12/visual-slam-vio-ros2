#!/usr/bin/env python3
"""Validate deterministic experiment manifests before execution."""
import hashlib, json
from pathlib import Path

REQUIRED=("dataset","sequence","seed","estimator","config_hash","command")
def check(m):
    keys=all(k in m and m[k] not in (None,"") for k in REQUIRED)
    seed=isinstance(m.get("seed"),int) and m["seed"]>=0
    h=isinstance(m.get("config_hash"),str) and len(m["config_hash"])>=8
    return keys and seed and h
def main():
    out=Path("out"); out.mkdir(exist_ok=True)
    base={"dataset":"EuRoC","sequence":"MH_01","seed":7,"estimator":"orbslam3","config_hash":"a1b2c3d4","command":"run_vio --manifest manifest.json"}
    cases=[("ready",base),("missing_seed",{k:v for k,v in base.items() if k!="seed"}),("negative_seed",{**base,"seed":-1}),("short_hash",{**base,"config_hash":"x"})]
    rows=[]
    for name,m in cases:
        rows.append({"case":name,"manifest_sha256":hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()[:16],"verdict":"ACCEPT" if check(m) else "REJECT"})
    (out/"experiment_manifest_scorecard.json").write_text(json.dumps({"cases":rows,"required":REQUIRED},indent=2))
    print("experiment manifest lab: PASS" if [r["verdict"] for r in rows]==["ACCEPT","REJECT","REJECT","REJECT"] else "experiment manifest lab: FAIL")
if __name__=="__main__": main()
