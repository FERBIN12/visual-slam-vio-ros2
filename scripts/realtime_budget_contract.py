#!/usr/bin/env python3
"""Deterministic real-time budget and latency contract."""
import json
from pathlib import Path
def main():
    stages={'capture_ms':3.2,'frontend_ms':8.5,'backend_ms':12.0,'publish_ms':1.1}; total=sum(stages.values()); budget=33.3
    report={'stages_ms':stages,'total_ms':total,'budget_ms':budget,'headroom_ms':budget-total,'verdict':'ACCEPT' if total<budget else 'DEGRADE'}
    Path('out').mkdir(exist_ok=True); Path('out/realtime_budget_report.json').write_text(json.dumps(report,indent=2)); print('real-time budget lab: PASS' if report['verdict']=='ACCEPT' else 'real-time budget lab: FAIL')
if __name__=='__main__': main()
