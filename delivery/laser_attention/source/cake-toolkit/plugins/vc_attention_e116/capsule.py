"""Repeat a verified frozen evaluator with new work/output and fresh stages.

Reusing a closure avoids accidental evaluator drift from a new directory glob.
It reuses no guard, correctness result or measurement. This is not rollback,
source equivalence certification, scheduling or an alternative executor.
"""
from dataclasses import replace,asdict
from pathlib import Path
import argparse,json
from core.opgen.runner import RunReceipt,identify,validate_plan
from .plan import snapshot

def repeat(receipt_path,output):
    source=Path(receipt_path).resolve();r=RunReceipt.from_json(source.read_text());r.verify()
    if r.state!='verified':raise ValueError('Only a verified capsule can seed an independent repeat')
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Repeat output exists: '+str(out))
    p=replace(r.plan,id=out.name,output_dir=str(out/'run'),cache_cpu=False)
    validate_plan(p)
    if p.fingerprint()==r.plan.fingerprint():raise ValueError('Repeat is not an independent plan')
    out.mkdir();(out/'plan.json').write_text(p.to_json(pretty=True));snapshot(p,out)
    (out/'capsule-origin.json').write_text(json.dumps(dict(parent=asdict(identify(source,'output')),
        parent_plan=r.plan.fingerprint(),new_plan=p.fingerprint(),reuse='Exact frozen inputs/stages/env/device/model; no result reuse',fresh=['build','guard','check','measure','verify']),indent=2)+'\n')
    return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    p=repeat(a.receipt,a.output);print(p.fingerprint());return 0

if __name__=='__main__':raise SystemExit(main())
