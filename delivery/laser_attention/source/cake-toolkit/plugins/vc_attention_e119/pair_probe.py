"""Pair128 repeats versus independent CPU max and actual legacy64-row selection."""
from pathlib import Path
import json
MASKS=((128,128),(128,123),(1,128),(128,0),(1,0),(0,0))
BASES=(0,16777344,2147483391)
SCALES=(1.,128.**-.5)
FAMILIES=('varied','flat','edge_half')

def source(torch,family):
    i=torch.arange(128*128).reshape(128,128)
    if family=='varied':return (((i*37+i//128*17)%2047)-1023).to(torch.float16)/8
    if family=='flat':return torch.zeros((128,128),dtype=torch.float16)
    half=torch.tensor([0.,-0.,2.**-24,-2.**-24,65504.,-65504.,2.**-14,-2.**-14],dtype=torch.float16)
    return half[(i+i//128)%8]

def compare(scores,pair,legacy,valid0,valid1,base,scale,torch):
    a,b=(x.cpu().contiguous() for x in pair)
    if a.shape!=(128,) or b.shape!=(128,) or a.dtype!=torch.float16 or b.dtype!=torch.int32:raise AssertionError('Pair output dtype/shape')
    masked=scores.clone();masked[:64,valid0:]=-float('inf');masked[64:,valid1:]=-float('inf')
    maxima,offset=masked.max(dim=1);expected=(maxima.float()*float(torch.tensor(scale,dtype=torch.float16))).half()
    legal=((b.to(torch.int64)-base-(torch.arange(128)//64)*128)==offset)
    old_values=torch.cat([x[0][0] for x in legacy]);old_ids=torch.cat([x[1][0] for x in legacy]).long()+base+(torch.arange(128)//64)*128
    # Legacy scale1 max output is rounded half before the same half-factor scale.
    old_scaled=(old_values.float()*float(torch.tensor(scale,dtype=torch.float16))).half()
    return dict(cpu_value_bytes=int((a.view(torch.uint8)!=expected.view(torch.uint8)).sum()),cpu_index_errors=int((~legal).sum()),legacy_value_bytes=int((a.view(torch.uint8)!=old_scaled.view(torch.uint8)).sum()),legacy_index_errors=int((b.long()!=old_ids).sum()))

def primitive_checks(adapter,torch,device,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);records=[];nondeterminism=0
    # Existing geometry constructor is the same as the production candidate prepare.
    geometry=adapter.make_positions(device)
    for family in FAMILIES:
        cpu=source(torch,family);x=cpu.to(device)
        for v0,v1 in MASKS:
            legacy=[tuple(t.cpu() for t in adapter.probe(x[:64].contiguous(),geometry,64,v0)),tuple(t.cpu() for t in adapter.probe(x[64:].contiguous(),geometry,64,v1))]
            for base in BASES:
                for scale in SCALES:
                    runs=[tuple(t.cpu() for t in adapter.pair_probe(x,geometry,v0,v1,base,scale)) for _ in range(2)]
                    for a,b in zip(*runs):nondeterminism+=int((a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8)).sum())
                    row=dict(family=family,valid0=v0,valid1=v1,base=base,scale=scale,**compare(cpu,runs[0],legacy,v0,v1,base,scale,torch));records.append(row)
                    if any(row[k] for k in ('cpu_value_bytes','cpu_index_errors','legacy_value_bytes','legacy_index_errors')) and not (directory/'first-discrepancy.pt').exists():torch.save(dict(input=cpu,pair=runs[0],legacy=legacy,row=row),directory/'first-discrepancy.pt')
    totals={k:sum(x[k] for x in records) for k in ('cpu_value_bytes','cpu_index_errors','legacy_value_bytes','legacy_index_errors')}
    result=dict(cases=len(records),repeats=2,nondeterministic_bytes=nondeterminism,records=records,totals=totals,passed=not any(totals.values()) and not nondeterminism,scope='Finite mask/sign/tie/base families; exact independent rows, no full attention quality claim')
    (directory/'detail.json').write_text(json.dumps(result,indent=2)+'\n');return result

def validate_evidence(r):
    if not isinstance(r,dict) or r.get('cases')!=108 or r.get('repeats')!=2 or type(r.get('nondeterministic_bytes')) is not int or r['nondeterministic_bytes']!=0 or r.get('passed') is not True:raise ValueError('Missing/failed pair evidence')
    rows=r.get('records',[]);expected={(f,v0,v1,b,s) for f in FAMILIES for v0,v1 in MASKS for b in BASES for s in SCALES}
    if len(rows)!=108 or {(x.get('family'),x.get('valid0'),x.get('valid1'),x.get('base'),x.get('scale')) for x in rows}!=expected:raise ValueError('Pair coverage incomplete/duplicate')
    totals={k:0 for k in ('cpu_value_bytes','cpu_index_errors','legacy_value_bytes','legacy_index_errors')}
    for x in rows:
        for k in totals:
            if type(x.get(k)) is not int or x[k]!=0:raise ValueError('Pair per-case failure/malformed')
    if r.get('totals')!=totals or any(type(v) is not int for v in r.get('totals',{}).values()):raise ValueError('Pair totals inconsistent')
