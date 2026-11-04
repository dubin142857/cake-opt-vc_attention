"""Retain protected runner; validate the explicitly new approximation obligations."""
import math
from plugins.vc_attention import execution as base
from plugins.vc_attention_e119.pair_probe import validate_evidence
_validate=base.validate_report

def integer(x):return type(x) is int

def validate_primitives(r,mode):
    p=r.get('primitive_checks',{})
    if (p.get('mode')!=mode or p.get('cases')!=90 or p.get('finite_half_patterns')!=63488 or p.get('repeats')!=2 or p.get('selected_mode_passed') is not True or p.get('nondeterministic_bits')!=0):raise ValueError('FMA primitive missing/failed')
    records=p.get('records',[])
    if len(records)!=90 or {(x.get('weight_phase'),x.get('accumulator')) for x in records}!={(w,a) for w in range(10) for a in range(9)}:raise ValueError('FMA coverage')
    totals={k:0 for k in ('old_vs_cpu','float_vs_old','half_vs_old')}
    for row in records:
        if row.get('finite_half_patterns')!=63488:raise ValueError('FMA finite domain')
        for key in totals:
            count=row.get(key,{}).get('bit_differences')
            if not integer(count) or count<0:raise ValueError('FMA malformed count')
            totals[key]+=count
    if p.get('totals')!=totals or totals['old_vs_cpu'] or totals[mode+'_vs_old']:raise ValueError('FMA totals/failure')
    validate_evidence(r.get('pair_checks'))

def validate_pipeline(row,shape):
    b,h,q,k,d=shape;rows=b*h*((q+127)//128)*128;raw=row.get('raw_source',{})
    if (raw.get('rows_checked'),raw.get('score_columns'),raw.get('padding_columns'))!=(rows,8,32):raise ValueError('Raw source coverage')
    for key in ('score_bit_errors','ID_errors'):
        if not integer(raw.get(key)) or raw[key]!=0:raise ValueError('Raw source missing/errors')
    for key in ('packet_bit_differences','public_cast_byte_differences'):
        if not integer(row.get(key)) or row[key]!=0:raise ValueError('Packet/public missing/errors')
    for key,limit in (('relative_rmse',1e-5),('max_abs',1e-4)):
        x=row.get('fp32_pv_error',{}).get(key)
        if type(x) not in (int,float) or not math.isfinite(x) or x<0 or x>limit:raise ValueError('Original ownPV gate')
    pc=row.get('producer_counters',{});rc=row.get('raw_reference_counters',{});w=row.get('centered_halfweights',{})
    if pc.get('query_tiles')!=rows//128 or rc.get('query_tiles')!=rows//128 or w.get('rows_checked')!=rows:raise ValueError('Pipeline counter/reference/weight coverage')
    if pc.get('scale_elements')!=rows//64*1536:raise ValueError('Scale element coverage')
    expected_raw=rows*((k+255)//256*2)
    if pc.get('raw_reduction_elements')!=expected_raw:raise ValueError('Raw reduction coverage')
    n=w.get('boundary_differences')
    if not integer(n) or n<0 or w.get('max_half_neighbor_steps')!=1:raise ValueError('Halfweight evidence')

def validate_report(c,r,*,timed):
    result=_validate(c,r,timed=timed);validate_primitives(r,c['fma_mode']);rows=r.get('finalize_checks',[])
    expected=((1,1,1,1,64),(1,1,17,257,64),(1,1,128,385,128),(2,3,17,769,128),(1,3,129,1153,64),(1,1,129,65537,64),(1,1,17,1025,128),(1,1,257,129,64),(1,1,17,385,128),(1,1,1,257,128),(1,1,64,641,128),(1,1,65,897,64),(1,2,128,1153,128))
    if len(rows)!=13 or [tuple(x.get('shape',[])) for x in rows]!=list(expected):raise ValueError('Component coverage')
    for x,shape in zip(rows,expected):
        validate_pipeline(x,shape)
        if not integer(x.get('initialized_state_repeat_byte_errors')) or x['initialized_state_repeat_byte_errors']!=0:raise ValueError('Initialized repeat state')
    if timed:
        validate_pipeline(r.get('target_check',{}),(1,1,c['shape'][2],c['shape'][2],128))
        result['delay']=dict(components=len(rows),contract='RawTop8 from actual E119binarytemperature1; centeredFP32scale/weights/source/PV checked; distinct roundingtie behavior, video unverified')
    return result

def main():
    base.validate_report=validate_report;return base.main()
if __name__=='__main__':raise SystemExit(main())
