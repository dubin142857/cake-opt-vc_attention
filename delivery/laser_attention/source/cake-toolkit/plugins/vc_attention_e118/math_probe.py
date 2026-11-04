"""Conditional half-factor product lemma and actual row-broadcast vmla audit."""
from pathlib import Path
import argparse,json

def weight_family(torch):
    return torch.tensor([0.,-0.,2.**-24,-2.**-24,2.**-14,.125,.5,1.,-1.,65504.],dtype=torch.float16)

def cpu_controls():
    import numpy as np
    bits=np.arange(65536,dtype=np.uint16);bits=bits[(bits&0x7c00)!=0x7c00];x=bits.view(np.float16).astype(np.float64)
    weights=np.array([0.,-0.,2.**-24,-2.**-24,2.**-14,.125,.5,1.,-1.,65504.],dtype=np.float16)
    differences=0
    for w in weights:
        product=x*np.float64(w);rounded=product.astype(np.float32).astype(np.float64)
        differences+=int(np.count_nonzero(product!=rounded))+int(np.count_nonzero(np.signbit(product)!=np.signbit(rounded)))
    assert len(bits)==63488 and differences==0
    a=np.float32(1+2.**-23);b=np.float32(1-2.**-23);c=np.float32(-1)
    separate=np.float32(np.float32(a*b)+c);fused=np.float32(np.float64(a)*np.float64(b)+np.float64(c))
    assert separate==0 and fused==np.float32(-2.**-46) and separate!=fused
    return dict(status='passed',finite_half_patterns=63488,weight_families=len(weights),product_float32_representation_errors=differences,
        non_half_counterexample=dict(a=float(a),b=float(b),c=float(c),separate=float(separate),exact_fma=float(fused)),
        scope='Finite CPU product family and constructive out-of-premise IEEE counterexample; actual vmla precision remains a hardware obligation')

def compare_outputs(values,weights,initial,outputs,torch):
    if len(outputs)!=3 or any(t.shape!=values.shape or t.dtype!=torch.float32 for t in outputs):raise AssertionError('Primitive output shape/dtype')
    reference=initial+(values.float()*weights.float()[:,None])
    result={}
    for name,a,b in [('old_vs_cpu',outputs[0],reference),('float_vs_old',outputs[1],outputs[0]),('half_vs_old',outputs[2],outputs[0])]:
        mismatch=(a.contiguous().view(torch.int32)!=b.contiguous().view(torch.int32));where=mismatch.nonzero()
        result[name]=dict(bit_differences=int(mismatch.sum()),first_positions=where[:8].tolist())
    return result

def primitive_checks(adapter,torch,device,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    bits=torch.arange(65536,dtype=torch.int32).to(torch.int16);finite=(bits.int()&0x7c00)!=0x7c00
    values=torch.cat((bits[finite],torch.zeros(2048,dtype=torch.int16))).view(torch.float16).reshape(512,128)
    family=weight_family(torch);source=values.to(device);records=[];totals={k:0 for k in ('old_vs_cpu','float_vs_old','half_vs_old')};nondeterminism=0
    for phase in range(len(family)):
        weights=family[(torch.arange(512)+phase)%len(family)];w=weights.to(device)
        product=values.float()*weights.float()[:,None]
        for accumulator in range(9):
            if accumulator<7:
                initial=torch.full(values.shape,[0.,-0.,1.,-1.,2.**-48,2.**24,2.**-100][accumulator],dtype=torch.float32)
            elif accumulator==7:initial=-product
            else:initial=torch.nextafter(-product,torch.full_like(product,float('inf')))
            a=initial.to(device);runs=[tuple(t.cpu() for t in adapter.fma_probe(source,w,a)) for _ in range(2)]
            for x,y in zip(*runs):nondeterminism+=int((x.contiguous().view(torch.int32)!=y.contiguous().view(torch.int32)).sum())
            check=compare_outputs(values,weights,initial,runs[0],torch)
            for key in totals:totals[key]+=check[key]['bit_differences']
            record=dict(weight_phase=phase,accumulator=accumulator,finite_half_patterns=63488,**check);records.append(record)
            if any(check[k]['bit_differences'] for k in check) and not (directory/'first-discrepancy.pt').exists():
                torch.save(dict(values=values,weights=weights,initial=initial,outputs=runs[0],record=record),directory/'first-discrepancy.pt')
    mode=adapter.metadata['pv_arithmetic'].removeprefix('fp32_vmla_')
    result=dict(status='observed',mode=mode,cases=len(records),finite_half_patterns=63488,weight_phases=len(family),accumulator_families=9,repeats=2,
        totals=totals,nondeterministic_bits=nondeterminism,records=records,scope='All finite half values across ten row-varying weight families and nine accumulator arrangements; not every IEEE input tuple')
    result['selected_mode_passed']=not totals['old_vs_cpu'] and not totals[mode+'_vs_old'] and not nondeterminism
    (directory/'detail.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu',action='store_true',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    result=cpu_controls()
    with Path(a.output).open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
