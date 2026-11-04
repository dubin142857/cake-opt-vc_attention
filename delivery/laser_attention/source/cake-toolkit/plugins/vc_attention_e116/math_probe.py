"""Finite FP16 mechanism audit; scalar values do not certify attention payloads."""
from pathlib import Path
import argparse,json,os,sys
from core.opgen.architecture import _dump
from core.opgen.runner import StageReport,Metric
from plugins.vc_attention.execution import digest
from core.opgen.architecture import Architecture

def cpu_cases():
    import numpy as np
    bits=np.arange(0x7c00,dtype=np.uint16);x=bits.view(np.float16)
    examples=[]
    for dim in (64,128):
        s=np.float16(dim**-.5);y=(x.astype(np.float32)*np.float32(s)).astype(np.float16)
        positions=np.flatnonzero((y[:-1]==y[1:]) & (x[:-1]<x[1:]))
        # Prefer ordinary finite normal values; underflow ties remain explicit.
        normal=positions[(x[positions]>=np.float16(.5)) & (x[positions]<np.float16(2))]
        i=int((normal if len(normal) else positions)[0]);a,b=x[i:i+2]
        assert a<b and y[i]==y[i+1]
        examples.append(dict(dim=dim,scale=float(s),raw=[float(a),float(b)],scaled=[float(y[i]),float(y[i+1])],
            scaled_first_argmax=0,raw_argmax=1,V=[-1,1],old_output=-1,new_output=1,absolute_output_difference=2,
            scope='Constructive first-tie argmax counterexample; NPU tie behavior independently measured'))
    return dict(status='counterexample',claim='Complete attention equivalence is false without tie/payload conditions',examples=examples)

def audit_map(x,mapped,old,raw,maximum,torch):
    """Check all exported input values; retain both value and index obligations."""
    if mapped.shape!=x.shape or old.shape!=(len(x),2) or raw.shape!=old.shape or maximum.shape!=(len(x),):raise ValueError('Probe extent')
    if any(t.dtype!=torch.float16 for t in (x,mapped,old,raw,maximum)):raise ValueError('Probe dtype')
    order=x.flatten().float().argsort(stable=True);ys=mapped.flatten()[order].float()
    violations=int((ys[1:]<ys[:-1]).sum())
    value_bits=int((old[:,0].contiguous().view(torch.int16)!=maximum.view(torch.int16)).sum())
    oldids=(old.contiguous().view(torch.int32).flatten()>>16).long();rawids=(raw.contiguous().view(torch.int32).flatten()>>16).long()
    if ((oldids<0)|(oldids>=128)|(rawids<0)|(rawids>=128)).any():raise ValueError('Packed reducer index outside row')
    source_old=mapped.gather(1,oldids[:,None]).flatten();source_raw=x.gather(1,rawids[:,None]).flatten()
    if not torch.equal(source_old.contiguous().view(torch.int16),old[:,0].contiguous().view(torch.int16)):raise ValueError('Old argmax score/source pair')
    if not torch.equal(source_raw.contiguous().view(torch.int16),raw[:,0].contiguous().view(torch.int16)):raise ValueError('Raw argmax score/source pair')
    if (old[:,0].float()<mapped.float().max(-1).values).any() or (raw[:,0].float()<x.float().max(-1).values).any():raise ValueError('Reducer did not select mathematical maximum')
    return dict(monotonicity_violations=violations,composition_value_bit_mismatches=value_bits,changed_argmax_rows=int((oldids!=rawids).sum()),
        source_pair_errors=0,rows=len(x),elements=x.numel())

def native_check(c,work):
    sys.path.insert(0,str(Path(c['project_root'])/'scripts'))
    import torch,torch_npu
    from mindie_laser_quant_fused import validate_preprocess_manifest
    torch.set_num_threads(2);torch.npu.set_device(c['device'])
    evidence=validate_preprocess_manifest(c['candidate_manifest'],operation='max_scale_probe');torch.ops.load_library(evidence['library_path'])
    if digest(evidence['library_path'])!=evidence['library_sha256']:raise ValueError('Probe provider differs')
    mapped_paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if len(line.split())>=6}
    if evidence['library_path'] not in mapped_paths:raise ValueError('Probe provider not mapped')
    op=torch.ops.vc_laser_max_scale_probe.run.default
    if str(op._schema)!='vc_laser_max_scale_probe::run(Tensor input, float scale) -> (Tensor, Tensor, Tensor, Tensor)':raise ValueError('Probe ABI differs')
    bits=torch.arange(65536,dtype=torch.int32).to(torch.int16);finite=((bits.to(torch.int32)&0x7c00)!=0x7c00);finite_bits=bits[finite]
    assert len(finite_bits)==63488
    source=torch.cat((finite_bits,torch.tensor([0x7c00,-1024],dtype=torch.int16),torch.zeros(2046,dtype=torch.int16))).view(torch.float16).reshape(512,128)
    reports=[];table={}
    for dim in (64,128):
        scale=float(torch.tensor(dim**-.5,dtype=torch.float16));s=source.to(f'npu:{c["device"]}')
        runs=[tuple(t.cpu() for t in op(s,scale)) for _ in range(3)]
        if any(not torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16)) for run in runs[1:] for a,b in zip(runs[0],run)):raise ValueError('Mechanism nondeterministic')
        mapped,old,raw,maximum=runs[0];record=audit_map(source,mapped,old,raw,maximum,torch)
        cpu=(source.float()*scale).half();record.update(dim=dim,scale=scale,finite_patterns=len(finite_bits),repeats=3,cpu_scale_bit_differences=int((cpu.view(torch.int16)!=mapped.view(torch.int16)).sum()))
        table[dim]=dict(input_bits=bits,valid_finite=finite,output_bits=mapped.flatten()[:63488].view(torch.int16).clone(),source=source,outputs=runs[0])
        reports.append(record)
    artifact=work/'scale-table.pt';torch.save(table,artifact)
    result=dict(status='observed',scope='All finite FP16 scalar maps for fixed half D64/128 scales plus infinities; finite row arrangements only, no universal reducer composition proof or attention equivalence',records=reports,provider=evidence,
        artifact=dict(path=str(artifact),sha256=digest(artifact)),cpu_counterexamples=cpu_cases())
    (work/'check.detail.json').write_text(_dump(result,pretty=True)+'\n')
    if any(r['monotonicity_violations'] or r['composition_value_bit_mismatches'] for r in reports):raise ValueError('Frozen scalar/composition prediction falsified; detail and table retained')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu',action='store_true');ap.add_argument('--output');ap.add_argument('--config');ap.add_argument('--stage');a=ap.parse_args()
    if a.cpu:
        result=cpu_cases();Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0
    c=json.loads(Path(a.config).read_text());work=Path(os.environ['CAKE_WORK']);kind=a.stage;metrics=()
    model=Architecture.from_json(Path(c['model']).read_text())
    if model.fingerprint()!=os.environ['CAKE_MODEL_FP'] or model.semantic.contract.id!=os.environ['CAKE_CONTRACT']:raise ValueError('Probe model/contract identity')
    for item in c['identities']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Probe dependency changed:'+item['path'])
    if Path(os.environ.get('ASCEND_HOME_PATH','')).resolve()!=Path(c['cann_root']).resolve():raise ValueError('Probe CANN runtime differs')
    for key in ('ASCEND_CACHE_PATH','ACL_OP_COMPILER_CACHE_DIR','ASCEND_PROCESS_LOG_PATH'):
        cache=work/key.lower();cache.mkdir(exist_ok=True);os.environ[key]=str(cache)
    if kind=='build':
        (work/'native.manifest.json').write_text(_dump(dict(libraries=c['native_libraries']),pretty=True)+'\n');outputs=('native.manifest.json',)
    elif kind=='check':native_check(c,work);outputs=('check.detail.json','scale-table.pt')
    elif kind=='measure':
        result=json.loads((work/'check.detail.json').read_text());count=sum(r['finite_patterns'] for r in result['records'])
        metrics=(Metric('finite_patterns_observed','patterns',count),)
        (work/'measure.detail.json').write_text(_dump(dict(status='observed',finite_patterns_observed=count,timing='unavailable; numerical mechanism scope only'),pretty=True)+'\n');outputs=('measure.detail.json',)
    else:outputs=()
    r=StageReport(os.environ['CAKE_PLAN_FP'],os.environ['CAKE_BUILD_FP'],kind,True,os.environ['CAKE_NONCE'],metrics)
    (work/(kind+'.json')).write_text(r.to_json(pretty=True));return 0

if __name__=='__main__':raise SystemExit(main())
