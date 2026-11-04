"""Freeze a new implementation against the actual protected postmax capsule."""
from dataclasses import replace
from pathlib import Path
import argparse,hashlib,json
from core.opgen.architecture import Architecture,Contract,Attribute
from core.opgen.runner import identify,validate_plan,MetricBinding
from core.opgen.delivery import DeliveryReceipt
from plugins.vc_attention_e116.capsule import repeat
from plugins.vc_attention_e116.plan import snapshot
from plugins.vc_attention_e116.preflight import check,load_validator

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--mode',choices=('half',),required=True);ap.add_argument('--incumbent',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    project=Path(a.project).resolve();operation='sparse_delay';manifest=project/'results'/(operation+'_cann920_build.json')
    preflight=check(project);evidence=load_validator(project)(manifest,operation=operation);m=json.loads(manifest.read_text())
    parent=DeliveryReceipt.from_json(Path(a.incumbent).read_text());parent.verify();oldfp=next(x.fingerprint for x in parent.after.models if x.id==parent.after.validated_best)
    out=Path(a.output).resolve()
    if out.exists():raise FileExistsError('Pair output exists')
    out.mkdir();p=repeat(parent.plan.runs[0].path,out/'base-capsule')
    oldcfg=Path(p.stages[0].argv[p.stages[0].argv.index('--config')+1]);c=json.loads(oldcfg.read_text());oldmodel=Architecture.from_json(Path(c['model']).read_text())
    if oldmodel.fingerprint()!=oldfp or p.model_fingerprint!=oldfp:raise ValueError('Actual capsule is not the protected incumbent model')
    incumbent_manifest=c['candidate_manifest'];env={v.name:v.value for v in p.env};old_measured_library=json.loads(Path(env['VC_ATTENTION_INCUMBENT_MANIFEST']).read_text())['library_path']
    libraries=[x for x in c['native_libraries'] if Path(x['path']).resolve()!=Path(old_measured_library).resolve()]
    if len(libraries)!=len(c['native_libraries'])-1:raise ValueError('Old unmeasured inline provider removal ambiguous')
    libraries.append(dict(path=evidence['library_path'],sha256=evidence['library_sha256']))
    reference=Path(__file__).with_name('checks.py').resolve();refsha=hashlib.sha256(reference.read_bytes()).hexdigest()
    contract=Contract('raw-top8-centered-temperature','approximate',str(reference)+'@'+refsha,
        'Original halfQK/blockmax/rawTop8 thenFP32centeredtemperature/softmax andoriginalhalf-factorFP32PV. No new point stage. New roundedtie/logit contract versus E119.',
        'Actual immutable E119binarytemperature1 rawTop8 IDs andFP32scaled score exact; original source coverage/PVrel<=1e-5/abs<=1e-4/packet/head/tail/publiccast/determinism. FP32delta-exp/halfweights one-half-neighbor boundary policy. Dense/video diagnostics separate.')
    values=tuple(replace(v,external_reference=str(reference)+'@'+refsha,attributes=v.attributes+(Attribute('delayed_temperature',True),)) if v.id=='out' else v for v in oldmodel.semantic.values)
    model=replace(oldmodel,id='raw-top8-centered-temperature-api',semantic=replace(oldmodel.semantic,values=values,contract=contract),label='Public API model and actual implementation evidence only; no internal native lowering/ideal pipeline certificate')
    modelpath=out/'model.json';modelpath.write_text(model.to_json(pretty=True));files={v.path:v for v in p.inputs};files.pop(str(oldcfg.resolve()))
    for path,role in [(modelpath,'input'),(manifest,'input'),(m['library_path'],'runtime')]+[(v['path'],'source') for v in m['source_files']]:
        f=identify(path,role);files[f.path]=f
    for path in list((project/'scripts').glob('*.py'))+[Path(__file__).with_name(x) for x in ('__init__.py','checks.py','execution.py','plan.py')]:
        f=identify(path,'harness');files[f.path]=f
    cann=Path(c['cann_root'])/'aarch64-linux/asc'
    for path in (cann/'include/basic_api/kernel_operator_vec_reduce_intf.h',cann/'impl/basic_api/kernel_operator_vec_reduce_intf_impl.h',cann/'impl/basic_api/dav_c220/kernel_operator_vec_reduce_impl.h'):
        f=identify(path,'tool');files[f.path]=f
    cfg=out/'attention.json';c.update(project_root=str(project),model=str(modelpath),candidate_manifest=str(manifest),fma_mode=a.mode,native_libraries=libraries,
        identities=[dict(path=f.path,sha256=f.sha256) for f in files.values() if f.role in ('source','harness','runtime')])
    cfg.write_text(json.dumps(c,indent=2));files[str(cfg)]=identify(cfg,'input');env['VC_ATTENTION_INCUMBENT_MANIFEST']=incumbent_manifest
    stages=tuple(s if s.kind=='guard' else replace(s,argv=(s.argv[0],'-m','plugins.vc_attention_e122.execution','--config',str(cfg),'--stage',s.kind)) for s in p.stages)
    scope='complete prepared attention, device task sum'
    p=replace(p,id=out.name,output_dir=str(out/'run'),model_fingerprint=model.fingerprint(),contract=contract.id,inputs=tuple(files.values()),stages=stages,
        env=tuple(Attribute(k,v) for k,v in sorted(env.items())),metric_bindings=(MetricBinding('candidate_device_task_sum',model.fingerprint(),scope),MetricBinding('previous_device_task_sum',oldfp,scope)))
    validate_plan(p);(out/'plan.json').write_text(p.to_json(pretty=True));snapshot(p,out)
    (out/'preflight.json').write_text(json.dumps(dict(old_providers=preflight,candidate=evidence),indent=2)+'\n');print(p.fingerprint());return 0

if __name__=='__main__':raise SystemExit(main())
