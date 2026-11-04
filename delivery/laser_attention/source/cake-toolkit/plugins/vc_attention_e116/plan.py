"""Freeze isolated operator sources while retaining original, unchanged guards."""
from dataclasses import replace,asdict
from pathlib import Path
import argparse,json,tarfile
from core.opgen.architecture import Architecture,SemanticProgram,Contract,Value,Storage,Resource,Allocation,Instance,Layout,Event,Region,Demand,Binding,Evidence,Platform
from core.opgen.runner import identify,validate_plan
from plugins.vc_attention.plan import build_plan

def snapshot(p,out):
    snapshot={f.path:f for f in p.inputs if f.role in ('source','harness','tool')}
    for file in (Path(__file__).resolve().parents[2]/'core/opgen').glob('*.py'):snapshot[str(file)]=identify(file,'tool')
    with tarfile.open(out/'sources.tar','w') as archive:
        for path in sorted(snapshot):archive.add(path,arcname=path.lstrip('/'),recursive=False)
    (out/'sources.json').write_text(json.dumps([asdict(f) for f in snapshot.values()],indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--guard-project',required=True);ap.add_argument('--project',required=True);ap.add_argument('--stack',required=True);ap.add_argument('--card',type=int,required=True);ap.add_argument('--output',required=True);ap.add_argument('--sandbox',required=True);a=ap.parse_args()
    stack=json.loads(Path(a.stack).read_text());p=build_plan(a.guard_project,a.output,a.card,stack,a.sandbox,candidate_kind='inline_pv');out=Path(a.output).resolve();project=Path(a.project).resolve()
    contract=Contract('postmax-scale-mechanism','exact','CANN9.2.0 Muls<half>/WholeReduceMax ORDER_VALUE_INDEX','All63488 finite FP16 inputs, fixed half scales D64/128, additional infinities; repeated finite row arrangements','Scalar monotonicity and row composition observations only; attention payload equivalence explicitly disproved by collapsed-tie/V counterexample')
    n=512*128*2;shape=(512,128)
    values=(Value('input','input',shape,'f16'),Value('mapped','half_scale',shape,'f16',('input',),external_reference='Actual Muls half output'))
    model=Architecture('fp16-scale-max-mechanism',SemanticProgram(values,('input',),('mapped',),contract),(Storage('GM','GM','global'),),(Resource('AIV','global',1,Evidence('assumed','Probe API boundary only')),),
        (Allocation('input','GM',n),Allocation('mapped','GM',n)),(Instance('input','input','input',Layout(shape,(256,2))),Instance('mapped','mapped','mapped',Layout(shape,(256,2)))),
        (Event('probe','AIV','compute',(Region('input',0,n),),(Region('mapped',0,n),),(Demand('AIV'),),('mapped',)),),(),(Binding('input',(),('input',)),Binding('mapped',('probe',),('mapped',))),platform=Platform('Ascend910B2'),label='Mechanism evidence only; no timing or attention certificate')
    modelpath=out/'model.json';modelpath.write_text(model.to_json(pretty=True));files={f.path:f for f in p.inputs};files[str(modelpath)]=identify(modelpath,'input')
    manifest=project/'results/max_scale_probe_cann920_build.json';m=json.loads(manifest.read_text())
    for path,role in [(manifest,'input'),(m['library_path'],'runtime')]+[(v['path'],'source') for v in m['source_files']]:
        f=identify(path,role);files[f.path]=f
    for path in list((project/'scripts').glob('*.py'))+list(Path(__file__).parent.glob('*.py')):
        f=identify(path,'harness');files[f.path]=f
    cfg=out/'mechanism.json';c=dict(project_root=str(project),cann_root=str(Path(stack['env']['ASCEND_HOME_PATH']).resolve()),model=str(modelpath),candidate_manifest=str(manifest),device=a.card,
        native_libraries=[dict(path=m['library_path'],sha256=m['library_sha256'])],identities=[dict(path=f.path,sha256=f.sha256) for f in files.values() if f.role in ('source','harness','runtime')])
    cfg.write_text(json.dumps(c,indent=2));files[str(cfg)]=identify(cfg,'input')
    stages=[]
    for stage in p.stages:
        if stage.kind=='guard':stages.append(stage);continue
        outputs={'build':('native.manifest.json',),'check':('check.detail.json','scale-table.pt'),'measure':('measure.detail.json',),'verify':()}[stage.kind]
        stages.append(replace(stage,argv=(stack['python'],'-m','plugins.vc_attention_e116.math_probe','--config',str(cfg),'--stage',stage.kind),outputs=outputs))
    p=replace(p,model_fingerprint=model.fingerprint(),contract=contract.id,workload='FP16 scale/max mechanism D64/128; all63488 finite patterns plus infinities; no attention or timing claim',inputs=tuple(files.values()),stages=tuple(stages),metric_bindings=())
    validate_plan(p);(out/'plan.json').write_text(p.to_json(pretty=True));snapshot(p,out);print(p.fingerprint());return 0

if __name__=='__main__':raise SystemExit(main())
