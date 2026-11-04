"""CPU-only native-source closure preflight for the isolated attention case.

This uses the same strict operator validator as the actual worker. Passing
does not certify runtime registration, device behavior or numerical results.
"""
from pathlib import Path
import argparse,importlib.util,json,tempfile

OPERATIONS=(('sparse_pv','mindie_laser_sparse_pv'),('sparse_finalize','mindie_laser_sparse_finalize'),
    ('sparse_candidates_score_chunk','mindie_laser_sparse_candidates_score_chunk'),('sparse_inline_pv','sparse_inline_pv'),
    ('max_scale_probe','max_scale_probe'),('sparse_postmax_scale','sparse_postmax_scale'))

def load_validator(project):
    file=Path(project)/'scripts/mindie_laser_quant_fused.py'
    spec=importlib.util.spec_from_file_location('_isolated_attention_manifest_validator',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate_preprocess_manifest

def check(project,*,fault_controls=False):
    project=Path(project).resolve();validate=load_validator(project);records=[]
    for op,name in OPERATIONS:
        manifest=project/'results'/(name+'_cann920_build.json');e=validate(manifest,operation=op)
        records.append(dict(operation=op,source_files=len(e['source_files']),library_sha256=e['library_sha256']))
    controls=[]
    if fault_controls:
        original=project/'results/sparse_postmax_scale_cann920_build.json';raw=json.loads(original.read_text())
        with tempfile.TemporaryDirectory(prefix='e116-manifest-control-') as directory:
            path=Path(directory)/'bad.json'
            for field in ('library_sha256','source_files'):
                bad=json.loads(json.dumps(raw))
                if field=='library_sha256':bad[field]='0'*64
                else:bad[field][0]['sha256']='0'*64
                path.write_text(json.dumps(bad))
                try:validate(path,operation='sparse_postmax_scale')
                except ValueError:controls.append(dict(corruption=field,rejected=True))
                else:raise AssertionError('Strict validator accepted corrupted '+field)
            # Reproduce the actual first isolation failure using its validator
            # at an empty project root, without changing real source links.
            other=Path(directory)/'project/scripts';other.mkdir(parents=True)
            (other/'mindie_laser_quant_fused.py').write_bytes((project/'scripts/mindie_laser_quant_fused.py').read_bytes())
            missing=load_validator(other.parent)
            try:missing(original,operation='sparse_postmax_scale')
            except ValueError as exc:
                if 'Unexpected/duplicate' not in str(exc):raise
                controls.append(dict(corruption='missing_native_source_root',rejected=True))
            else:raise AssertionError('Isolation source-root failure was not reproduced')
    return dict(status='passed',scope='CPU native path/source/library closure only; runtime and numerical behavior unverified',records=records,fault_controls=controls)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--fault-controls',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args()
    result=check(a.project,fault_controls=a.fault_controls)
    with Path(a.output).open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
