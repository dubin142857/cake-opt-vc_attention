"""Audited Cube-pipeline derivative of the unchanged half-sum reference."""
from pathlib import Path
import hashlib
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import prepare_mindie_laser_global_half_sum_source as parent
from scripts import mindie_laser_offline_cache as cache
from scripts.prepare_mindie_laser_global_probe_source import once
NAMES=(*parent.NAMES,'global_cube.h')


def expected_files():
    files={n:b.decode() for n,b in parent.expected_files().items()}
    kernel=files['global_kernel_impl.h']
    begin='#ifdef __DAV_C220_CUBE__\nclass GlobalCube {'
    end='\n#endif\n\n#ifdef __DAV_C220_VEC__'
    if kernel.count(begin)!=1 or kernel.count(end)!=1:raise ValueError('Ambiguous Cube/Vector boundary')
    first=kernel.index(begin)+len('#ifdef __DAV_C220_CUBE__\n');last=kernel.index(end,first)
    kernel=once(kernel,kernel[first:last],'#include "global_cube.h"')
    if kernel[kernel.index('#ifdef __DAV_C220_VEC__'):]!=files['global_kernel_impl.h'][files['global_kernel_impl.h'].index('#ifdef __DAV_C220_VEC__'):]:
        raise ValueError('Pipeline must preserve Vector bytes')
    files['global_kernel_impl.h']=kernel
    files['global_cube.h']=(ROOT/'ascend/laser_global_pipe/global_cube.h').read_text()
    files['global_torch.cpp']=once(files['global_torch.cpp'],'global_attention_half_sum\\\"','global_attention_pipe\\\"')
    files['global_torch.cpp']=once(files['global_torch.cpp'],'serial_cube_primitives\\\":true','serial_cube_primitives\\\":false')
    rename={n:n.replace('vc_laser_global_half_sum','vc_laser_global_pipe') for n in
            ('vc_laser_global_half_sum','vc_laser_global_half_sum_fp16','vc_laser_global_half_sum_bf16',
             'aclrtlaunch_vc_laser_global_half_sum_fp16','aclrtlaunch_vc_laser_global_half_sum_bf16')}
    rename['vc_global_half_sum_kernel']='vc_global_pipe_kernel'
    pattern=re.compile(r'\b('+'|'.join(map(re.escape,rename))+r')\b')
    return {n:pattern.sub(lambda m:rename[m.group()],s).encode() for n,s in files.items()}


def inputs():
    paths=[Path(__file__).resolve(),ROOT/'ascend/laser_global_pipe/global_cube.h',Path(parent.__file__).resolve()]
    return parent.inputs()+[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]


def prepare(directory):
    cache.checked_generated_source(directory,expected_files(),allowed_root=ROOT/'build',materialize=True)
    return inputs()


def validate(directory,recorded_inputs):
    if recorded_inputs!=inputs():raise ValueError('Pipeline generator inputs changed')
    return cache.checked_generated_source(directory,expected_files(),allowed_root=ROOT/'build')
