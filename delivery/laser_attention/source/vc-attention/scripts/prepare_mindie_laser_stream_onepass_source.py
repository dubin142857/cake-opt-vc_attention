"""Q64 online controls: two passes or one K2048 UB-resident score pass."""
from pathlib import Path
import hashlib
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import prepare_mindie_laser_global_stream_source as parent
from scripts import mindie_laser_offline_cache as cache
from scripts.prepare_mindie_laser_global_probe_source import once
NAMES=parent.NAMES
OPERATIONS=('global_stream_q64','global_stream_onepass')


def geometry(operation):
    if operation not in OPERATIONS:raise ValueError('Unsupported one-pass/control operation')
    return dict(q_rows=64,vector_rows=32,ub_bytes=182272 if operation.endswith('onepass') else 83968,
                stream_k_blocks=16,onepass=operation.endswith('onepass'))


def expected_files(operation):
    g=geometry(operation)
    files={n:b.decode() for n,b in parent.expected_files().items()}
    contract=files['global_contract.h']
    for old,new in [('kRows = 128','kRows = 64'),('kVectorRows = 64','kVectorRows = 32'),
                    ('kVectorUbBytes == 167936',f'kVectorUbBytes == {g["ub_bytes"]}')]:
        contract=once(contract,old,new)
    if g['onepass']:
        contract=once(contract,'constexpr uint32_t kStreamBlocks = 16;',
            'constexpr uint32_t kStreamBlocks = 16;\nconstexpr uint32_t kStreamElements = kVectorRows * 128 * kStreamBlocks;')
        contract=once(contract,'kVectorUbBytes = kChunkElements * 2','kVectorUbBytes = kStreamElements * 2')
        vector=files['global_stream_vector.h']
        for old,new in [('pipe.InitBuffer(input_,kChunkElements*2);','pipe.InitBuffer(input_,kStreamElements*2);'),
                        ('auto active=rowMax[6*kVectorRows];auto oldMax=rowMax[7*kVectorRows];',
                         'auto active=rowMax[18*kVectorRows];auto oldMax=rowMax[20*kVectorRows];'),
                        ('auto sums=rows_.Get<float>()[4*kVectorRows];','auto sums=rows_.Get<float>()[11*kVectorRows];')]:
            vector=once(vector,old,new)
        begin='                Muls(oldMax,rowMax,half(1),kVectorRows);'
        end='                CrossCoreSetFlag<2,PIPE_MTE3>(kProbabilityReady);'
        if vector.count(begin)!=1 or vector.count(end)!=1:raise ValueError('Ambiguous stream softmax body')
        first,last=vector.index(begin),vector.index(end)
        vector=once(vector,vector[first:last],(ROOT/'ascend/laser_stream_onepass/softmax_body.h').read_text())
        files['global_stream_vector.h']=vector
    files['global_contract.h']=contract
    host=files['global_torch.cpp']
    for old,new in [('contract_version\\\":4','contract_version\\\":5'),
                    ('global_attention_stream\\\"',f'global_attention_{operation.removeprefix("global_")}\\\"'),
                    ('q_rows\\\":128','q_rows\\\":64'),('vector_rows\\\":64','vector_rows\\\":32'),
                    ('ub_bytes\\\":167936',f'ub_bytes\\\":{g["ub_bytes"]}'),
                    ('online_softmax\\\":true','online_softmax\\\":true,\\\"onepass\\\":'+str(g['onepass']).lower())]:
        host=once(host,old,new)
    files['global_torch.cpp']=host
    rename={n:n.replace('vc_laser_global_stream','vc_laser_'+operation) for n in
            ('vc_laser_global_stream','vc_laser_global_stream_fp16','vc_laser_global_stream_bf16',
             'aclrtlaunch_vc_laser_global_stream_fp16','aclrtlaunch_vc_laser_global_stream_bf16')}
    rename['vc_global_stream_kernel']='vc_'+operation+'_kernel'
    pattern=re.compile(r'\b('+'|'.join(map(re.escape,rename))+r')\b')
    return {n:pattern.sub(lambda m:rename[m.group()],s).encode() for n,s in files.items()}


def inputs():
    paths=[Path(__file__).resolve(),ROOT/'ascend/laser_stream_onepass/softmax_body.h']
    return parent.inputs()+[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]


def prepare(directory,operation):
    cache.checked_generated_source(directory,expected_files(operation),allowed_root=ROOT/'build',materialize=True)
    return inputs()


def validate(directory,recorded_inputs,operation):
    if recorded_inputs!=inputs():raise ValueError('One-pass generator inputs changed')
    return cache.checked_generated_source(directory,expected_files(operation),allowed_root=ROOT/'build')
