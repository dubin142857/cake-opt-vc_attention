"""Independent FP16 chunk-rowsum variant; original global sources stay intact."""
from pathlib import Path
import hashlib
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import mindie_laser_offline_cache as cache
from scripts.prepare_mindie_laser_global_probe_source import NAMES,once


def expected_files():
    files={n:(ROOT/'ascend/laser_global'/n).read_text() for n in NAMES}
    old='''                        Cast(wide,data,RoundMode::CAST_NONE,kChunkElements); PipeBarrier<PIPE_V>();
                        for(uint32_t b=1;b<kChunkBlocks;++b){
                            Add(wide,wide,wide[b*kVectorRows*128],kVectorRows*128); PipeBarrier<PIPE_V>();
                        }
                        WholeReduceSum(sum0,wide,64,kVectorRows,1,1,16);
                        WholeReduceSum(sum1,wide[64],64,kVectorRows,1,1,16); PipeBarrier<PIPE_V>();
                        Add(sum0,sum0,sum1,16); PipeBarrier<PIPE_V>();'''
    new='''                        // FP16 vector width for chunk-local sums only. P stays unchanged.
                        auto halfSum=wide_.Get<half>();
                        Add(halfSum,data,data[kVectorRows*128],kVectorRows*128); PipeBarrier<PIPE_V>();
                        for(uint32_t b=2;b<kChunkBlocks;++b){
                            Add(halfSum,halfSum,data[b*kVectorRows*128],kVectorRows*128); PipeBarrier<PIPE_V>();
                        }
                        WholeReduceSum(newMax,halfSum,128,kVectorRows,1,1,8); PipeBarrier<PIPE_V>();
                        Cast(sum0,newMax,RoundMode::CAST_NONE,16); PipeBarrier<PIPE_V>();'''
    files['global_kernel_impl.h']=once(files['global_kernel_impl.h'],old,new)
    files['global_torch.cpp']=once(files['global_torch.cpp'],'contract_version\\\":1','contract_version\\\":2')
    files['global_torch.cpp']=once(files['global_torch.cpp'],'global_attention\\\"','global_attention_half_sum\\\"')
    rename={n:n.replace('vc_laser_global','vc_laser_global_half_sum') for n in
            ('vc_laser_global','vc_laser_global_fp16','vc_laser_global_bf16',
             'aclrtlaunch_vc_laser_global_fp16','aclrtlaunch_vc_laser_global_bf16')}
    rename['vc_global_kernel']='vc_global_half_sum_kernel'
    pattern=re.compile(r'\b('+'|'.join(map(re.escape,rename))+r')\b')
    return {n:pattern.sub(lambda m:rename[m.group()],text).encode() for n,text in files.items()}


def inputs():
    paths=[ROOT/'ascend/laser_global'/n for n in NAMES]+[
        Path(__file__).resolve(),Path(cache.__file__).resolve(),ROOT/'scripts/prepare_mindie_laser_global_probe_source.py']
    return [{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]


def prepare(directory):
    cache.checked_generated_source(directory,expected_files(),allowed_root=ROOT/'build',materialize=True)
    return inputs()


def validate(directory,recorded_inputs):
    if recorded_inputs!=inputs():raise ValueError('Half-sum generator inputs changed')
    return cache.checked_generated_source(directory,expected_files(),allowed_root=ROOT/'build')
