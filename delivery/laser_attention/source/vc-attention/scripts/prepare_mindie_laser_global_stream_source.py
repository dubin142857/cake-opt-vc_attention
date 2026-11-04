"""Independent bounded-K score ring with FP32 online output recurrence."""
from pathlib import Path
import hashlib
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import prepare_mindie_laser_global_geometry_source as parent
from scripts import mindie_laser_offline_cache as cache
from scripts.prepare_mindie_laser_global_probe_source import once
NAMES=(*parent.NAMES,'global_stream_vector.h')
STREAM_BLOCKS=16


def expected_files():
    files={n:b.decode() for n,b in parent.expected_files(128).items()}
    files['global_contract.h']=once(files['global_contract.h'],'constexpr uint32_t kColumns = 128;',
        f'constexpr uint32_t kColumns = 128;\nconstexpr uint32_t kStreamBlocks = {STREAM_BLOCKS};')
    cube=files['global_cube.h']
    changes=[('// Included only by the generated C220 Cube branch. Vector is byte-identical\n// to the half-sum reference. Each phase returns every allocated event to zero.',
              '// Bounded-K online variant. Each chunk drains local events and pairs\n// all four cross-core publications before reusing the ring.'),
             ('uint32_t blocks,uint32_t begin,uint32_t skip,uint64_t* count)',
              'uint32_t blocks,uint32_t begin,uint32_t first,uint32_t skip,uint64_t* count)'),
             ('if(skip && zero && kb)', 'if(skip && zero && kb!=first)'),
             ('if(!kb && zero)', 'if(kb==first && zero)'),
             ('            DataCopy(a1,qGm[(head*qpad+qr)*128],ndQ);',
              '''            for(uint32_t start=0;start<kblocks;start+=kStreamBlocks) {
            const uint32_t end=kblocks-start<kStreamBlocks?kblocks:start+kStreamBlocks;
            ++count[7];
            DataCopy(a1,qGm[(head*qpad+qr)*128],ndQ);'''),
             ('DataCopy(b1,kGm[head*kpad*128],ndK);','DataCopy(b1,kGm[(head*kpad+uint64_t(start)*128)*128],ndK);'),
             ('for(uint32_t kb=0;kb<kblocks;++kb)', 'for(uint32_t kb=start;kb<end;++kb)'),
             ('if(kb+1<kblocks)', 'if(kb+1<end)'),
             ('NextPv(flagGm,core,flagpad,kblocks,0,skipZero,count)', 'NextPv(flagGm,core,flagpad,end,start,start,skipZero,count)'),
             ('while(kb<kblocks)', 'while(kb<end)'),
             ('NextPv(flagGm,core,flagpad,kblocks,kb+1,skipZero,count)', 'NextPv(flagGm,core,flagpad,end,kb+1,start,skipZero,count)'),
             ('if(nextKb<kblocks)', 'if(nextKb<end)'),
             ('false,kb==0)', 'false,kb==start)'),
             ('            CrossCoreWaitFlag(kOutputUsed);\n        }',
              '            CrossCoreWaitFlag(kOutputUsed);\n            } // bounded K chunk\n        }')]
    for old,new in changes:cube=once(cube,old,new)
    for index in ('kb','nextKb'):
        old=f'core*kRows*kpad+uint64_t({index})*kRows*128'
        new=f'core*kRows*kStreamBlocks*128+uint64_t({index}-start)*kRows*128'
        expected=2 if index=='kb' else 1
        if cube.count(old)!=expected:raise ValueError('Ambiguous score ring address')
        cube=cube.replace(old,new)
    files['global_cube.h']=cube
    kernel=files['global_kernel_impl.h']
    begin='#ifdef __DAV_C220_VEC__\ntemplate <typename Output>\nclass GlobalVector {'
    end='\n#endif\n\ntemplate <typename Output>\n__aicore__ inline void RunGlobal'
    if kernel.count(begin)!=1 or kernel.count(end)!=1:raise ValueError('Ambiguous Vector boundary')
    first=kernel.index(begin)+len('#ifdef __DAV_C220_VEC__\n');last=kernel.index(end,first)
    files['global_kernel_impl.h']=once(kernel,kernel[first:last],'#include "global_stream_vector.h"')
    files['global_stream_vector.h']=(ROOT/'ascend/laser_global_stream/global_stream_vector.h').read_text()
    host=files['global_torch.cpp']
    for old,new in [('contract_version\\\":3','contract_version\\\":4'),
                    ('global_attention_q128\\\"','global_attention_stream\\\"'),
                    ('score_slots_per_cube\\\":1','score_slots_per_cube\\\":1,\\\"stream_k_blocks\\\":16,\\\"online_softmax\\\":true')]:
        host=once(host,old,new)
    old='k.size(2)/128,vc_global::kRows,128';new='vc_global::kStreamBlocks,vc_global::kRows,128'
    if host.count(old)!=2:raise ValueError('Ambiguous bounded score allocation')
    files['global_torch.cpp']=host.replace(old,new)
    rename={n:n.replace('vc_laser_global_q128','vc_laser_global_stream') for n in
            ('vc_laser_global_q128','vc_laser_global_q128_fp16','vc_laser_global_q128_bf16',
             'aclrtlaunch_vc_laser_global_q128_fp16','aclrtlaunch_vc_laser_global_q128_bf16')}
    rename['vc_global_q128_kernel']='vc_global_stream_kernel'
    pattern=re.compile(r'\b('+'|'.join(map(re.escape,rename))+r')\b')
    return {n:pattern.sub(lambda m:rename[m.group()],s).encode() for n,s in files.items()}


def inputs():
    paths=[Path(__file__).resolve(),ROOT/'ascend/laser_global_stream/global_stream_vector.h']
    return parent.inputs()+[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]


def prepare(directory):
    cache.checked_generated_source(directory,expected_files(),allowed_root=ROOT/'build',materialize=True)
    return inputs()


def validate(directory,recorded_inputs):
    if recorded_inputs!=inputs():raise ValueError('Online-stream generator inputs changed')
    return cache.checked_generated_source(directory,expected_files(),allowed_root=ROOT/'build')
