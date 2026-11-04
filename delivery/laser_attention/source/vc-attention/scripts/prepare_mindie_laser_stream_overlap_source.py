"""Two-slot lookahead with unchanged E81 online math and local Cube phases."""
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


def cube(text):
    run='public:\n    __aicore__ inline void Run(VC_GLOBAL_ARGS) {'
    constants='        constexpr uint32_t aSlot=kRows*128,bSlot=128*128,cSlot=kRows*128;'
    loop='        for (uint64_t task = core; task < heads*qtiles; task += cores) {'
    qbegin='            DataCopy(a1,qGm[(head*qpad+qr)*128],ndQ);'
    qend='            CrossCoreSetFlag<2,PIPE_FIX>(kScoreReady);'
    pbegin='            DataCacheCleanAndInvalid<uint16_t,CacheLine::ENTIRE_DATA_CACHE>(flagGm);'
    pend='            CrossCoreSetFlag<2,PIPE_FIX>(kOutputReady);'
    tail='        for(uint32_t i=0;i<kCounterStride;++i)counterGm.SetValue'
    for anchor in (run,constants,loop,qbegin,qend,pbegin,pend,tail):
        if text.count(anchor)!=1:raise ValueError('Ambiguous Cube phase: '+anchor)
    prefix=text[:text.index(run)]
    prefix=once(prefix,'// Bounded-K online variant. Each chunk drains local events and pairs\n// all four cross-core publications before reusing the ring.',
                '// One-chunk lookahead. Local phase bodies are copied from E81;\n// score/P slot ownership and raw acknowledgements are explicit.')
    view='''    struct View {
        LocalTensor<half> a1,b1,a2,b2;
        LocalTensor<float> acc;
        GlobalTensor<half> qGm,kGm,vGm,scoreGm;
        GlobalTensor<float> rawGm;
        GlobalTensor<uint16_t> flagGm;
        uint64_t core,head;
        uint32_t qr,qpad,kpad,flagpad,skipZero;
        uint64_t* count;
    };
'''
    aliases='''        auto a1=x.a1;auto b1=x.b1;auto a2=x.a2;auto b2=x.b2;auto acc=x.acc;
        auto& qGm=x.qGm;auto& kGm=x.kGm;auto& vGm=x.vGm;auto& scoreGm=x.scoreGm;
        auto& rawGm=x.rawGm;auto& flagGm=x.flagGm;
        const uint64_t core=x.core,head=x.head;
        const uint32_t qr=x.qr,qpad=x.qpad,kpad=x.kpad,flagpad=x.flagpad,skipZero=x.skipZero;
        auto* count=x.count;
'''
    params=text[text.index(constants):text.index(loop)]
    methods=''
    for name,begin,end,expected in [('Qk',qbegin,qend,1),('Pv',pbegin,pend,2)]:
        body=text[text.index(begin):text.index(end)]
        old='core*kRows*kStreamBlocks*128'
        if body.count(old)!=expected:raise ValueError('Ambiguous phase score addresses')
        body=body.replace(old,'(core*2+ringSlot)*kRows*kStreamBlocks*128')
        methods+='    __aicore__ inline void '+name+'(View& x,uint32_t start,uint32_t end,uint32_t ringSlot) {\n'+aliases+params+body+'    }\n'
    methods+='''    __aicore__ inline void Consume(View& x,uint32_t chunk) {
        const uint32_t start=chunk*kStreamBlocks;
        const uint32_t blocks=x.kpad/128;
        const uint32_t end=blocks-start<kStreamBlocks?blocks:start+kStreamBlocks;
        const uint32_t ringSlot=chunk&1;
        CrossCoreWaitFlag(kProbabilityReady+ringSlot);
        // P_ready also implies previous raw consumption, but used must be drained.
        if(start)CrossCoreWaitFlag(kOutputUsed);
        Pv(x,start,end,ringSlot);
        CrossCoreSetFlag<2,PIPE_FIX>(kOutputReady);
    }
'''
    setup=text[text.index(run):text.index(constants)]
    schedule='''        View x{a1,b1,a2,b2,acc,qGm,kGm,vGm,scoreGm,rawGm,flagGm,
            core,0,0,qpad,kpad,flagpad,skipZero,count};
        const uint32_t chunks=(kblocks+kStreamBlocks-1)/kStreamBlocks;
        for (uint64_t task = core; task < heads*qtiles; task += cores) {
            x.head=task/qtiles;x.qr=task%qtiles*kRows;++count[0];
            for(uint32_t chunk=0;chunk<chunks;++chunk) {
                const uint32_t start=chunk*kStreamBlocks;
                const uint32_t end=kblocks-start<kStreamBlocks?kblocks:start+kStreamBlocks;
                ++count[7];
                Qk(x,start,end,chunk&1);
                CrossCoreSetFlag<2,PIPE_FIX>(kScoreReady+(chunk&1));
                if(chunk)Consume(x,chunk-1);
            }
            Consume(x,chunks-1);
            CrossCoreWaitFlag(kOutputUsed);
        }
'''
    return prefix+view+methods+setup+schedule+text[text.index(tail):]


def expected_files():
    files={n:b.decode() for n,b in parent.expected_files().items()}
    files['global_cube.h']=cube(files['global_cube.h'])
    old='kScoreReady = 8, kProbabilityReady = 9, kOutputReady = 10, kOutputUsed = 11'
    files['global_contract.h']=once(files['global_contract.h'],old,
        'kScoreReady = 0, kProbabilityReady = 2, kOutputReady = 4, kOutputUsed = 5')
    original=files['global_stream_vector.h'];vector=original
    declaration='                const uint32_t end=kblocks-start<kStreamBlocks?kblocks:start+kStreamBlocks;'
    addition='\n                const uint32_t ringSlot=(start/kStreamBlocks)&1;'
    changes=[(declaration,declaration+addition),('CrossCoreWaitFlag(kScoreReady);','CrossCoreWaitFlag(kScoreReady+ringSlot);'),
             ('CrossCoreSetFlag<2,PIPE_MTE3>(kProbabilityReady);','CrossCoreSetFlag<2,PIPE_MTE3>(kProbabilityReady+ringSlot);')]
    for old,new in changes:vector=once(vector,old,new)
    oldbase='core*kRows*kStreamBlocks*128';newbase='(core*2+ringSlot)*kRows*kStreamBlocks*128'
    if vector.count(oldbase)!=2:raise ValueError('Ambiguous Vector ring addresses')
    vector=vector.replace(oldbase,newbase)
    inverse=vector
    if inverse.count(newbase)!=2:raise ValueError('Ambiguous inverse ring addresses')
    inverse=inverse.replace(newbase,oldbase)
    for old,new in reversed(changes):inverse=once(inverse,new,old)
    if inverse!=original:raise ValueError('Vector changes exceed addresses/channels')
    files['global_stream_vector.h']=vector
    host=files['global_torch.cpp']
    for old,new in [('contract_version\\\":4','contract_version\\\":6'),
                    ('global_attention_stream\\\"','global_attention_stream_overlap\\\"'),
                    ('score_slots_per_cube\\\":1','score_slots_per_cube\\\":2,\\\"lookahead_chunks\\\":1')]:
        host=once(host,old,new)
    old='cores,vc_global::kStreamBlocks,vc_global::kRows,128'
    if host.count(old)!=2:raise ValueError('Ambiguous two-slot allocation')
    files['global_torch.cpp']=host.replace(old,'cores,2,vc_global::kStreamBlocks,vc_global::kRows,128')
    rename={n:n.replace('vc_laser_global_stream','vc_laser_global_stream_overlap') for n in
            ('vc_laser_global_stream','vc_laser_global_stream_fp16','vc_laser_global_stream_bf16',
             'aclrtlaunch_vc_laser_global_stream_fp16','aclrtlaunch_vc_laser_global_stream_bf16')}
    rename['vc_global_stream_kernel']='vc_global_stream_overlap_kernel'
    pattern=re.compile(r'\b('+'|'.join(map(re.escape,rename))+r')\b')
    return {n:pattern.sub(lambda m:rename[m.group()],s).encode() for n,s in files.items()}


def inputs():
    p=Path(__file__).resolve()
    return parent.inputs()+[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())]


def prepare(directory):
    cache.checked_generated_source(directory,expected_files(),allowed_root=ROOT/'build',materialize=True)
    return inputs()


def validate(directory,recorded_inputs):
    if recorded_inputs!=inputs():raise ValueError('Lookahead generator inputs changed')
    return cache.checked_generated_source(directory,expected_files(),allowed_root=ROOT/'build')
