"""Generate an isolated global-attention diagnostic from audited local sources.

Reuse the existing PhaseProbe timestamp/finish machinery; replace only its
old VC tiling/sample policy. Production sources are never edited here.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
NAMES=('CMakeLists.txt','global_contract.h','global_kernel_impl.h',
       'global_kernel.cpp','global_bf16_kernel.cpp','global_torch.cpp')
MAGIC=0x56434750524F4245
FIELDS=64


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Global probe requires one exact anchor: '+repr(old[:100]))
    return text.replace(old,new,1)


def probe_header(text):
    text=once(text,'#include "vc_tiling.h"','')
    text=once(text,'namespace vc910 {','namespace vc910 {\nconstexpr uint32_t PROFILE_FIELDS=64, PROFILE_ROLES=3, PROFILE_SAMPLES=3;')
    start=text.index('    __aicore__ inline void Begin(')
    end=text.index('    __aicore__ inline void Mark(',start)
    text=once(text,text[start:end],'''    __aicore__ inline void Begin(uint64_t task, uint64_t round, uint64_t total, uint32_t mode) {
        active_=false;
        if (!mode || !total || round>=total) return;
        for (uint32_t s=0;s<PROFILE_SAMPLES;++s) {
            const uint64_t target=s==0?0:(s==1?total/2:total-1);
            if (round!=target) continue;
            slot_=s; active_=true; break;
        }
        if (!active_) return;
        for(uint32_t i=0;i<PROFILE_FIELDS;++i) values_[i]=0;
        values_[0]=0x56434750524F4245ULL; values_[1]=task;
        values_[2]=round; values_[3]=total; values_[14]=1; values_[15]=mode;
        values_[16]=role_; values_[17]=core_; values_[18]=slot_;
        if(mode==2) AscendC::PipeBarrier<PIPE_ALL>();
        last_=uint64_t(AscendC::GetSystemCycle()); values_[12]=last_;
    }
    __aicore__ inline void SampleMark(uint32_t phase) {
        if(!active_) return;
        if(values_[15]==2) MarkAll(phase); else Mark(phase);
    }

''')
    text=once(text,'            values_[4 + phase] = now - last_;',
                   '            values_[4 + phase] = now - last_;\n            values_[19] |= uint64_t(1)<<phase;')
    text=once(text,'    __aicore__ inline void Begin(uint64_t, uint32_t, const Tiling&) {}',
                   '    __aicore__ inline void Begin(uint64_t, uint64_t, uint64_t, uint32_t) {}\n    __aicore__ inline void SampleMark(uint32_t) {}')
    return text


def kernel(text):
    text=once(text,'#include "global_contract.h"','#include "global_contract.h"\n#include "vc_phase_probe.h"')
    # Role-specific anchors: the two local loops intentionally have the same text.
    for role,anchor in [(0,'        uint64_t count[8] = {};'),(1,'        const uint32_t kblocks=kpad/128;')]:
        core_role='0' if role==0 else '1+sub'
        setup='''
        const uint32_t profileMode=skipZero>>1; skipZero &= 1;
        vc910::PhaseProbe<true> probe;
        probe.Init(stats + uint64_t((cores*64+127)/128*128)*4,core,'''+core_role+''');
        const uint64_t totalTasks=heads*qtiles;
        const uint64_t localTotal=(totalTasks-1-core)/cores+1;'''
        text=once(text,anchor,anchor+setup)
    loop='        for (uint64_t task = core; task < heads*qtiles; task += cores) {\n'
    if text.count(loop)!=2:raise ValueError('Expected paired Cube/Vector loops')
    # Only these two deliberately identical anchors may be changed together.
    text=text.replace(loop,loop+'            probe.Begin(task,(task-core)/cores,localTotal,profileMode);\n')
    for old,new in [
        ('            CrossCoreSetFlag<2,PIPE_FIX>(kScoreReady);',
         '            CrossCoreSetFlag<2,PIPE_FIX>(kScoreReady);\n            probe.SampleMark(0);'),
        ('            CrossCoreWaitFlag(kProbabilityReady);',
         '            CrossCoreWaitFlag(kProbabilityReady);\n            probe.SampleMark(1);'),
        ('            CrossCoreSetFlag<2,PIPE_FIX>(kOutputReady);',
         '            CrossCoreSetFlag<2,PIPE_FIX>(kOutputReady);\n            probe.SampleMark(2);'),
        ('            CrossCoreWaitFlag(kOutputUsed);',
         '            CrossCoreWaitFlag(kOutputUsed);\n            probe.SampleMark(3); probe.Finish();'),
        ('            CrossCoreWaitFlag(kScoreReady);',
         '            CrossCoreWaitFlag(kScoreReady);\n            probe.SampleMark(0);'),
        ('            Cast(max32,rowMax,RoundMode::CAST_NONE,16);',
         '            Cast(max32,rowMax,RoundMode::CAST_NONE,16);'),
        ('                    }\n                }\n            }\n            Cast(max32',
         '                    }\n                }\n                probe.SampleMark(1+pass);\n            }\n            Cast(max32'),
        ('            CrossCoreSetFlag<2,PIPE_MTE3>(kProbabilityReady);',
         '            CrossCoreSetFlag<2,PIPE_MTE3>(kProbabilityReady);\n            probe.SampleMark(3);'),
        ('            CrossCoreWaitFlag(kOutputReady);',
         '            CrossCoreWaitFlag(kOutputReady);\n            probe.SampleMark(4);'),
        ('            CrossCoreSetFlag<2,PIPE_MTE3>(kOutputUsed);\n            SetFlag<HardEvent::MTE3_V>(m3v); WaitFlag<HardEvent::MTE3_V>(m3v);',
         '            CrossCoreSetFlag<2,PIPE_MTE3>(kOutputUsed);\n            SetFlag<HardEvent::MTE3_V>(m3v); WaitFlag<HardEvent::MTE3_V>(m3v);\n            probe.SampleMark(5); probe.Finish();')]:
        text=once(text,old,new)
    return text


def host(text):
    signature='int64_t queries,int64_t keys,int64_t dimension,bool bf16,bool skipZero)'
    if text.count(signature)!=2:raise ValueError('Expected native+Meta attention signatures')
    text=text.replace(signature,'int64_t queries,int64_t keys,int64_t dimension,bool bf16,bool skipZero,int64_t mode)')
    text=once(text,'    auto cores=Cores(q,queries);','    TORCH_CHECK(mode>=0 && mode<=2,"Probe mode must be 0/1/2");\n    auto cores=Cores(q,queries);')
    text=once(text,'at::empty({cores,64},q.options().dtype(at::kFloat))',
                   'at::empty({(cores*64+127)/128*128+2*cores*3*3*64},q.options().dtype(at::kFloat))')
    text=once(text,'    const uint32_t cores=Cores(q,queries);',
                   '    TORCH_CHECK(mode>=0 && mode<=2,"Probe mode must be 0/1/2");\n    const uint32_t cores=Cores(q,queries);')
    text=once(text,'    auto stats=at_npu::native::empty_with_format({cores,64},options.dtype(at::kFloat),ACL_FORMAT_ND);',
'''    const int64_t statElements=(cores*64+127)/128*128+2*cores*3*3*64;
    auto statsStorage=at_npu::native::empty_with_format({statElements+128},options.dtype(at::kFloat),ACL_FORMAT_ND);
    const auto statOffset=((512-reinterpret_cast<uintptr_t>(statsStorage.data_ptr())%512)%512)/4;
    auto stats=statsStorage.narrow(0,statOffset,statElements);
    stats.zero_(); // All diagnostic modes use the same allocation and zero fill.''')
    text=once(text,'uint32_t(skipZero));','uint32_t(skipZero)|(uint32_t(mode)<<1));')
    text=once(text,'bool skip_zero) -> (Tensor, Tensor, Tensor, Tensor)',
                   'bool skip_zero, int mode) -> (Tensor, Tensor, Tensor, Tensor)')
    # This is a string value, not a short token substitution in CMake commands.
    text=once(text,'global_attention\\\"','global_attention_profile\\\"')
    return text


def expected_files():
    data={name:(ROOT/'ascend/laser_global'/name).read_text() for name in NAMES}
    data['global_kernel_impl.h']=kernel(data['global_kernel_impl.h'])
    data['global_torch.cpp']=host(data['global_torch.cpp'])
    data['vc_phase_probe.h']=probe_header((ROOT/'ascend/csrc/vc_phase_probe.h').read_text())
    rename={'vc_laser_global':'vc_laser_global_profile','vc_global_kernel':'vc_global_profile_kernel',
            'vc_laser_global_fp16':'vc_laser_global_profile_fp16','vc_laser_global_bf16':'vc_laser_global_profile_bf16',
            'aclrtlaunch_vc_laser_global_fp16':'aclrtlaunch_vc_laser_global_profile_fp16',
            'aclrtlaunch_vc_laser_global_bf16':'aclrtlaunch_vc_laser_global_profile_bf16'}
    pattern=re.compile(r'\b('+'|'.join(map(re.escape,rename))+r')\b')
    return {name:pattern.sub(lambda m:rename[m.group()],text).encode() for name,text in data.items()}


def inputs():
    paths=[ROOT/'ascend/laser_global'/name for name in NAMES]+[ROOT/'ascend/csrc/vc_phase_probe.h',Path(__file__).resolve()]
    return [{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]


def prepare(directory):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ROOT/'build'):raise ValueError('Diagnostic source must stay in workspace build/')
    data=expected_files()
    directory.mkdir(parents=True,exist_ok=True)
    if any(p.is_symlink() for p in directory.rglob('*')):raise ValueError('Diagnostic source symlinks forbidden')
    unexpected=[p for p in directory.rglob('*') if p.is_file() and p.name not in data]
    if unexpected:raise ValueError('Unexpected diagnostic source files: '+str(unexpected))
    for name,value in data.items():
        p=directory/name
        if p.exists() and p.read_bytes()!=value:raise ValueError('Preserve stale diagnostic source; use a new build directory: '+str(p))
        if not p.exists():p.write_bytes(value)
    return inputs()


def validate(directory,recorded_inputs):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ROOT/'build') or inputs()!=recorded_inputs:
        raise ValueError('Diagnostic input identity changed')
    data=expected_files()
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual!=set(data):raise ValueError('Diagnostic source file set changed')
    if any(p.is_symlink() for p in directory.rglob('*')):raise ValueError('Diagnostic source symlinks forbidden')
    if any((directory/name).read_bytes()!=value for name,value in data.items()):raise ValueError('Diagnostic source bytes changed')
    return set(directory/name for name in data)
