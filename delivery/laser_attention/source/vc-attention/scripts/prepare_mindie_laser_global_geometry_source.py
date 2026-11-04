"""Audited Q64/Q128 derivatives; each row retains the pipe arithmetic.

Contract 3 changes ring/stat geometry, not floating point semantics. All
constants below are validated before materializing any generated file.
"""
from pathlib import Path
import hashlib
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import prepare_mindie_laser_global_pipe_source as parent
from scripts import mindie_laser_offline_cache as cache
from scripts.prepare_mindie_laser_global_probe_source import once
NAMES=parent.NAMES


def geometry(rows):
    if rows not in (64,128):raise ValueError('Only audited Q64/Q128 geometries are supported')
    return dict(q_rows=rows,vector_rows=rows//2,ub_bytes=rows//2*(128*8+3*128*4+64))


def expected_files(rows):
    g=geometry(rows)
    files={n:b.decode() for n,b in parent.expected_files().items()}
    contract=files['global_contract.h']
    for old,new in [('kRows = 32','kRows = '+str(rows)),('kVectorRows = 16','kVectorRows = '+str(rows//2)),
                    ('kChunkElements * 6 + 1024','kChunkElements * 2 + 3 * kVectorRows * 128 * 4 + kVectorRows * 64'),
                    ('kVectorUbBytes == 50176',f'kVectorUbBytes == {g["ub_bytes"]}')]:
        contract=once(contract,old,new)
    files['global_contract.h']=contract
    cube=files['global_cube.h']
    cube=once(cube,'LoadData2DParams(0,8,2,0,0,false,0)','LoadData2DParams(0,8,kRows/16,0,0,false,0)')
    assert cube.count('m<2;')==2
    cube=cube.replace('m<2;','m<kRows/16;')
    files['global_cube.h']=cube
    kernel=files['global_kernel_impl.h']
    changes=[('pipe.InitBuffer(wide_,kChunkElements*4);','pipe.InitBuffer(wide_,3*kVectorRows*128*4);'),
             ('pipe.InitBuffer(rows_,1024);','pipe.InitBuffer(rows_,kVectorRows*64);'),
             ('auto newMax=rowMax[16];','auto newMax=rowMax[kVectorRows];'),
             ('auto coarse=rowMax[32]; auto active=rowMax[96]; auto broadcast=rowMax[256];',
              'auto coarse=rowMax[2*kVectorRows]; auto active=rowMax[6*kVectorRows];'),
             ('auto sums=rows_.Get<float>()[64]; auto sum0=sums[16]; auto sum1=sums[32]; auto max32=sums[48];',
              'auto sums=rows_.Get<float>()[4*kVectorRows]; auto sum0=sums[kVectorRows]; auto max32=sums[3*kVectorRows];'),
             ('Duplicate(rowMax,half(-__builtin_inff()),16);','Duplicate(rowMax,half(-__builtin_inff()),kVectorRows);'),
             ('Duplicate(sums,0.0f,16);','Duplicate(sums,0.0f,kVectorRows);'),
             ('if (pass) { Brcb(broadcast,rowMax,2,BrcbRepeatParams(1,8)); PipeBarrier<PIPE_V>(); }',
              '''if (pass) {
                    // Materialize row maxima: old cross-K repeat strides overflow uint8_t.
                    auto maxTile=wide_.Get<half>()[kVectorRows*128];
                    for(uint32_t column=0;column<8;++column)
                        Brcb(maxTile[column*16],rowMax,kVectorRows/8,BrcbRepeatParams(8,64));
                    PipeBarrier<PIPE_V>();
                }'''),
             ('DataCopyParams{static_cast<uint16_t>(blocks),128,128,0}',
              'DataCopyParams{static_cast<uint16_t>(blocks),kVectorRows*8,kVectorRows*8,0}'),
             ('Max(rowMax,rowMax,newMax,16);','Max(rowMax,rowMax,newMax,kVectorRows);'),
             ('''for(uint32_t r=0;r<kVectorRows;++r)
                            Sub(data[r*128],data[r*128],broadcast[r*16],uint64_t(128),uint8_t(kChunkBlocks),
                                BinaryRepeatParams(1,1,0,128,128,0));''',
              '''for(uint32_t b=0;b<kChunkBlocks;++b)
                            Sub(data[b*kVectorRows*128],data[b*kVectorRows*128],wide_.Get<half>()[kVectorRows*128],kVectorRows*128);'''),
             ('WholeReduceMax(coarse,data,128,kVectorRows*kChunkBlocks,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);',
              '''// The explicit repeat API takes uint8_t: Q128 needs 256 total repeats.
                        for(uint32_t r=0;r<kVectorRows*kChunkBlocks;r+=128){
                            constexpr uint32_t repeat=kVectorRows*kChunkBlocks<128?kVectorRows*kChunkBlocks:128;
                            WholeReduceMax(coarse[r],data[r*128],128,repeat,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
                        }'''),
             ('WholeReduceMax(active,coarse,16,kChunkBlocks,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);',
              'WholeReduceMax(active,coarse,kVectorRows,kChunkBlocks,1,1,kVectorRows/16,ReduceOrder::ORDER_ONLY_VALUE);'),
             ('Cast(sum0,newMax,RoundMode::CAST_NONE,16);','Cast(sum0,newMax,RoundMode::CAST_NONE,kVectorRows);'),
             ('Add(sums,sums,sum0,16);','Add(sums,sums,sum0,kVectorRows);'),
             ('DataCopyParams{static_cast<uint16_t>(blocks),128,0,128}',
              'DataCopyParams{static_cast<uint16_t>(blocks),kVectorRows*8,0,kVectorRows*8}'),
             ('Cast(max32,rowMax,RoundMode::CAST_NONE,16);','Cast(max32,rowMax,RoundMode::CAST_NONE,kVectorRows);'),
             ('DataCopy(statsGm[core*64+sub*16],max32,16);',
              'DataCopy(statsGm[core*2*kRows+sub*kVectorRows],max32,kVectorRows);'),
             ('DataCopy(statsGm[core*64+32+sub*16],sums,16);',
              'DataCopy(statsGm[core*2*kRows+kRows+sub*kVectorRows],sums,kVectorRows);'),
             ('auto denom=wide[4096];','auto denom=wide[256*kVectorRows];'),
             ('Brcb(denom,sums,2,BrcbRepeatParams(16,128));','Brcb(denom,sums,kVectorRows/8,BrcbRepeatParams(16,128));')]
    for old,new in changes:kernel=once(kernel,old,new)
    files['global_kernel_impl.h']=kernel
    host=files['global_torch.cpp']
    for old,new in [('((queries+31)/32)','((queries+vc_global::kRows-1)/vc_global::kRows)'),
                    ('contract_version\\\":2','contract_version\\\":3'),
                    ('global_attention_pipe\\\"',f'global_attention_q{rows}\\\"'),
                    ('q_rows\\\":32',f'q_rows\\\":{rows}'),
                    ('vector_rows\\\":16',f'vector_rows\\\":{rows//2}'),
                    ('ub_bytes\\\":50176',f'ub_bytes\\\":{g["ub_bytes"]}')]:host=once(host,old,new)
    # The same allocation occurs in Meta and device dispatch: assert both sites.
    for old,new,count in [('k.size(2)/128,32,128','k.size(2)/128,vc_global::kRows,128',2),
                          ('{cores,64}','{cores,2*vc_global::kRows}',2),
                          ('{cores,32,128}','{cores,vc_global::kRows,128}',1)]:
        if host.count(old)!=count:raise ValueError('Ambiguous geometry allocation: '+old)
        host=host.replace(old,new)
    files['global_torch.cpp']=host
    rename={n:n.replace('vc_laser_global_pipe',f'vc_laser_global_q{rows}') for n in
            ('vc_laser_global_pipe','vc_laser_global_pipe_fp16','vc_laser_global_pipe_bf16',
             'aclrtlaunch_vc_laser_global_pipe_fp16','aclrtlaunch_vc_laser_global_pipe_bf16')}
    rename['vc_global_pipe_kernel']=f'vc_global_q{rows}_kernel'
    pattern=re.compile(r'\b('+'|'.join(map(re.escape,rename))+r')\b')
    return {n:pattern.sub(lambda m:rename[m.group()],s).encode() for n,s in files.items()}


def inputs():
    p=Path(__file__).resolve()
    return parent.inputs()+[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())]


def prepare(directory,rows):
    cache.checked_generated_source(directory,expected_files(rows),allowed_root=ROOT/'build',materialize=True)
    return inputs()


def validate(directory,recorded_inputs,rows):
    if recorded_inputs!=inputs():raise ValueError('Q-geometry generator inputs changed')
    return cache.checked_generated_source(directory,expected_files(rows),allowed_root=ROOT/'build')
