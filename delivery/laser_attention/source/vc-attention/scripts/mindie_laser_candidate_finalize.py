"""E94 producer + builtin TopK + fused map/softmax/packet/PV; explicit large-C fallback."""
from pathlib import Path
from mindie_laser_candidate_score_chunk import CandidateScoreChunkAttention,load_candidate_score_chunk_attention
from mindie_laser_sparse_finalize import load_sparse_finalize,finalize_geometry,TopkInputs,FinalizedSelection
from mindie_laser_sparse_pv import SparsePvPayload

class CandidateFinalizeAttention(CandidateScoreChunkAttention):
    def __init__(self,producer,finalizer):
        super().__init__(producer.base,producer.native,producer.probe,producer.pv,producer.evidence,producer.metadata,producer.device)
        self.finalizer=finalizer
        self.evidence=dict(finalizer.evidence,producer_dependency=producer.evidence)
        self.metadata=dict(producer.metadata,operation='block1_score8_finalize',finalize=finalizer.metadata,
            stage_scopes=dict(selection='transpose + builtin TopK only',consumer='map + FP32 softmax + packet + PV + trim/cast',fallback='C>512 uses original select/consume'))
    def make_layout(self,heads,queries,keys):
        g=super().make_layout(heads,queries,keys);g['native_finalize']=g['columns']<=512;return g
    def prepare(self,q,k,v):
        payload=super().prepare(q,k,v)
        if payload.layout['native_finalize']:
            key=('finalize',payload.layout['rank'])
            if key not in self.geometry_cache:self.geometry_cache[key]=finalize_geometry(key[1]).to(q.device)
            payload.finalize_geometry=self.geometry_cache[key]
        return payload
    def select(self,payload,state):
        if not payload.layout['native_finalize']:return super().select(payload,state)
        g=payload.layout;rows=state[0].transpose(1,2).contiguous().view(-1,g['columns'])
        scores,positions=rows.topk(g['rank'],dim=-1,sorted=True)
        return TopkInputs(scores,positions,state[1])
    def _finish(self,payload,topk):
        p,g=payload.base,payload.layout;value=p.tensors[2].view(-1,128)
        native=self.finalizer.run_native(value,topk.candidates,topk.scores,topk.positions,payload.finalize_geometry)
        result=(native[0],native[5]);packet=SparsePvPayload(value,native[3],native[4],g['heads']*g['qwork'],g['rank'])
        out=native[0].view(p.shape[0],p.shape[1],g['qwork'],128)[:,:,:g['queries'],:p.shape[3]].to(p.dtype).contiguous()
        selected=FinalizedSelection(native[1],native[2],topk.scores,topk)
        return out,result,packet,selected
    def consume(self,payload,selection):
        if not payload.layout['native_finalize']:return super().consume(payload,selection)
        topk=selection.topk if isinstance(selection,FinalizedSelection) else selection
        out,result,packet,_=self._finish(payload,topk);return out,result,packet
    def run_native(self,payload):
        if not payload.layout['native_finalize']:return super().run_native(payload)
        producer=self.produce(payload);topk=self.select(payload,producer)
        out,result,packet,selected=self._finish(payload,topk)
        return out,producer,selected,result,packet


def load_candidate_finalize_attention(base,manifest,pv_manifest,device=5):
    producer_manifest=Path(__file__).resolve().parents[1]/'results/mindie_laser_sparse_candidates_score_chunk_cann920_build.json'
    producer=load_candidate_score_chunk_attention(base,producer_manifest,pv_manifest,device)
    return CandidateFinalizeAttention(producer,load_sparse_finalize(manifest,device))
