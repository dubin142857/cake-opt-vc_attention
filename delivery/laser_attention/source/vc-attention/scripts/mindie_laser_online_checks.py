"""Exact score/source obligations for online block-max Top8; ties are a relation."""
def compare_online_states(actual,reference,g,torch):
    from mindie_laser_sparse_candidates import validate_producer_counters
    from mindie_laser_sparse_reference import _error
    validate_producer_counters(actual[1],g)
    rg=dict(g,columns=(g['kpad']//128+31)//32*32,producer_counter_version=3)
    validate_producer_counters(reference[1],rg)
    scores=actual[2][2].cpu().contiguous();want=reference[2][2].cpu().contiguous()
    if scores.shape!=want.shape or scores.dtype!=want.dtype or not torch.equal(scores.view(torch.uint8),want.view(torch.uint8)):
        raise AssertionError('Online global Top8 score bits differ')
    ids=actual[2][0].cpu().long();rows=g['heads']*g['qwork'];rank=g['rank']
    if ids.shape!=(rows,rank):raise AssertionError('Online ID extent')
    heads=torch.arange(rows)//g['qwork'];query=torch.arange(rows)%g['qwork']
    if (ids<0).any() or (ids>=g['heads']*g['kpad']).any() or not (ids//g['kpad']==heads[:,None]).all() or (ids%g['kpad']>=g['keys']).any():
        raise AssertionError('Online key/head range')
    blocks=(ids%g['kpad'])//128
    source_ids=reference[1][1].cpu()[heads[:,None],blocks,query[:,None]].long()
    source_scores=reference[1][0].cpu()[heads[:,None],blocks,query[:,None]].contiguous()
    if not torch.equal(source_ids,ids) or not torch.equal(source_scores.view(torch.uint8),scores.view(torch.uint8)):
        raise AssertionError('Online pair absent from original block maximum')
    if rank>1 and (ids.sort(-1).values[:,1:]==ids.sort(-1).values[:,:-1]).any():raise AssertionError('Online duplicate key')
    compact=actual[1][0].cpu()
    if compact.shape!=(rows,8) or not torch.equal(compact[:,:rank].contiguous().view(torch.uint8),scores.view(torch.uint8)):
        raise AssertionError('Online direct score layout')
    candidate=actual[1][1].cpu()
    if candidate.shape!=(g['heads'],32,g['qwork']) or not (candidate[:,8:]==-1).all():raise AssertionError('Online candidate padding')
    direct_ids=candidate[:,:rank].transpose(1,2).reshape(rows,rank).long()
    if not torch.equal(direct_ids,ids):raise AssertionError('Online finalizer ID mapping')
    weights=actual[2][1].cpu();expected=scores.float().softmax(-1).half()
    maxw=float((weights.float()-expected.float()).abs().max());sumw=float((weights.float().sum(-1)-1).abs().max())
    if not torch.isfinite(weights).all() or (weights<0).any() or maxw>1e-3 or sumw>1e-3:raise AssertionError('Online softmax obligation')
    public=actual[0].cpu()
    cast=actual[3][0].cpu().view(public.shape[0],public.shape[1],g['qwork'],128)[:,:,:g['queries'],:public.shape[3]].to(public.dtype).contiguous()
    if not torch.equal(public.contiguous().view(torch.uint8),cast.view(torch.uint8)):raise AssertionError('Online public cast')
    return dict(scope='All descending global block-max Top8 score bits exact; each ID/score pair is an original E97 per-block maximum. Tied IDs need not match torch.TopK. Own-weight PV/packet checked separately.',
        score_bit_differences=0,source_pair_differences=0,weights_max_abs=maxw,weights_sum_max_abs=sumw,
        public_vs_previous=_error(public,reference[0].cpu()),pv_vs_previous=_error(actual[3][0].cpu(),reference[3][0].cpu()))
