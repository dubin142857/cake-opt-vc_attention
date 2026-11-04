"""Stronger bitwise obligations for the FMA rewrite, not a looser PV threshold."""
def compare_incumbent(actual,previous,torch):
    pairs=[('public',actual[0],previous[0]),('fp32_pv',actual[3][0],previous[3][0]),('pv_counter',actual[3][1],previous[3][1]),
        ('producer_score',actual[1][0],previous[1][0]),('producer_ids',actual[1][1],previous[1][1]),('producer_counter',actual[1][2],previous[1][2]),
        ('selected_ids',actual[2][0],previous[2][0]),('weights',actual[2][1],previous[2][1]),('scores',actual[2][2],previous[2][2]),
        ('packet_ids',actual[4].indices,previous[4].indices),('packet_weights',actual[4].weights,previous[4].weights)]
    result={}
    for name,a,b in pairs:
        a,b=a.cpu().contiguous(),b.cpu().contiguous()
        if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('FMA incumbent shape/dtype: '+name)
        wrong=int((a.view(torch.uint8)!=b.view(torch.uint8)).sum());result[name]=wrong
        if wrong:raise AssertionError('FMA bitwise incumbent mismatch: '+name+' '+str(wrong))
    return result

FIELDS={'public','fp32_pv','pv_counter','producer_score','producer_ids','producer_counter','selected_ids','weights','scores','packet_ids','packet_weights'}
