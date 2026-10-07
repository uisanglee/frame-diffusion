import json

from framediff.api_cost import summarize


def test_api_cost_averages_repeats_before_summing_pages(tmp_path):
    path=tmp_path/'metrics.jsonl'
    rows=[]
    for page in ('a','b'):
        for repeat in (1,2,3):
            rows += [
                {'id':f'{page}/repeat-{repeat}','page_id':page,'method':'initial','vlm_calls':1,
                 'input_tokens':100,'output_tokens':10},
                {'id':f'{page}/repeat-{repeat}','page_id':page,'method':'self-revision-1','vlm_calls':2,
                 'input_tokens':250,'output_tokens':30},
                {'id':f'{page}/repeat-{repeat}','page_id':page,'method':'abstract-policy','vlm_calls':1,
                 'input_tokens':100,'output_tokens':10},
            ]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result={row['method']:row for row in summarize(path,2.5,10)}
    assert result['initial']['pages']==2 and result['initial']['input_tokens']==200
    assert result['self-revision-1']['api_calls']==4
    assert result['abstract-policy']['estimated_usd']==(200*2.5+20*10)/1_000_000
