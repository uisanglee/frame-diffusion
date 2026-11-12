#!/usr/bin/env python3
"""Portable target / initial / TUIDE gallery, using saved PNGs only (stdlib)."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil


def read_rows(path):
    if not path.is_file():return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:20]


def first_file(paths):
    return next((Path(p) for p in paths if p and Path(p).is_file()),None)


def export(run,out,repeat=1):
    run=Path(run).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    assets=out/'assets';assets.mkdir(exist_ok=True)
    cards=[];records=[]
    def image(path,label):
        if path is None:return '<div class="missing">저장된 이미지 없음</div>'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        destination=assets/(digest+path.suffix.lower())
        if not destination.exists():shutil.copyfile(path,destination)
        relative='assets/'+destination.name
        return f'<a href="{relative}" target="_blank"><img loading="lazy" src="{relative}" alt="{label}"></a>'
    manifests=sorted(run.glob('*/ */seeds/*/prepare/prepared.jsonl'.replace(' ','')))
    if not manifests:raise ValueError(f'No dataset/model/seeds/seed/prepare/prepared.jsonl under {run}')
    for manifest in manifests:
        base=manifest.parent.parent
        dataset,model,_,seed=base.relative_to(run).parts
        repairs={r['page_id'] if r.get('page_id') else r['id'].rsplit('/repeat-',1)[0]:r
                 for r in read_rows(base/'repair/results.jsonl') if r.get('repeat',1)==repeat}
        metrics={(r['id'],r['method']):r for r in read_rows(base/'evaluation/metrics.jsonl')}
        for source in read_rows(manifest):
            id=source['id'];trial=repairs.get(id);trial_id=trial['id'] if trial else f'{id}/repeat-{repeat}'
            initial=source.get('methods',{}).get('initial',{})
            method=(trial or {}).get('methods',{}).get('abstract-policy')
            initial_failed=bool(initial.get('failed') or source.get('errors',{}).get('initial'))
            work=base/'evaluation/pages'/signature(trial_id)
            target=first_file([source.get('screenshot')])
            # Never use *-target.png: it is a detector abstraction, not the target RGB.
            initial_png=None if initial_failed else first_file([
                work/'initial.png',base/'evaluation/pages'/signature(id)/'initial.png',
                source.get('initial_screenshot'),
                Path(initial['html']).parent/'initial.png' if initial.get('html') else None])
            repaired=None if initial_failed or method is None else first_file([work/'abstract-policy.png'])
            errors=[]
            if initial_failed:errors.append('Initial 생성 실패: '+str(initial.get('error') or source.get('errors',{}).get('initial')))
            elif method is None:errors.append('TUIDE 결과 기록 없음')
            elif method.get('failed'):errors.append('TUIDE 실패/보존 결과: '+str(method.get('error')))
            for label,path in [('Target',target),('Initial',initial_png),('TUIDE',repaired)]:
                if path is None:errors.append(label+' 저장 PNG 없음 (재렌더링하지 않음)')
            title=f'{dataset} / {model} / seed {seed} / {id}'
            item=dict(dataset=dataset,model=model,seed=seed,id=id,repeat=repeat,
                      images={k:str(v) if v else None for k,v in [('target',target),('initial',initial_png),('tuide',repaired)]},
                      errors=errors,metrics={k:metrics.get((trial_id,k),metrics.get((id,k),{})) for k in ('initial','abstract-policy')})
            records.append(item)
            def score(name):
                m=item['metrics'][name]
                return ' | '.join(f'{key}: {m[key]:.4f}' for key in ('official_position','official_clip','box_iou')
                                  if isinstance(m.get(key),(int,float)))
            figures=''.join(f'<figure><figcaption>{label}</figcaption>{image(path,label)}<p>{html.escape(score(name)) if name else ""}</p></figure>'
                for label,path,name in [('Target',target,None),('VLM Initial',initial_png,'initial'),('TUIDE',repaired,'abstract-policy')])
            cards.append(f'<article data-dataset="{html.escape(dataset,quote=True)}" data-model="{html.escape(model,quote=True)}" '
                         f'data-search="{html.escape(title.lower(),quote=True)}"><h2>{html.escape(title)}</h2>'
                         f'<p class="warning">{html.escape(" / ".join(errors))}</p><div class="grid">{figures}</div></article>')
    def options(key):
        return ''.join(f'<option>{html.escape(v)}</option>' for v in sorted({r[key] for r in records}))
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>TUIDE comparison</title>
<style>body{font:15px system-ui;background:#eee;margin:0;color:#222}header{position:sticky;top:0;background:#fff;padding:14px;z-index:1;border-bottom:1px solid #aaa}select,input{font:inherit;padding:6px;margin:4px}article{background:white;margin:16px;padding:12px}h2{font-size:16px;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;min-width:0}figcaption{font-weight:bold;padding:8px;background:#eee}img{width:100%;height:auto;display:block}figure p{font-size:12px}.missing{padding:40px 8px;background:#eee;color:#555}.warning{color:#a22}a{display:block}article[hidden]{display:none}</style>
<header><b>Target → VLM Initial → TUIDE</b><br>저장된 평가 화면입니다. 이미지를 클릭하면 원본 크기로 열립니다. 실패·누락도 표시합니다.<br>
<select id="dataset"><option value="">모든 데이터셋</option>DATASETS</select><select id="model"><option value="">모든 모델</option>MODELS</select><input id="search" placeholder="페이지 ID 검색"><span id="count"></span></header>
CARDS
<script>const cards=[...document.querySelectorAll('article')];const dataset=document.querySelector('#dataset'),model=document.querySelector('#model'),search=document.querySelector('#search');function filter(){let n=0;for(const c of cards){c.hidden=!!((dataset.value&&c.dataset.dataset!==dataset.value)||(model.value&&c.dataset.model!==model.value)||!c.dataset.search.includes(search.value.toLowerCase()));if(!c.hidden)n++;}document.querySelector('#count').textContent=` ${n} / ${cards.length}개`;}for(const el of [dataset,model,search])el.addEventListener('input',filter);filter();</script></html>'''
    document=document.replace('DATASETS',options('dataset')).replace('MODELS',options('model')).replace('CARDS','\n'.join(cards))
    (out/'index.html').write_text(document)
    (out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(gallery=str(out/'index.html'),records=len(records),with_missing_images=sum(any(p is None for p in r['images'].values()) for r in records)),ensure_ascii=False))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/paper-color-pilot10-seed42-n2-v1')
    p.add_argument('--out');p.add_argument('--repeat',type=int,default=1)
    args=p.parse_args()
    if args.repeat<1:p.error('repeat must be positive')
    export(args.run,args.out or str(Path(args.run)/'comparison-gallery'),args.repeat)

if __name__=='__main__':main()
