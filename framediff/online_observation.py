"""Collect the unchanged full-DOM observation in a single browser round trip."""
from .visual import VISUAL_GEOMETRY_JS, VISUAL_CLASS_JS


def observe(browser, tree, viewport):
    data = browser.page.evaluate('''() => {
      // CSS is unchanged during this evaluation; ancestors share computed styles.
      const styles=new Map();
      const getComputedStyle=e=>{if(!styles.has(e))styles.set(e,window.getComputedStyle(e));return styles.get(e);};
      const measure=(''' + VISUAL_GEOMETRY_JS + '''),classify=(''' + VISUAL_CLASS_JS + ''');
      const boxes={},geometry={},items=[],memo=new Map(),rects=new Map(),clips=new Map();
      const get=e=>{if(!memo.has(e))memo.set(e,measure(e));return memo.get(e);};
      const rect=e=>{if(!rects.has(e))rects.set(e,e.getBoundingClientRect());return rects.get(e);};
      // Compute each ancestor's accumulated clip once per observation. Rebuild
      // on every call so CSS edits, scrolling, and reflow remain observable.
      const clip=e=>{
        const chain=[];let p=e;
        while(p&&!clips.has(p)){chain.push(p);p=p.parentElement;}
        let c=p?clips.get(p):[0,0,innerWidth,innerHeight];
        for(let i=chain.length-1;i>=0;i--){
          const n=chain[i],s=getComputedStyle(n),q=rect(n);let [l,t,r,b]=c;
          if(['hidden','clip','scroll','auto'].includes(s.overflowX)){
            l=Math.max(l,q.left+n.clientLeft);r=Math.min(r,q.left+n.clientLeft+n.clientWidth);}
          if(['hidden','clip','scroll','auto'].includes(s.overflowY)){
            t=Math.max(t,q.top+n.clientTop);b=Math.min(b,q.top+n.clientTop+n.clientHeight);}
          c=[l,t,r,b];clips.set(n,c);
        }
        return c;
      };
      for(const e of document.querySelectorAll('[data-fd-id]')){
        const id=e.dataset.fdId,q=rect(e);
        boxes[id]=[q.x,q.y,q.width,q.height];geometry[id]=get(e);
      }
      for(const e of document.querySelectorAll('body,body *')){
        const g=get(e);if(!g.visible)continue;
        if(e.parentElement?.closest('button,input,textarea,select,img,svg,canvas,video'))continue;
        const [l,t,r,b]=clip(e.parentElement);
        const q=rect(e),box=[Math.max(l,q.left),Math.max(t,q.top),Math.min(r,q.right),Math.min(b,q.bottom)];
        const label=classify(e,g);
        if(label&&box[2]>box[0]&&box[3]>box[1])items.push({label,box,owner_id:e.getAttribute('data-fd-id')});
      }
      return {boxes,geometry,items};
    }''')
    ids={n['id'] for n in tree['nodes'][1:]}
    if ids != set(data['boxes']):
        raise ValueError('DOM identities changed during feedback; refusing to rematch silently')
    tree['full_visual_elements']=data['items']
    for node in tree['nodes']:
        g=data['geometry'].get(node['id'],{'visible':False,'text_boxes':[]})
        node['visual_visible']=g['visible'];node['visual_boxes']=g['text_boxes']
    data['boxes'][tree['nodes'][0]['id']]=[0,0,*viewport]
    return data['boxes'],data['items']
