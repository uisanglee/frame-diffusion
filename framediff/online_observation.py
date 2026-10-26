"""Collect the unchanged full-DOM observation in a single browser round trip."""
from .visual import VISUAL_GEOMETRY_JS, VISUAL_CLASS_JS


def observe(browser, tree, viewport):
    data = browser.page.evaluate('''() => {
      // CSS is unchanged during this evaluation; ancestors share computed styles.
      const styles=new Map();
      const getComputedStyle=e=>{if(!styles.has(e))styles.set(e,window.getComputedStyle(e));return styles.get(e);};
      const measure=(''' + VISUAL_GEOMETRY_JS + '''),classify=(''' + VISUAL_CLASS_JS + ''');
      const boxes={},geometry={},items=[],memo=new Map();
      const get=e=>{if(!memo.has(e))memo.set(e,measure(e));return memo.get(e);};
      for(const e of document.querySelectorAll('[data-fd-id]')){
        const id=e.dataset.fdId,q=e.getBoundingClientRect();
        boxes[id]=[q.x,q.y,q.width,q.height];geometry[id]=get(e);
      }
      for(const e of document.querySelectorAll('body,body *')){
        const g=get(e);if(!g.visible)continue;
        if(e.parentElement?.closest('button,input,textarea,select,img,svg,canvas,video'))continue;
        let l=0,t=0,r=innerWidth,b=innerHeight;
        for(let p=e.parentElement;p;p=p.parentElement){
          const s=getComputedStyle(p),q=p.getBoundingClientRect();
          if(['hidden','clip','scroll','auto'].includes(s.overflowX)){
            l=Math.max(l,q.left+p.clientLeft);r=Math.min(r,q.left+p.clientLeft+p.clientWidth);}
          if(['hidden','clip','scroll','auto'].includes(s.overflowY)){
            t=Math.max(t,q.top+p.clientTop);b=Math.min(b,q.top+p.clientTop+p.clientHeight);}
        }
        const q=e.getBoundingClientRect(),box=[Math.max(l,q.left),Math.max(t,q.top),Math.min(r,q.right),Math.min(b,q.bottom)];
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
