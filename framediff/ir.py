"""The executable subset is deliberately explicit, not an arbitrary CSS emulator."""
from __future__ import annotations
import copy
import html
import json
from pathlib import Path

FIELDS = ("width", "height", "dx", "dy", "flow", "gap", "padding", "columns", "align", "order")
# Encoded values: width / 256 of parent content width; height * 4px;
# offsets (value - 32) * 4px, except children of absolute containers where
# offsets are fractions /256 of parent content width/height.
LIMITS = {"width": (1, 256), "height": (1, 256), "dx": (0, 256), "dy": (0, 256),
          "flow": (0, 3), "gap": (0, 16), "padding": (0, 16), "columns": (1, 6),
          "align": (0, 2), "order": (0, 31)}
DEFAULTS = dict(width=256, height=24, dx=32, dy=32, flow=1, gap=2,
                padding=0, columns=3, align=0, order=0)
ROLES = ("page", "container", "header", "nav", "sidebar", "main", "section", "card",
         "title", "text", "image", "button", "input", "footer", "list", "icon", "unknown")

def node(node_id, parent, role="container", **props):
    return {"id": node_id, "parent": parent, "role": role, "name": node_id,
            "props": {**DEFAULTS, **props}}

def validate(tree):
    if tree.get("version") != 1:
        raise ValueError("LayoutIR version must be 1")
    nodes = tree["nodes"]
    if not 1 <= len(nodes) <= 256:
        raise ValueError("LayoutIR requires 1..256 nodes")
    ids = [n["id"] for n in nodes]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Node IDs must be unique nonempty strings")
    if nodes[0]["parent"] is not None or any(n["parent"] is None for n in nodes[1:]):
        raise ValueError("Exactly one root, at index zero, is required")
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        for field, (lo, hi) in LIMITS.items():
            v = n["props"].get(field)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"Invalid {n['id']}.{field}: {v}")
        seen = {n["id"]}
        parent = n["parent"]
        while parent is not None:
            if parent not in by_id or parent in seen:
                raise ValueError("Missing parent or cyclic tree")
            seen.add(parent)
            parent = by_id[parent]["parent"]
    return tree

def children(tree):
    out = {n["id"]: [] for n in tree["nodes"]}
    for n in tree["nodes"][1:]:
        out[n["parent"]].append(n)
    for group in out.values():
        group.sort(key=lambda n: n["props"]["order"])
    return out

def editable(tree):
    """Legal (node index, property) pairs, excluding irrelevant leaf fields."""
    groups = children(tree)
    for i, n in enumerate(tree["nodes"]):
        for f in FIELDS:
            if i == 0 and f in ("width", "height", "dx", "dy", "order"):
                continue
            if f in ("flow", "gap", "padding", "columns", "align") and not groups[n["id"]]:
                continue
            yield i, f

def apply_edit(tree, edit):
    i, field, value = edit
    if (i, field) not in set(editable(tree)):
        raise ValueError("Illegal edit location")
    lo, hi = LIMITS[field]
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError("Illegal edit value")
    result = copy.deepcopy(tree)
    result["nodes"][i]["props"][field] = value
    return validate(result)

def execute(tree, viewport=(1024, 768)):
    """CPU frame executor. Mirrors compile_html's limited CSS contract."""
    validate(tree)
    groups, boxes = children(tree), {}
    def visit(n, x, y, w, h):
        boxes[n["id"]] = [x, y, w, h]
        p = n["props"]
        pad, gap = p["padding"] * 4, p["gap"] * 4
        inner_w, inner_h = max(0, w - 2*pad), max(0, h - 2*pad)
        kids = groups[n["id"]]
        flow, cols = p["flow"], p["columns"]
        cell_w = max(0, (inner_w - gap*(cols-1))/cols) if flow == 2 else inner_w
        cursor_x = cursor_y = row_h = 0.0
        for j, c in enumerate(kids):
            q = c["props"]
            cw, ch = cell_w*q["width"]/256, q["height"]*4
            if flow == 0:
                cx, cy = cursor_x, (inner_h-ch)*p["align"]/2
                cursor_x += cw + gap
            elif flow == 1:
                cx, cy = (inner_w-cw)*p["align"]/2, cursor_y
                cursor_y += ch + gap
            elif flow == 2:
                if j and j % cols == 0:
                    cursor_y += row_h + gap
                    row_h = 0.0
                cx, cy = (j % cols)*(cell_w+gap) + (cell_w-cw)*p["align"]/2, cursor_y
                row_h = max(row_h, ch)
            else:
                cx, cy = inner_w*q['dx']/256, inner_h*q['dy']/256
            ox, oy = ((q['dx']-32)*4, (q['dy']-32)*4) if flow != 3 else (0,0)
            visit(c, x+pad+cx+ox, y+pad+cy+oy, cw, ch)
    visit(tree["nodes"][0], 0.0, 0.0, float(viewport[0]), float(viewport[1]))
    return boxes

def compile_html(tree, viewport=(1024,768)):
    validate(tree)
    groups = children(tree)
    def emit(n, root=False, parent=None):
        p = n["props"]
        align = ("start", "center", "end")[p["align"]]
        if p['flow'] == 3:
            layout = 'display:block;'
        elif p["flow"] == 2:
            layout = f"display:grid;grid-template-columns:repeat({p['columns']},minmax(0,1fr));align-content:start;align-items:start;justify-items:{align};"
        else:
            layout = f"display:flex;flex-direction:{('row','column')[p['flow']]};align-items:{('flex-start','center','flex-end')[p['align']]};"
        size = f"width:{viewport[0]}px;height:{viewport[1]}px;" if root else f"width:{p['width']/256*100}%;height:{p['height']*4}px;"
        style = size + layout + f"padding:{p['padding']*4}px;gap:{p['gap']*4}px;order:{p['order']};"
        if parent is not None and parent['props']['flow'] == 3:
            pad = parent['props']['padding']*4
            style += (f"position:absolute;width:calc((100% - {2*pad}px) * {p['width']/256});"
                      f"left:calc({pad}px + (100% - {2*pad}px) * {p['dx']/256});"
                      f"top:calc({pad}px + (100% - {2*pad}px) * {p['dy']/256});")
        elif not root:
            style += f"transform:translate({(p['dx']-32)*4}px,{(p['dy']-32)*4}px);"
        content = "".join(emit(c,parent=n) for c in groups[n["id"]])
        label = html.escape(str(n.get("name", n["id"])), quote=True)
        return f'<div data-frame-id="{html.escape(n["id"],quote=True)}" data-name="{label}" style="{style}">{content}</div>'
    return '<!doctype html><meta charset="utf-8"><style>html,body{margin:0;padding:0}*{box-sizing:border-box}div{position:relative;min-width:0;min-height:0;flex-shrink:0;outline:1px solid #7393bf;background:#7393bf0a}div:after{content:attr(data-name);position:absolute;top:1px;left:2px;font:10px sans-serif;color:#355273;pointer-events:none}</style>'+emit(tree["nodes"][0],True)

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
