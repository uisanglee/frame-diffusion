"""Render a deterministic detector/abstraction example for the paper figure.

This is an explanatory visualization using TUIDE's class/color contract. It is
not a saved detector prediction because the trained checkpoint is not bundled
in this workspace.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / "tuide-ui-v5.html").as_uri() + "?state=target"
VIEWPORT = (1672, 941)
COLORS = {
    "text": "#2864b4",
    "image": "#26946c",
    "control": "#df7832",
    "painted-region": "#b49ac7",
}

SELECTORS = {
    "painted-region": [".project-panel", ".task", ".feature"],
    "text": [
        ".brand", ".nav span", "h1", ".subtitle", ".panel-head > span:first-child",
        ".task > span:nth-child(2)", ".feature-name", ".feature-note",
    ],
    "control": [".new-project", ".cta", ".check", ".status"],
}


def font(size: int):
    path = ROOT / "assets" / "InterVariable.ttf"
    return ImageFont.truetype(str(path), size=size)


def collect(page):
    items = []
    for label, selectors in SELECTORS.items():
        for selector in selectors:
            boxes = page.locator(selector).evaluate_all(
                "els => els.map(e => { const r=e.getBoundingClientRect(); "
                "return [r.left,r.top,r.right,r.bottom]; })"
            )
            for box in boxes:
                x1, y1, x2, y2 = [float(value) for value in box]
                x1, x2 = max(0, x1), min(VIEWPORT[0], x2)
                y1, y2 = max(0, y1), min(VIEWPORT[1], y2)
                if x2 > x1 and y2 > y1:
                    items.append({"label": label, "box": (x1, y1, x2, y2)})
    return items


def order(items):
    labels = {name: index for index, name in enumerate(COLORS, start=1)}
    return sorted(
        items,
        key=lambda item: (
            -(item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]),
            labels[item["label"]],
            *item["box"],
        ),
    )


def render_abstract(items, size, output):
    width, height = size
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    sx, sy = width / VIEWPORT[0], height / VIEWPORT[1]
    for item in order(items):
        x1, y1, x2, y2 = item["box"]
        draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), fill=COLORS[item["label"]])
    canvas.save(output)


def render_overlay(base, items, output):
    canvas = base.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    label_font = font(18)
    for item in items:
        color = COLORS[item["label"]]
        box = item["box"]
        draw.rectangle(box, outline=color, width=4)
        text = item["label"]
        bounds = draw.textbbox((0, 0), text, font=label_font)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x, y = box[0], max(0, box[1] - th - 8)
        draw.rectangle((x, y, x + tw + 12, y + th + 8), fill=color)
        draw.text((x + 6, y + 3), text, fill="white", font=label_font)
    canvas.save(output)


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]})
        page.goto(SOURCE, wait_until="load")
        page.evaluate("document.fonts.ready")
        items = collect(page)
        browser.close()

    base = Image.open(ROOT / "tuide-ui-target-v5.png")
    render_overlay(base, items, ROOT / "tuide-ui-target-v5-detector-overlay-example.png")
    render_abstract(items, VIEWPORT, ROOT / "tuide-ui-target-v5-abstract-example.png")
    render_abstract(items, (384, 216), ROOT / "tuide-ui-target-v5-abstract-example-384.png")
    print({"elements": len(items), "counts": {key: sum(i["label"] == key for i in items) for key in COLORS}})


if __name__ == "__main__":
    main()
