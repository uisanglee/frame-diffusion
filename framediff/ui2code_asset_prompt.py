"""Asset-only UI reconstruction rules shared by UI2Code draft and polishing."""
import re

CONTRACT = 'ui2coden-supplied-file-assets-v1'
WEBUI_CONTRACT = 'webui-five-color-images-v1'


def image_asset_instruction(html='', image_contract=None):
    """Use only protocol metadata/current HTML; never inspect target DOM geometry."""
    if image_contract is None:
        image_contract = WEBUI_CONTRACT if (WEBUI_CONTRACT in html or
            re.search(r'\brick[1-5]\.jpg\b', html)) else 'design2code-rick-v1'
    common = (
        'Focus on matching the UI layout, element positions, sizes, and spacing. '
        'During refinement, preserve the existing text, supplied assets, and application behavior. '
        'React, JSX, Babel, and Tailwind are allowed when needed to render the application. '
        'Use supplied image assets by filename; preserve existing registered local asset URLs. '
        'Do not generate new images, artwork, logos, SVG illustrations, canvas drawings, '
        'or base64/data-URI image payloads. Preserve existing UI icons. '
        'Do not add image, icon, or font packs, or fetch additional image assets. '
        'Do not copy library contents or asset binaries into the HTML or CSS. '
        'Reference required React, Babel, or Tailwind dependencies rather than embedding their source.\n'
    )
    if image_contract == WEBUI_CONTRACT:
        assets = ('For WebUI, the supplied image placeholders are red=rick1.jpg, green=rick2.jpg, '
                  'yellow=rick3.jpg, blue=rick4.jpg, and black=rick5.jpg. Use the corresponding '
                  'filename for each colored image region, including CSS background images. '
                  'Match its width and height with CSS; do not replace it with generated artwork.\n')
    elif image_contract == 'design2code-rick-v1':
        assets = ('For Design2Code and Design2Code-Hard, use rick.jpg for all placeholder image regions, '
                  'including blue rectangles and CSS background images. Adjust the image dimensions '
                  'with CSS; do not generate a replacement image.\n')
    else:
        assets = ('Use only image files already supplied with the current page; do not invent '
                  'new image filenames or substitute a different dataset\'s placeholders.\n')
    return common + assets
