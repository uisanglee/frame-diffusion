"""Prompts and text extraction used by Design2Code's visual Self-Revision.

The protocol is reproduced from NoviScl/Design2Code, ``prompting/gpt4v.py``
and ``prompting/gpt4v_utils.py``.  The upstream revision prompt accidentally
joins an already-joined string character by character; this module preserves
the intended newline-separated text elements instead of that implementation
bug.
"""
import re

from bs4 import BeautifulSoup


SOURCE_URL = 'https://github.com/NoviScl/Design2Code/blob/main/Design2Code/prompting/gpt4v.py'


def extract_text_elements(html):
    """Match the upstream text-element extraction semantics."""
    html = re.sub(r'<!.*?>', '', html, flags=re.DOTALL).strip()
    soup = BeautifulSoup(html, 'html.parser')
    for style in soup.find_all('style'):
        style.decompose()
    return [
        ' '.join(element.strip().replace('\n', ' ').split())
        for element in soup.find_all(string=True)
        if element.parent.name != 'script'
        and element.strip()
        and element.strip() != 'html'
    ]


def text_augmented_prompt(texts):
    texts = '\n'.join(texts)
    return (
        'You are an expert web developer who specializes in HTML and CSS.\n'
        'A user will provide you with a screenshot of a webpage, along with all texts that they want to put on the webpage.\n'
        'The text elements are:\n' + texts + '\n'
        'You should generate the correct layout structure for the webpage, and put the texts in the correct places so that the resultant webpage will look the same as the given one.\n'
        'You need to return a single html file that uses HTML and CSS to reproduce the given website.\n'
        'Include all CSS code in the HTML file itself.\n'
        'If it involves any images, use "rick.jpg" as the placeholder.\n'
        'Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well.\n'
        'Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n'
        'Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n'
        'Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n'
    )


def revision_prompt(current_html, texts):
    texts = '\n'.join(texts)
    return (
        'You are an expert web developer who specializes in HTML and CSS.\n'
        'I have an HTML file for implementing a webpage but it has some missing or wrong elements that are different from the original webpage. The current implementation I have is:\n'
        + current_html + '\n\n'
        'I will provide the reference webpage that I want to build as well as the rendered webpage of the current implementation.\n'
        'I also provide you all the texts that I want to include in the webpage here:\n'
        + texts + '\n\n'
        'Please compare the two webpages and refer to the provided text elements to be included, and revise the original HTML implementation to make it look exactly like the reference webpage. Make sure the code is syntactically correct and can render into a well-formed webpage. You can use "rick.jpg" as the placeholder image file.\n'
        'Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n'
        'Respond directly with the content of the new revised and improved HTML file without any extra explanations:\n'
    )
