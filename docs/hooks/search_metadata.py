"""Derive readable summaries and absolute language annotations from page content."""
from html import escape, unescape
from html.parser import HTMLParser
import re
from urllib.parse import urljoin


def on_env(env, **kwargs):
    env.filters["absolute_url"] = lambda url, base: urljoin(base, url)
    return env


class Paragraphs(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.paragraphs = []
        self.parts = None
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.parts = []

    def handle_data(self, data):
        if self.parts is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.parts is not None:
            value = " ".join("".join(self.parts).split())
            if len(value) >= 30:
                self.paragraphs.append(value)
            self.parts = None


def on_page_content(html, *, page, **kwargs):
    # Authored front matter always takes precedence over the fallback summary.
    if not page.meta.get("description"):
        paragraphs = Paragraphs(html).paragraphs
        if paragraphs:
            text = paragraphs[0]
            if len(text) > 170:
                # Do not split English words; Chinese can end between characters.
                text = re.sub(r"\s+\S*$", "", text[:167]) if text[:167].isascii() else text[:167]
                text += "…"
            page.meta["description"] = text
    return html


def on_post_page(output, *, page, **kwargs):
    # Material's language switcher uses root-relative URLs. Google requires
    # fully qualified hreflang URLs, including the project/version prefix.
    return re.sub(
        r'(<link\s+rel="alternate"\s+href=")([^"]+)("\s+hreflang=)',
        lambda match: match[1] + escape(urljoin(page.canonical_url, unescape(match[2])), quote=True) + match[3],
        output,
    )
