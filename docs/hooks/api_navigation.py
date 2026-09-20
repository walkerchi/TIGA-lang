"""Module-level sidebar and automatically derived per-module method indexes."""
import html
from pathlib import PurePosixPath
import re


def _is_api(page):
    return PurePosixPath(page.file.src_uri).name in {"api.md", "api.zh.md"}


def on_page_markdown(markdown, *, page, **kwargs):
    if not _is_api(page):
        return markdown
    sections = re.split(r"(?=^## )", markdown, flags=re.M)
    for i, section in enumerate(sections):
        if not section.startswith("## "):
            continue
        methods = re.findall(r"^### (.+?) \{ #([^ }]+) \}", section, re.M)
        if not methods:
            continue
        links = []
        for signature, anchor in methods:
            label = signature.split("(", 1)[0].removeprefix("class ")
            links.append(f'<li><a href="#{html.escape(anchor)}"><code>{html.escape(label)}</code></a></li>')
        title = "方法索引" if page.file.src_uri.endswith(".zh.md") else "Method index"
        index = f'<nav class="tg-api-index" aria-label="{title}"><ul>' + "".join(links) + "</ul></nav>"
        heading, body = section.split("\n", 1)
        sections[i] = heading + "\n\n" + index + "\n" + body
    return "".join(sections)


def on_page_content(html, *, page, **kwargs):
    if not _is_api(page):
        return html

    def modules_only(items):
        result = []
        for item in items:
            if item.level <= 2:
                item.children = modules_only(item.children)
                result.append(item)
        return result

    page.toc.items = modules_only(page.toc.items)
    return html
