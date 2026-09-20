"""The API sidebar is a module index, not a list of full signatures."""
from pathlib import Path
import re
import runpy
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = runpy.run_path(str(ROOT / "docs/hooks/api_navigation.py"))


@pytest.mark.parametrize("suffix", ["", ".zh"])
def test_every_method_is_indexed_without_changing_anchors(suffix):
    path = f"api{suffix}.md"
    source = (ROOT / "docs" / path).read_text()
    page = SimpleNamespace(file=SimpleNamespace(src_uri=path))
    rendered = HOOK["on_page_markdown"](source, page=page)
    headings = re.findall(r"^### .*\{ #([^ }]+) \}", source, re.M)
    assert len(headings) == 120
    for anchor in headings:
        assert rendered.count(f'href="#{anchor}"') == 1
    assert re.findall(r"^### .*\{ #([^ }]+) \}", rendered, re.M) == headings
    assert 'class="tg-api-index"' in rendered


def test_sidebar_prunes_methods_but_keeps_modules_and_other_pages():
    method = SimpleNamespace(level=3, children=[])
    module = SimpleNamespace(level=2, children=[method])
    title = SimpleNamespace(level=1, children=[module])
    page = SimpleNamespace(file=SimpleNamespace(src_uri="api.md"), toc=SimpleNamespace(items=[title]))
    assert HOOK["on_page_content"]("html", page=page) == "html"
    assert page.toc.items == [title] and title.children == [module]
    assert module.children == []
    page.file.src_uri = "examples/message-passing.md"
    assert HOOK["on_page_markdown"]("# Unchanged", page=page) == "# Unchanged"
