"""SEO metadata stays tied to visible content, with no runtime dependencies."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "search_metadata", ROOT / "docs/hooks/search_metadata.py")
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


def test_summary_uses_visible_paragraph_text_not_markup():
    page = SimpleNamespace(meta={})
    content = '<h1>Title</h1><p><img src="logo.png"></p><p>Use <code>torch.Tensor</code> inputs &amp; outputs for graph aggregation.</p>'
    assert metadata.on_page_content(content, page=page) == content
    assert page.meta['description'] == 'Use torch.Tensor inputs & outputs for graph aggregation.'


def test_explicit_descriptions_are_not_replaced():
    page = SimpleNamespace(meta={'description': 'A deliberately written summary.'})
    metadata.on_page_content('<p>' + 'An unrelated sentence. ' * 20 + '</p>', page=page)
    assert page.meta['description'] == 'A deliberately written summary.'


def test_summaries_are_short_and_localized():
    for text in ('Graph aggregation with PyTorch tensors. ' * 20,
                 '使用普通 Tensor 编写可微的图计算程序。' * 20):
        page = SimpleNamespace(meta={})
        metadata.on_page_content(f'<p>{text}</p>', page=page)
        assert 30 <= len(page.meta['description']) <= 170
        assert page.meta['description'].endswith('…')


def test_hreflang_uses_absolute_urls_and_preserves_prefix():
    page = SimpleNamespace(canonical_url='https://example.com/project/zh/guide/')
    source = '<link rel="alternate" href="/project/guide/" hreflang="en"><a href="../">Back</a>'
    result = metadata.on_post_page(source, page=page)
    assert 'href="https://example.com/project/guide/" hreflang="en"' in result
    assert '<a href="../">Back</a>' in result


def test_homepages_have_descriptive_metadata_without_duplicate_logo():
    for name in ('index.md', 'index.zh.md'):
        source = (ROOT / 'docs' / name).read_text()
        assert 'title: Tiga — ' in source
        assert 'description:' in source
        assert source.count('assets/tiga-logo.svg') == 1
        assert 'pytorch-sparse-message-passing.md' in source


def test_documentation_template_is_in_the_source_distribution():
    import tomllib
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert 'overrides/**' in project['tool']['scikit-build']['sdist']['include']


def test_both_guides_use_one_checked_example_source():
    for name in ('pytorch-sparse-message-passing.md', 'pytorch-sparse-message-passing.zh.md'):
        source = (ROOT / 'docs' / name).read_text()
        assert '--8<-- "examples/pytorch_sparse_message_passing.py"' in source
        assert 'https://github.com/walkerchi/TIGA-lang/blob/main/examples/pytorch_sparse_message_passing.py' in source


def test_sparse_message_passing_example_on_cpu():
    import pytest
    pytest.importorskip('torch')
    pytest.importorskip('tiga')
    import runpy
    namespace = runpy.run_path(str(ROOT / 'examples/pytorch_sparse_message_passing.py'))
    namespace['run']('cpu')


def test_seo_checker_detects_broken_metadata_and_sitemap(tmp_path):
    from tools.check_docs_seo import check_site, VERIFICATION_FILES
    base = 'https://example.com/project/'
    (tmp_path / 'zh').mkdir()
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets/mark.png').write_bytes(b'example')
    for path, url in [('index.html', base), ('zh/index.html', base + 'zh/')]:
        (tmp_path / path).write_text(
            '<title>Graph message passing with PyTorch</title>'
            '<meta name="description" content="A practical guide to differentiable graph programs.">'
            f'<link rel="canonical" href="{url}">'
            f'<meta property="og:url" content="{url}">'
            f'<meta property="og:image" content="{base}assets/mark.png">'
            f'<link rel="alternate" hreflang="en" href="{base}">'
            f'<link rel="alternate" hreflang="zh" href="{base}zh/">'
            '<script type="application/ld+json">{"@type":"SoftwareSourceCode"}</script>')
    (tmp_path / '404.html').write_text('<meta name="robots" content="noindex">')
    for name, expected in VERIFICATION_FILES.items():
        assert (ROOT / 'docs' / name).read_text().strip() == expected
        (tmp_path / name).write_text(expected + '\n')
        (tmp_path / 'zh' / name).write_text(expected + '\n')
    (tmp_path / 'sitemap.xml').write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f'<url><loc>{base}</loc></url><url><loc>{base}zh/</loc></url></urlset>')
    assert check_site(tmp_path) == (2, [])
    for name, expected in VERIFICATION_FILES.items():
        proof = tmp_path / name
        proof.unlink()
        assert any('missing or altered site verification' in error
                   for error in check_site(tmp_path)[1])
        proof.write_text('<html>' + expected + '</html>')
        assert any('site verification' in error for error in check_site(tmp_path)[1])
        proof.write_text(expected + '\n')
    assert check_site(tmp_path) == (2, [])
    page = tmp_path / 'zh/index.html'
    page.write_text(page.read_text().replace(f'href="{base}zh/"', 'href="/zh/"'))
    assert any('canonical' in error for error in check_site(tmp_path)[1])
    assert any('alternate' in error for error in check_site(tmp_path)[1])
