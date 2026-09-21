"""Public docs must work under the Pages project prefix without native builds."""
from pathlib import Path
import tomllib

from tools.check_docs_links import DEFAULT_SITE_URL, check_site

ROOT = Path(__file__).resolve().parents[2]


def test_public_documentation_urls_are_consistent():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['urls']['Documentation'] == DEFAULT_SITE_URL
    assert DEFAULT_SITE_URL in (ROOT / 'mkdocs.yml').read_text()
    for name in ('README.md', 'README.zh.md'):
        assert DEFAULT_SITE_URL in (ROOT / name).read_text()


def test_pages_only_deploys_main_and_keeps_prs_read_only():
    source = (ROOT / '.github/workflows/docs-pages.yml').read_text()
    assert 'branches: [main, dev]' in source
    assert 'pull_request:' in source
    assert "if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'" in source
    build, deploy = source.split('\n  deploy:', 1)
    assert 'pages: write' not in build and 'id-token: write' not in build
    assert 'needs: build' in deploy
    assert 'pages: write' in deploy and 'id-token: write' in deploy
    assert 'name: github-pages' in deploy
    assert 'actions/deploy-pages@v4' in deploy
    assert 'pip install -r docs/requirements.txt' in build
    assert '-m mkdocs build --strict' in build
    assert 'tools/check_docs_links.py site' in build
    assert 'pip install -e' not in source
    assert 'pull_request_target' not in source


def test_prefix_links_assets_anchors_and_languages(tmp_path):
    (tmp_path / 'zh').mkdir()
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets/logo.svg').write_text('<svg/>')
    (tmp_path / 'index.html').write_text(
        '<a href="zh/#intro">中文</a>'
        '<a href="https://walkerchi.github.io/TIGA-lang/zh/#intro">Canonical</a>'
        '<img src="/TIGA-lang/assets/logo.svg">')
    (tmp_path / 'zh/index.html').write_text(
        '<h1 id="intro">开始</h1><a href="../">English</a>'
        '<img src="../assets/logo.svg">')
    assert check_site(tmp_path) == (2, 5, [])


def test_reject_root_links_that_drop_project_prefix(tmp_path):
    (tmp_path / 'index.html').write_text('<img src="/assets/logo.svg">')
    assert 'outside site base' in check_site(tmp_path)[2][0]


def test_check_absolute_same_site_links_not_only_relative_links(tmp_path):
    (tmp_path / 'index.html').write_text(
        '<a href="https://walkerchi.github.io/TIGA-lang/missing/">Missing</a>')
    assert 'missing' in check_site(tmp_path)[2][0]


def test_missing_anchor_and_encoded_path_escape(tmp_path):
    (tmp_path / 'index.html').write_text(
        '<a href="#absent">Missing anchor</a><a href="%2e%2e/outside">Escape</a>')
    errors = check_site(tmp_path)[2]
    assert any('missing anchor' in error for error in errors)
    assert any('outside site directory' in error for error in errors)


def test_preview_root_url_and_external_links(tmp_path):
    (tmp_path / 'index.html').write_text(
        '<a href="/#intro">Start</a><h1 id="intro">Intro</h1>'
        '<a href="https://example.org/">External</a><a href="mailto:dev@example.org">Email</a>')
    assert check_site(tmp_path, 'https://graphforge-docs.app.walkerchi.com/') == (1, 1, [])
