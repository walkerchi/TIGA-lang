"""Keep localized entry points and rendered brand assets in sync."""
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def test_mark_is_a_flat_triangular_emblem_with_geometric_g():
    root = ET.parse(ROOT/'assets/tiga-mark.svg').getroot()
    ns = '{http://www.w3.org/2000/svg}'
    paths = list(root.iter(ns+'path'))
    assert len(paths) == 3
    assert paths[0].attrib['d'] == 'M44 20 H24 L12 32 L36 51 M84 20 H104 L116 32 L92 51'
    assert paths[1].attrib['d'] == 'M42 61 L64 112 L86 61'
    assert all('Z' not in p.attrib['d'] for p in paths)
    assert 'A12 12' in paths[2].attrib['d'] and paths[2].attrib['d'].endswith('H64')
    for tag in ('circle', 'text', 'linearGradient', 'radialGradient', 'filter', 'image'):
        assert not list(root.iter(ns+tag))
    assert [p.attrib['stroke-width'] for p in paths] == ['7', '7', '6']
    group = root.find(ns+'g')
    assert group is not None
    assert group.attrib['fill'] == 'none' and group.attrib['stroke'] == '#746194'
    logo = ET.parse(ROOT/'assets/tiga-logo.svg').getroot()
    assert [p.attrib for p in logo.iter(ns+'path')] == [p.attrib for p in paths]
    assert logo.find(ns+'g').attrib == group.attrib
    assert [t.text for t in logo.iter(ns+'text')] == ['Tiga']
    assert {p.name for p in (ROOT/'assets').iterdir()} == {
        'README.md', 'tiga-mark.svg', 'tiga-mark.png', 'tiga-logo.svg', 'tiga-logo.png',
    }


def test_brand_canonical_assets_match_docs():
    for name in ('tiga-logo', 'tiga-mark'):
        for ext in ('svg', 'png'):
            source = ROOT / 'assets' / f'{name}.{ext}'
            assert source.read_bytes() == (ROOT/'docs/assets'/source.name).read_bytes()
        ET.parse(ROOT/'assets'/f'{name}.svg')
        png = (ROOT/'assets'/f'{name}.png').read_bytes()
        assert png[:8] == b'\x89PNG\r\n\x1a\n'
        assert struct.unpack('>II', png[16:24]) == ((1520, 512) if name == 'tiga-logo' else (512, 512))


def test_homepages_use_the_logo_as_the_only_page_heading():
    for name in ('index.md', 'index.zh.md'):
        text = (ROOT/'docs'/name).read_text()
        metadata, body = text.split('---', 2)[1:]
        assert 'title: Tiga' in metadata
        headings = re.findall(r'^# (.+)$', body, re.M)
        assert len(headings) == 1
        assert re.fullmatch(
            r'!\[Tiga\]\(assets/tiga-logo\.svg\)\{[^}]*\.tiga-brand-logo[^}]*\} \{ #tiga \}',
            headings[0],
        )
        assert '<div class="tiga-home-hero" markdown>' in body
        assert body.count('assets/tiga-logo.svg') == 1


def test_readmes_share_a_working_torch_example():
    import torch
    import pytest
    english = (ROOT/'README.md').read_text()
    chinese = (ROOT/'README.zh.md').read_text()
    assert 'README.zh.md' in english and 'README.md' in chinese
    blocks = [re.findall(r'```python\n(.*?)```', text, re.S)[0]
              for text in (english, chinese)]
    assert blocks[0] == blocks[1]
    assert 'torch.device("cuda")' in blocks[0]
    if not torch.cuda.is_available():
        pytest.skip("README GPU example requires CUDA; exercised by GPU release gate")
    namespace = {}
    exec(compile(blocks[0], 'README example', 'exec'), namespace)
    assert isinstance(namespace['out'], torch.Tensor)
    assert namespace['out'].tolist() == [11., 8., 17.]
    assert namespace['dx'].tolist() == [7., 10., 3.]
    assert namespace['dw'].tolist() == [1., 3., 2., 1., 2.]


def test_homepages_share_a_working_cuda_example():
    import pytest
    import torch
    blocks = [re.findall(r'```python\n(.*?)```', (ROOT/'docs'/name).read_text(), re.S)[0]
              for name in ('index.md', 'index.zh.md')]
    assert blocks[0] == blocks[1]
    assert 'torch.device("cuda")' in blocks[0]
    if not torch.cuda.is_available():
        pytest.skip('Homepage example requires an NVIDIA GPU')
    namespace = {}
    exec(compile(blocks[0], 'Homepage CUDA example', 'exec'), namespace)
    for name in ('x', 'weight', 'out', 'dx', 'dw'):
        assert isinstance(namespace[name], torch.Tensor)
        assert namespace[name].is_cuda
    assert namespace['out'].tolist() == [11., 8., 17.]
    assert namespace['dx'].tolist() == [7., 10., 3.]
    assert namespace['dw'].tolist() == [1., 3., 2., 1., 2.]


def test_readme_local_file_links_exist():
    for name in ('README.md', 'README.zh.md', 'assets/README.md'):
        path = ROOT/name
        text = path.read_text()
        links = re.findall(r'(?:src|href)="([^"]+)"', text)
        links += re.findall(r'\]\(([^)]+)\)', text)
        for link in links:
            if ':' in link or link.startswith('#'):
                continue
            assert (path.parent/link.split('#')[0]).is_file(), (name, link)

    project = (ROOT/'pyproject.toml').read_text()
    assert '"README.zh.md"' in project and '"assets/**"' in project


def test_readme_images_use_matching_repository_relative_paths():
    expected = ['assets/tiga-logo.png', 'docs/assets/compiler-performance-overview.svg']
    for name in ('README.md', 'README.zh.md'):
        images = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', (ROOT/name).read_text())
        assert images == expected
        assert all((ROOT/path).is_file() for path in images)


def test_uv_lock_is_local_only_and_not_required_by_ci():
    assert '/uv.lock' in (ROOT/'.gitignore').read_text().splitlines()
    for path in (ROOT/'.github/workflows').glob('*.yml'):
        text = path.read_text()
        assert 'uv.lock' not in text and 'uv sync --frozen' not in text


def test_report_repository_is_consistent_across_public_entry_points():
    import tomllib
    report = 'https://github.com/walkerchi/tiga-lang-paper'
    project = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
    assert project['urls']['Technical report'] == report
    for name in ('README.md', 'README.zh.md', 'docs/index.md', 'docs/index.zh.md',
                 'docs/support.md', 'docs/support.zh.md'):
        assert report in (ROOT/name).read_text(), name
    for name in ('docs/support.md', 'docs/support.zh.md'):
        assert 'arXiv ID' in (ROOT/name).read_text()
