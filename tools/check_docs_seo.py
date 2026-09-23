"""Validate built search metadata and sitemap coverage without network access."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
import xml.etree.ElementTree as ET


# Persistent Search Console ownership proof, not a rendered documentation page.
# Keep the file published even after verification succeeds.
VERIFICATION_FILES = {
    'google57fbec322702f87c.html':
        'google-site-verification: google57fbec322702f87c.html',
}


class Metadata(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.title = ""
        self.meta = {}
        self.canonicals = []
        self.alternates = {}
        self.json_ld = []
        self.capture = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            self.meta[attrs.get('name', attrs.get('property'))] = attrs.get('content', '')
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonicals.append(attrs['href'])
        if tag == 'link' and attrs.get('rel') == 'alternate' and 'hreflang' in attrs:
            self.alternates[attrs['hreflang']] = attrs['href']
        if tag == 'title':
            self.capture = 'title'
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self.capture = 'json'
            self.json_ld.append('')

    def handle_data(self, data):
        if self.capture == 'title':
            self.title += data
        if self.capture == 'json':
            self.json_ld[-1] += data

    def handle_endtag(self, tag):
        if tag in ('title', 'script'):
            self.capture = None


def check_site(root):
    root = Path(root)
    errors = []
    for name, expected in VERIFICATION_FILES.items():
        proof = root / name
        if not proof.is_file() or proof.read_text().strip() != expected:
            errors.append(f'{name}: missing or altered site verification file')
    # Archived standalone chart exports are evidence assets, not MkDocs pages.
    pages = {}
    for path in root.rglob('*.html'):
        relative = path.relative_to(root).as_posix()
        if relative.startswith('assets/'):
            continue
        if path.name in VERIFICATION_FILES:
            # The language plugin can also copy static files into locale roots.
            if path.read_text().strip() != VERIFICATION_FILES[path.name]:
                errors.append(f'{relative}: altered site verification file')
            continue
        pages[relative] = Metadata(path.read_text())
    base = pages['index.html'].canonicals[0]
    canonical_pages = {}
    for path, page in pages.items():
        if path == '404.html':
            if 'noindex' not in page.meta.get('robots', ''):
                errors.append('404.html: missing noindex')
            continue
        expected = urljoin(base, path.removesuffix('index.html'))
        canonical_pages[expected] = page
        if page.canonicals != [expected]:
            errors.append(f'{path}: incorrect canonical {page.canonicals}')
        if not page.title.strip() or len(page.meta.get('description', '')) < 30:
            errors.append(f'{path}: missing descriptive title/summary')
        if 'noindex' in page.meta.get('robots', ''):
            errors.append(f'{path}: unexpectedly blocked from indexing')
        if page.meta.get('og:url') != expected:
            errors.append(f'{path}: incorrect Open Graph URL')
        image = page.meta.get('og:image', '')
        relative_image = unquote(urlsplit(image).path).removeprefix(unquote(urlsplit(base).path))
        if not image.startswith(base) or not (root / relative_image).is_file():
            errors.append(f'{path}: missing social image {image}')
        if set(page.alternates) != {'en', 'zh'}:
            errors.append(f'{path}: missing language alternate')
    for canonical, page in canonical_pages.items():
        for language, target in page.alternates.items():
            if target not in canonical_pages:
                errors.append(f'{canonical}: invalid {language} alternate {target}')
            elif canonical_pages[target].alternates != page.alternates:
                errors.append(f'{canonical}: non-reciprocal alternates')
    sitemap = ET.parse(root / 'sitemap.xml')
    urls = [node.text for node in sitemap.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
    if len(urls) != len(set(urls)) or set(urls) != set(canonical_pages):
        errors.append('sitemap.xml: URLs do not match indexable HTML pages')
    for path in ('index.html', 'zh/index.html'):
        schemas = [json.loads(raw) for raw in pages[path].json_ld]
        if len(schemas) != 1 or schemas[0].get('@type') != 'SoftwareSourceCode':
            errors.append(f'{path}: missing software source metadata')
    return len(canonical_pages), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', type=Path)
    args = parser.parse_args()
    count, errors = check_site(args.site)
    for error in errors:
        print(error)
    print(f'{count} indexable pages; {len(errors)} SEO errors')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
