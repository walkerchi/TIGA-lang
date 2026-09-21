"""Check generated HTML links, assets and local anchors without network access."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links = set(), []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ["href", "src", "poster", *(["data"] if tag == "object" else [])]:
            if key in attrs:
                self.links.append(attrs[key])


DEFAULT_SITE_URL = "https://walkerchi.github.io/TIGA-lang/"


def check_site(directory, site_url=DEFAULT_SITE_URL):
    root = directory.resolve()
    site_url = site_url.rstrip("/") + "/"
    origin = urlsplit(site_url)
    prefix = unquote(origin.path)
    pages = {path: Page(path.read_text()) for path in root.rglob("*.html")}
    errors, count = [], 0
    for path, page in pages.items():
        base = urljoin(site_url, path.relative_to(root).as_posix())
        for link in page.links:
            parsed = urlsplit(urljoin(base, link))
            if parsed.scheme not in ("http", "https") or parsed.netloc != origin.netloc:
                continue
            target_path = unquote(parsed.path)
            if target_path == prefix.rstrip("/"):
                target_path = prefix
            if not target_path.startswith(prefix):
                # An explicit URL may intentionally reference another project;
                # relative/root-relative links must stay inside this site.
                if not urlsplit(link).netloc:
                    errors.append(f"{path.relative_to(root)}: outside site base {link}")
                continue
            target = (root / target_path[len(prefix):]).resolve()
            if not target.is_relative_to(root):
                errors.append(f"{path.relative_to(root)}: outside site directory {link}")
                continue
            if target.is_dir():
                target /= "index.html"
            count += 1
            if not target.is_file():
                errors.append(f"{path.relative_to(root)}: missing {link}")
            elif parsed.fragment and target in pages and unquote(parsed.fragment) not in pages[target].ids:
                errors.append(f"{path.relative_to(root)}: missing anchor {link}")
    return len(pages), count, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL)
    args = parser.parse_args()
    page_count, count, errors = check_site(args.site, args.site_url)
    for error in errors:
        print(error)
    print(f"{page_count} HTML files; {count} local references; {len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
