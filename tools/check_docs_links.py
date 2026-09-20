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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    args = parser.parse_args()
    root = args.site.resolve()
    pages = {path: Page(path.read_text()) for path in root.rglob("*.html")}
    errors, count = [], 0
    for path, page in pages.items():
        base = "/" + str(path.relative_to(root))
        for link in page.links:
            parsed = urlsplit(urljoin(base, link))
            if parsed.scheme not in ("", "http", "https") or parsed.netloc not in ("", "graphforge-docs.app.walkerchi.com"):
                continue
            target = root / unquote(parsed.path).lstrip("/")
            if target.is_dir():
                target /= "index.html"
            count += 1
            if not target.is_file():
                errors.append(f"{path.relative_to(root)}: missing {link}")
            elif parsed.fragment and target in pages and unquote(parsed.fragment) not in pages[target].ids:
                errors.append(f"{path.relative_to(root)}: missing anchor {link}")
    for error in errors:
        print(error)
    print(f"{len(pages)} HTML files; {count} local references; {len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
