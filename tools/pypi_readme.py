"""Resolve README image URLs for PyPI without changing the repository README.

Only used by the build backend. Public image URLs require a public repository;
no credentials or temporary signed URLs are embedded in package metadata.
"""
from pathlib import Path
import re
from urllib.parse import urlsplit


def package_readme(source: str, repository: str) -> str:
    """Expand local HTML image sources, leaving text, links and remote images intact."""
    base = repository.removesuffix(".git").rstrip("/") + "/raw/refs/heads/main/"

    def image(match: re.Match[str]) -> str:
        url = match["url"]
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or url.startswith(("/", "#")):
            return match[0]
        return match["prefix"] + match["quote"] + base + url.removeprefix("./") + match["quote"]

    return re.sub(
        r'''(?P<prefix><img\b[^>]*?\bsrc=)(?P<quote>["'])(?P<url>[^"']+)(?P=quote)''',
        image, source,
    )


def dynamic_metadata(settings, project):
    if settings:
        raise ValueError("The README metadata provider accepts no settings")
    source = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    return {"readme": {
        "content-type": "text/markdown",
        "text": package_readme(source, project["urls"]["Repository"]),
    }}
