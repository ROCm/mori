#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Check local href targets and fragments in a rendered Sphinx site, offline.

Scope: ``<a href>`` only. ``<link rel="next|prev">``, ``<img src>``, ``<script
src>`` and stylesheet ``<link href>`` are not followed -- the theme generates
those from the toctree and the static tree, so they do not rot independently of
the anchors that do. Do not read a clean run as "every asset resolves".

``--markdown`` also checks the links a Markdown file such as the README makes
into the published site (``--site-url``), against the same rendered tree. They
are absolute URLs, so the page check above skips them, and nothing else would
notice a renamed page leaving them dangling.
"""

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE_URL = "https://rocm.github.io/mori/"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.anchors = set()
        self.links = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.anchors.add(attrs["id"])
        if tag == "a":
            if "name" in attrs:
                self.anchors.add(attrs["name"])
            if "href" in attrs:
                self.links.append(attrs["href"])


def site_links(text, site_url):
    """Targets of ``[text](url)``, ``<url>`` and ``href="url"`` links into the site."""
    site = re.escape(site_url)
    pattern = rf"\]\(<?({site}[^)\s>]*)>?\)|<({site}[^>\s]*)>|href=\"({site}[^\"]*)\""
    return [next(g for g in m.groups() if g) for m in re.finditer(pattern, text)]


def problem(root, pages, target, fragment):
    """Why a link to ``target#fragment`` does not resolve, or None if it does."""
    if target.is_dir():
        target /= "index.html"
    if not target.is_relative_to(root) or not target.is_file():
        return "missing local target"
    if fragment and target in pages and unquote(fragment) not in pages[target].anchors:
        return "missing anchor"
    return None


def check(root, markdown=(), site_url=SITE_URL):
    root = root.resolve()
    if not (root / "index.html").is_file():
        raise ValueError(f"Missing documentation root: {root / 'index.html'}")
    pages = {path.resolve(): Page(path) for path in root.rglob("*.html")}
    errors = []
    checked = 0
    for path, page in pages.items():
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            checked += 1
            if not url.path:
                target = path
            elif url.path.startswith("/"):
                target = (root / unquote(url.path).lstrip("/")).resolve()
            else:
                target = (path.parent / unquote(url.path)).resolve()
            why = problem(root, pages, target, url.fragment)
            if why:
                errors.append(f"{path.relative_to(root)}: {why} {href}")
    site_url = site_url.rstrip("/") + "/"
    linked = 0
    for doc in markdown:
        for href in site_links(doc.read_text(encoding="utf-8"), site_url):
            linked += 1
            url = urlsplit(href[len(site_url) :])
            target = (root / unquote(url.path)).resolve()
            why = problem(root, pages, target, url.fragment)
            if why:
                errors.append(f"{doc}: {why} {href}")
    for error in sorted(set(errors)):
        print(error)
    print(
        f"Checked {checked} local links in {len(pages)} HTML pages and {linked} site "
        f"links in {len(markdown)} Markdown files; {len(set(errors))} errors"
    )
    return bool(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_dir", type=Path)
    parser.add_argument(
        "--markdown",
        type=Path,
        action="append",
        default=[],
        help="Markdown file whose links into the site to check (repeatable)",
    )
    parser.add_argument("--site-url", default=SITE_URL)
    args = parser.parse_args()
    raise SystemExit(check(args.html_dir, args.markdown, args.site_url))
