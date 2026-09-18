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
"""Fixtures for tools/check_docs_links.py.

The docs job gates publication on this checker, so a checker that silently
passes everything would be worse than no checker at all: the build would stay
green while the site rotted. These pin that it fails on the two things it
claims to catch, and that it does not invent failures on a clean site.

Pure CPU, no Sphinx and no network -- the fixtures are hand-written HTML.
"""

import importlib.util
from pathlib import Path

import pytest

_CHECKER = Path(__file__).resolve().parents[2] / "tools" / "check_docs_links.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_docs_links", _CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check


def _site(root, pages):
    for name, body in pages.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return root


def test_clean_site_passes(check, tmp_path):
    _site(
        tmp_path,
        {
            "index.html": '<a href="guide.html">g</a><a href="guide.html#s">s</a>',
            "guide.html": '<h2 id="s">s</h2><a href="index.html">home</a>',
        },
    )
    assert check(tmp_path) is False


def test_missing_target_fails(check, tmp_path):
    _site(tmp_path, {"index.html": '<a href="gone.html">g</a>'})
    assert check(tmp_path) is True


def test_missing_anchor_fails(check, tmp_path):
    _site(
        tmp_path,
        {
            "index.html": '<a href="guide.html#nope">n</a>',
            "guide.html": '<h2 id="real">real</h2>',
        },
    )
    assert check(tmp_path) is True


def test_external_links_are_not_fetched(check, tmp_path):
    # Offline by contract: a scheme or netloc means skip, however dead the URL.
    _site(
        tmp_path,
        {"index.html": '<a href="https://example.invalid/nope.html">x</a>'},
    )
    assert check(tmp_path) is False


def test_directory_href_resolves_to_index(check, tmp_path):
    _site(
        tmp_path,
        {"index.html": '<a href="sub/">s</a>', "sub/index.html": "<p>s</p>"},
    )
    assert check(tmp_path) is False


def test_escaping_the_root_fails(check, tmp_path):
    # ../ out of the site is a missing target, not a silent pass on a real file.
    root = tmp_path / "site"
    root.mkdir()
    (tmp_path / "outside.html").write_text("<html></html>", encoding="utf-8")
    _site(root, {"index.html": '<a href="../outside.html">o</a>'})
    assert check(root) is True


def test_name_attribute_counts_as_an_anchor(check, tmp_path):
    _site(
        tmp_path,
        {
            "index.html": '<a href="guide.html#old">o</a>',
            "guide.html": '<a name="old"></a>',
        },
    )
    assert check(tmp_path) is False


def test_missing_root_index_is_an_error(check, tmp_path):
    with pytest.raises(ValueError):
        check(tmp_path)
