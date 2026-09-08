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
"""Machine-readable result rows, in aiter's ``print_json_table`` format.

One single-line JSON object per table, ``{"name": ..., "rows": [...]}``, so a
parent driver can validate and forward it without parsing a human table. The
format is aiter's ``aiter/test_common.py``; this is a copy rather than an import,
and pure Python rather than pandas -- a benchmark should not need a dataframe to
print its own results, and mori's test env does not ship pandas.
"""

from __future__ import annotations

import json


def print_json_table(name, rows) -> None:
    """Print rows as one JSON object, dropping columns no row ever filled in.

    Column order is first-seen across the rows, matching what pandas builds from
    a list of dicts. Empty and ``None`` cells are dropped only when the WHOLE
    column is empty, so a value missing from one row still shows as null there.
    """
    rows = [r for r in rows if r is not None]
    cols: list[str] = []
    for row in rows:
        cols += [k for k in row if k not in cols]
    keep = [c for c in cols if any(r.get(c) not in (None, "") for r in rows)]
    records = [{c: _plain(row.get(c)) for c in keep} for row in rows]
    print(json.dumps({"name": name, "rows": records}), flush=True)


def _plain(value):
    """Anything json cannot encode becomes its str(), the way to_json would."""
    if value is None or isinstance(value, (bool, int, float, str, list, dict)):
        return value
    return str(value)
