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
"""Audit Claude Code transcripts for access to the case's hidden paths.

    python audit.py <transcript.jsonl or session directory>... [--hidden hidden_paths.txt]

The clean-room rule of a case is enforced after the fact: settings can deny the Read
tool, but a shell command can still read anything. This scans every ``*.jsonl`` given,
or found under a directory (a session's ``subagents/`` included), and every file under
a ``tool-results/`` directory, for two things:

* **access**: a tool call whose input names a hidden path (the implementer asked for it);
* **exposure**: a tool result whose text names one (it came back, e.g. from a grep over a
  directory, whether or not it was asked for).

A line of ``hidden_paths.txt`` matches any text that contains it; ``#`` starts a
comment. Exit status: 0 when nothing is found, 1 when something is, 2 on bad input.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_patterns(path):
    patterns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            patterns.append(line)
    return patterns


def transcript_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.rglob("*.jsonl"))
            files += sorted(
                f for f in p.rglob("*") if f.is_file() and "tool-results" in f.parts
            )
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return list(dict.fromkeys(files))


def _text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def scan_record(record, patterns):
    """(kind, tool, pattern, excerpt) for every hit in one transcript record."""
    hits = []
    content = (record.get("message") or {}).get("content")
    if not isinstance(content, list):
        return hits
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            kind, tool, text = "access", block.get("name"), _text(block.get("input"))
        elif block.get("type") == "tool_result":
            kind, tool, text = (
                "exposure",
                block.get("tool_use_id"),
                _text(block.get("content")),
            )
        else:
            continue
        for pattern in patterns:
            at = text.find(pattern)
            if at >= 0:
                hits.append(
                    (
                        kind,
                        tool,
                        pattern,
                        text[max(0, at - 60) : at + len(pattern) + 60],
                    )
                )
    return hits


def audit(paths, patterns):
    findings = []
    for f in transcript_files(paths):
        if f.suffix != ".jsonl":
            text = f.read_text(encoding="utf-8", errors="replace")
            for pattern in patterns:
                if pattern in text:
                    findings.append(
                        {
                            "file": str(f),
                            "line": None,
                            "kind": "exposure",
                            "tool": None,
                            "pattern": pattern,
                            "excerpt": "",
                        }
                    )
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for number, line in enumerate(fh, 1):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                for kind, tool, pattern, excerpt in scan_record(record, patterns):
                    findings.append(
                        {
                            "file": str(f),
                            "line": number,
                            "kind": kind,
                            "tool": tool,
                            "pattern": pattern,
                            "excerpt": excerpt,
                        }
                    )
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit transcripts for access to hidden paths."
    )
    parser.add_argument(
        "paths", nargs="+", help="transcript .jsonl files or session directories"
    )
    parser.add_argument("--hidden", default=str(HERE / "hidden_paths.txt"))
    parser.add_argument("--json", help="write the findings here")
    args = parser.parse_args(argv)
    try:
        findings = audit(args.paths, read_patterns(args.hidden))
    except OSError as e:
        print(f"audit.py: {e}", file=sys.stderr)
        return 2
    if args.json:
        Path(args.json).write_text(
            json.dumps(findings, indent=2) + "\n", encoding="utf-8"
        )
    for f in findings:
        where = f"{f['file']}:{f['line']}" if f["line"] else f["file"]
        print(f"{f['kind']:8} {f['pattern']}  [{where}]  {f['excerpt'][:120]!r}")
    access = sum(f["kind"] == "access" for f in findings)
    print(f"audit: {access} access, {len(findings) - access} exposure")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
