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
"""Audit the transcripts of a gemm_ar_bf16 session for the clean-room rule.

    python audit.py <transcript.jsonl or session directory>... [--json <report>]
        [--hidden hidden_paths.txt] [--repo <checkout>] [--base <commit>]
        [--visible <dir>]...

The rule: no text of a hidden file (``hidden_paths.txt``) reaches the session. The
settings deny Claude Code's file tools, and the file commands it recognizes in Bash, on
those paths; a script that opens a file itself, a grep over a parent directory or
``git show`` gets past them. So after the run this reads every ``*.jsonl`` given or found
under a session directory (``subagents/`` included) and the files under
``tool-results/``, and looks for the hidden files' text in what the session was shown:
tool results, attachments, and user and system messages. A path or a module name alone
is not text: the facts cite hidden files, and that is allowed.

A hidden file's fingerprint is its lines of at least ``--min-line`` characters, with
whitespace collapsed, that no other file at ``--base`` contains, leaving out boilerplate
any file could share (imports, ``add_argument`` lines, licence lines). ``--base`` defaults to
the last commit that changed ``hidden_paths.txt``, so that code written during the
session cannot hide a copied line. ``--visible`` adds a directory whose files count as
visible, for a library that shares code with the hidden files.

Each finding is one of:

* ``leak``: one fingerprint line whose source is a hidden file (the Read tool's path, a
  grep prefix, a diff header, an attachment's file name), or one from a call that named
  that file, or three from one hidden file with no such source. Any leak voids the run
  unless a person shows that the lines are a coincidence.
* ``review``: one or two lines with no source, or hidden source lines inside a Python
  traceback (a black-box baseline that failed). A person decides.
* ``blocked``: a call that named a hidden path or module and was refused.
* ``attempt``: a call that named a hidden path or module, ran, and returned no hidden
  text (``ls``, ``wc``, a failed run).

Exit status: 1 with a leak, 0 without, 2 on bad input. Review, blocked and attempt items
are printed for a person to read; ``--json`` writes the full report.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SCHEMA = "cco-fusion/audit/v1"
MIN_LINE = 30
PLAIN_LEAK = 3  # lines from one hidden file, with no source, that make a leak
BOILERPLATE = re.compile(
    r"^(import |from \S+ import |#include)|add_argument\(|(?i:spdx-license|copyright|"
    r"permission is hereby granted|licen[cs]e)"
)
MAX_FILE = 4 << 20
DENIAL = re.compile(r"(?i)\b(denied|deny rule|not permitted|permission)\b")
LINE_NO = re.compile(r"^\s*\d+(?:\t|→)")
GREP = re.compile(r"^(?P<path>[^\s:]*[/.][^\s:]*?)[:-]\d+[:-]")
GREP_N = re.compile(r"^\d+[:-]")
DIFF_FILE = re.compile(r"^(?:\+\+\+ b/|diff --git a/\S+ b/)(?P<path>\S+)")
FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line \d+')
KINDS = ("leak", "review", "blocked", "attempt")


def read_patterns(path):
    patterns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            patterns.append(line)
    return patterns


def hidden_pattern(path, patterns):
    """The pattern that hides a file path, or None. A directory pattern ends in "/"."""
    p = "/" + str(path).replace("\\", "/").lstrip("/")
    for pattern in patterns:
        if pattern.endswith("/"):
            if "/" + pattern in p:
                return pattern
        elif p.endswith("/" + pattern):
            return pattern
    return None


def name_regexes(patterns):
    """(pattern, regex) pairs that find a hidden path, or a hidden module, in a call."""
    out = []
    for pattern in patterns:
        stem = re.escape(pattern.rstrip("/"))
        tail = r"(?![\w.-])" if pattern.endswith("/") else r"(?!\w)"
        out.append((pattern, re.compile(r"(?<![\w-])" + stem + tail)))
        if pattern.endswith(".py"):
            module = re.escape(pattern[:-3].replace("/", "."))
            out.append((pattern, re.compile(r"(?<![\w.])" + module + r"(?!\w)")))
    return out


def norm(line):
    return " ".join(line.split())


def _git(repo, *args, data=None):
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        input=data,
        capture_output=True,
        check=True,
    ).stdout


def default_base(repo, hidden_file):
    """The last commit that changed the hidden list; the session starts after it."""
    try:
        rel = Path(hidden_file).resolve().relative_to(Path(repo).resolve())
    except ValueError:
        return "HEAD"
    return _git(repo, "log", "-1", "--format=%H", "--", str(rel)).decode().strip()


def tree_files(repo, base):
    """{path: text} of the text files in the tree of ``base``."""
    entries = []
    for rec in _git(repo, "ls-tree", "-r", "-z", "--full-tree", base).split(b"\0"):
        if rec:
            meta, path = rec.split(b"\t", 1)
            _, kind, sha = meta.split()
            if kind == b"blob":
                entries.append((path.decode("utf-8", "replace"), sha))
    out = _git(
        repo, "cat-file", "--batch", data=b"".join(s + b"\n" for _, s in entries)
    )
    files, at = {}, 0
    for path, _ in entries:
        nl = out.index(b"\n", at)
        size = int(out[at:nl].split()[2])
        body = out[nl + 1 : nl + 1 + size]
        at = nl + 2 + size
        if size <= MAX_FILE and b"\0" not in body:
            files[path] = body.decode("utf-8", "replace")
    return files


def fingerprints(files, patterns, min_line=MIN_LINE, visible_dirs=()):
    """{normalized line: [hidden paths]} for the lines that only hidden files hold."""
    hidden, visible = defaultdict(set), set()
    for path, text in files.items():
        is_hidden = hidden_pattern(path, patterns)
        for line in text.splitlines():
            n = norm(line)
            if not is_hidden:
                visible.add(n)
            elif (
                len(n) >= min_line
                and re.search(r"[A-Za-z]", n)
                and not BOILERPLATE.search(n)
            ):
                hidden[n].add(path)
    for root in visible_dirs:
        for f in Path(root).rglob("*"):
            if not f.is_file() or hidden_pattern(f, patterns):
                continue
            try:
                if f.stat().st_size <= MAX_FILE:
                    visible.update(
                        map(norm, f.read_text(encoding="utf-8").splitlines())
                    )
            except (OSError, UnicodeDecodeError):
                continue
    return {n: sorted(p) for n, p in hidden.items() if n not in visible}


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
    """The text of a tool result's content: a string or a list of blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in value
        )
    return "" if value is None else json.dumps(value, ensure_ascii=False)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def read_session(files):
    """Calls by id, results by call id, and the other context the session was shown."""
    calls, results, context = {}, {}, []
    for f in files:
        if (
            f.suffix != ".jsonl"
        ):  # tool-results/<call id>.txt: a result too long to inline
            r = results.setdefault(f.stem, {"texts": [], "error": False, "at": []})
            r["texts"].append(f.read_text(encoding="utf-8", errors="replace"))
            r["at"].append(str(f))
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for number, line in enumerate(fh, 1):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                at, kind = f"{f}:{number}", record.get("type")
                content = (record.get("message") or {}).get("content")
                if kind == "assistant" and isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            calls[b.get("id")] = {
                                "tool": b.get("name"),
                                "input": b.get("input") or {},
                                "at": at,
                            }
                elif kind == "user":
                    blocks = content if isinstance(content, list) else [content]
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            r = results.setdefault(
                                b.get("tool_use_id"),
                                {"texts": [], "error": False, "at": []},
                            )
                            r["texts"].append(_text(b.get("content")))
                            r["error"] = r["error"] or bool(b.get("is_error"))
                            r["at"].append(at)
                        elif isinstance(b, dict) and b.get("type") == "text":
                            context.append({"text": b.get("text", ""), "at": at})
                        elif isinstance(b, str):
                            context.append({"text": b, "at": at})
                elif kind == "attachment":
                    a = record.get("attachment")
                    source = a.get("filename") if isinstance(a, dict) else None
                    text = "\n".join(_strings(a))
                    context.append({"text": text, "at": at, "source": source})
                elif kind == "system":
                    text = "\n".join(_strings(record.get("content")))
                    context.append({"text": text, "at": at})
    return calls, results, context


def match_lines(text, fp, patterns, source=None):
    """[(line, hidden paths, how)] for the fingerprint lines in a text.

    ``how`` is "source" when the line's own file is hidden (a grep prefix, a diff
    header, or ``source``), "traceback" for the source line of a frame in a hidden
    file, and "plain" otherwise.
    """
    hits, diff_path, frame = [], None, False
    for raw in text.splitlines():
        in_frame, frame = frame, False
        m = DIFF_FILE.match(raw)
        if m:
            diff_path = m.group("path")
            continue
        m = FRAME.match(raw)
        if m:
            frame = bool(hidden_pattern(m.group("path"), patterns))
            continue
        line = LINE_NO.sub("", raw, count=1)
        forms, origin = [], source
        g = GREP.match(line)
        if g:
            forms.append(line[g.end() :])
            origin = g.group("path")
        elif GREP_N.match(line):
            forms.append(GREP_N.sub("", line, count=1))
        if diff_path and line[:1] in "+- ":
            forms.append(line[1:])
            origin = origin if g else diff_path
        forms.append(line)
        found = next((n for n in map(norm, forms) if n in fp), None)
        if found is None:
            continue
        if in_frame:
            how = "traceback"
        elif origin and hidden_pattern(origin, patterns):
            how = "source"
        else:
            how = "plain"
        hits.append((found, fp[found], how))
    return hits


def _judge(hits, named_files):
    """Findings for one result or context block: [(kind, hidden path, lines)]."""
    by_file = defaultdict(lambda: {"source": [], "traceback": [], "plain": []})
    for line, paths, how in hits:
        for p in paths:
            by_file[p][how].append(line)
    out = []
    for path, h in sorted(by_file.items()):
        plain = sorted(set(h["plain"]))
        lines = sorted(set(h["source"]) | set(plain) | set(h["traceback"]))
        if h["source"] or len(plain) >= PLAIN_LEAK or (plain and path in named_files):
            out.append(("leak", path, lines))
        else:
            out.append(("review", path, lines))
    return out


def audit(paths, patterns, fp):
    files = transcript_files(paths)
    calls, results, context = read_session(files)
    regexes = name_regexes(patterns)
    findings = []

    def add(kind, path, lines, where, call=None, call_id=None):
        findings.append(
            {
                "kind": kind,
                "hidden": path,
                "lines": len(lines),
                "sample": lines[:3],
                "at": where,
                "tool": (call or {}).get("tool"),
                "call_id": call_id,
                "call": (
                    json.dumps((call or {}).get("input"), ensure_ascii=False)[:300]
                    if call
                    else None
                ),
            }
        )

    for call_id in sorted(set(calls) | set(results), key=str):
        call, result = calls.get(call_id), results.get(call_id)
        named = []
        if call:
            text = json.dumps(call["input"], ensure_ascii=False)
            named = sorted({p for p, rx in regexes if rx.search(text)})
        source = None
        if call and call["tool"] in ("Read", "NotebookRead"):
            source = call["input"].get("file_path") or call["input"].get(
                "notebook_path"
            )
        judged = []
        if result:
            hits = match_lines("\n".join(result["texts"]), fp, patterns, source)
            named_files = {f for line, ps, _ in hits for f in ps for p in named}
            named_files = {f for f in named_files if hidden_pattern(f, named)}
            judged = _judge(hits, named_files)
        where = (call or {}).get("at") or (result or {}).get("at", [None])[0]
        for kind, path, lines in judged:
            add(kind, path, lines, where, call, call_id)
        if named and not any(k == "leak" for k, _, _ in judged):
            refused = bool(result) and result["error"]
            refused = refused and DENIAL.search("\n".join(result["texts"]))
            for p in named:
                add("blocked" if refused else "attempt", p, [], where, call, call_id)

    for block in context:
        hits = match_lines(block["text"], fp, patterns, block.get("source"))
        for kind, path, lines in _judge(hits, set()):
            add(kind, path, lines, block["at"])
    return findings, files


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit a session's transcripts for hidden text."
    )
    parser.add_argument(
        "paths", nargs="+", help="transcript .jsonl files or session directories"
    )
    parser.add_argument("--hidden", default=str(HERE / "hidden_paths.txt"))
    parser.add_argument("--repo", default=str(REPO), help="checkout holding --base")
    parser.add_argument(
        "--base", help="commit the session started from (default: see the docstring)"
    )
    parser.add_argument(
        "--visible",
        action="append",
        default=[],
        help="a directory of files that count as visible (repeatable)",
    )
    parser.add_argument("--min-line", type=int, default=MIN_LINE)
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args(argv)
    try:
        patterns = read_patterns(args.hidden)
        base = args.base or default_base(args.repo, args.hidden) or "HEAD"
        files = tree_files(args.repo, base)
        fp = fingerprints(files, patterns, args.min_line, args.visible)
        findings, transcripts = audit(args.paths, patterns, fp)
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        print(f"audit.py: {e}", file=sys.stderr)
        return 2
    counts = {k: sum(f["kind"] == k for f in findings) for k in KINDS}
    hidden_files = sorted(p for p in files if hidden_pattern(p, patterns))
    if not hidden_files or not fp:
        print(f"audit.py: no hidden text at {base}; check --base", file=sys.stderr)
        return 2
    verdict = (
        "void"
        if counts["leak"]
        else ("review" if counts["review"] or counts["attempt"] else "clean")
    )
    report = {
        "schema": SCHEMA,
        "verdict": verdict,
        "base": _git(args.repo, "rev-parse", base).decode().strip(),
        "hidden_files": len(hidden_files),
        "fingerprint_lines": len(fp),
        "transcripts": [str(f) for f in transcripts],
        "counts": counts,
        "findings": findings,
    }
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    for f in sorted(findings, key=lambda f: KINDS.index(f["kind"])):
        detail = f"{f['lines']} lines" if f["lines"] else (f["call"] or "")[:100]
        print(f"{f['kind']:8} {f['hidden']}  {f['tool'] or 'context'}: {detail}")
        print(f"         at {f['at']}")
        for line in f["sample"][:1]:
            print(f"         > {line[:110]}")
    print(
        f"audit: {verdict}; base {report['base'][:12]}, {len(hidden_files)} hidden files, "
        f"{len(fp)} fingerprint lines; " + ", ".join(f"{counts[k]} {k}" for k in KINDS)
    )
    return 1 if counts["leak"] else 0


if __name__ == "__main__":
    sys.exit(main())
