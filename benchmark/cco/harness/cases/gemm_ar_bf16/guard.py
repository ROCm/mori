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
"""PreToolUse hook of a gemm_ar_bf16 session: refuse Bash commands that name hidden code.

Claude Code applies the settings' Read denies to its file tools and to the file commands
it recognizes in Bash, not to a script that opens a file itself. This hook closes the
routes that name the file in the command: ``python3 -c "open(...)"``, a heredoc script,
``git show HEAD:<path>``, ``grep <pattern> <hidden file>``, ``import mori.ops.gemm_ar.op``.
It also refuses a recursive search (``grep -r``, ``rg``, ``ag``, ``ack``, ``git grep``,
``find ... | xargs grep``) whose scope holds a hidden file, in the repository or in an
installed copy: Claude Code's Grep tool skips the denied files, so the agent is pointed
there. ``audit.py`` still checks the transcripts for what neither catches, such as a
script written earlier that opens a hidden file.

``settings.implementer.json`` runs it before every Bash call, the subagents' included.
It reads the hook input on stdin and prints a deny decision, or nothing.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit  # noqa: E402

REPO = HERE.parents[4]
SEARCHERS = {"rg", "ag", "ack", "ack-grep"}
GREPS = {"grep", "egrep", "fgrep"}
RECURSIVE = re.compile(
    r"^(-[A-Za-z]*[rR][A-Za-z]*|--recursive|--dereference-recursive)$"
)
WRAPPERS = {"sudo", "env", "nice", "nohup", "time", "timeout", "stdbuf", "command"}
WALK_LIMIT = 300_000


def named(command, patterns):
    """The hidden patterns a command names, as a path or as a module."""
    return sorted({p for p, rx in audit.name_regexes(patterns) if rx.search(command)})


def _simple_commands(command):
    """The words of each simple command, split at pipes and command separators."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    out, words = [], []
    try:
        for tok in lexer:
            if tok and set(tok) <= set("|&;()<>"):
                if words:
                    out.append(words)
                words = []
            else:
                words.append(tok)
    except ValueError:
        return []
    if words:
        out.append(words)
    return out


def _strip_wrappers(words):
    while words and (words[0] in WRAPPERS or re.match(r"^\w+=", words[0])):
        words = words[1:]
        while words and words[0].startswith("-"):  # timeout -s 9, env -i
            words = words[1:]
        if words and re.match(r"^\d+[smhd]?$", words[0]):  # timeout 30
            words = words[1:]
    return words


def _operands(args, takes_value=("-e", "-f", "-m", "-A", "-B", "-C", "-g", "-t")):
    """Non-option arguments, with the values of the options that take one dropped."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a == "--":
            continue
        elif a.startswith("-"):
            skip = a in takes_value
        else:
            out.append(a)
    return out


def search_scopes(command, cwd):
    """The paths a recursive search in ``command`` reads, resolved against ``cwd``."""
    scopes, previous = [], []
    for words in _simple_commands(command):
        words = _strip_wrappers(words)
        if not words:
            continue
        name, args = os.path.basename(words[0]), words[1:]
        if name == "cd":
            cwd = str(Path(cwd, os.path.expanduser(args[0] if args else "~")))
            continue
        paths = None
        if name in GREPS and any(RECURSIVE.match(a) or a == "recurse" for a in args):
            ops = _operands(args)
            explicit_pattern = any(
                a in ("-e", "-f") or a.startswith("--regexp") for a in args
            )
            paths = ops if explicit_pattern else ops[1:]
        elif name in SEARCHERS:
            ops = _operands(args)
            paths = ops[1:]
        elif name == "git" and "grep" in args:
            after = args[args.index("grep") + 1 :]
            paths = after[after.index("--") + 1 :] if "--" in after else ["."]
        elif name == "xargs" and any(
            os.path.basename(a) in GREPS | SEARCHERS for a in args
        ):
            paths = previous or ["."]
        elif name == "find":
            previous = [a for a in _operands(args) if not a.startswith(("(", "!"))][:1]
            if "-exec" in args and any(os.path.basename(a) in GREPS for a in args):
                paths = previous or ["."]
            continue
        if paths is not None:
            scopes += [str(Path(cwd, p)) for p in (paths or ["."])]
    return scopes


_HIDDEN_IN_REPO = None


def _hidden_in_repo(patterns):
    global _HIDDEN_IN_REPO
    if _HIDDEN_IN_REPO is None:
        try:
            tracked = subprocess.run(
                ["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "ls-files"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            tracked = []
        _HIDDEN_IN_REPO = [
            str(REPO / f) for f in tracked if audit.hidden_pattern(f, patterns)
        ]
    return _HIDDEN_IN_REPO


def holds_hidden(path, patterns):
    """Whether a search of ``path`` would read a hidden file."""
    path = os.path.realpath(path)
    if audit.hidden_pattern(path, patterns):
        return True
    if any(h == path or h.startswith(path + os.sep) for h in _hidden_in_repo(patterns)):
        return True
    if (
        not os.path.isdir(path)
        or path == str(REPO)
        or path.startswith(str(REPO) + os.sep)
    ):
        return False
    seen = 0  # outside the repository: an installed copy, found by walking
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules")]
        for f in files:
            if audit.hidden_pattern(os.path.join(root, f), patterns):
                return True
        seen += len(files) + len(dirs)
        if seen > WALK_LIMIT:
            return False
    return False


def decision(hook_input, patterns):
    """The hook's output for one PreToolUse input, or None to let the call through."""
    if hook_input.get("tool_name") != "Bash":
        return None
    command = (hook_input.get("tool_input") or {}).get("command") or ""
    hits = named(command, patterns)
    if hits:
        reason = (
            f"Denied by the gemm_ar_bf16 clean room: the command names {', '.join(hits)}, "
            "which is hidden (hidden_paths.txt). Run hidden code only as a black box "
            "through the case's files (op_harness.py --impl existing, arms.bounds.yaml), "
            "and take the facts as written."
        )
    else:
        cwd = hook_input.get("cwd") or os.getcwd()
        wide = [s for s in search_scopes(command, cwd) if holds_hidden(s, patterns)]
        if not wide:
            return None
        reason = (
            f"Denied by the gemm_ar_bf16 clean room: a recursive search of {wide[0]} would "
            "read hidden files (hidden_paths.txt). Use the Grep tool, which skips them, "
            "or search paths that hold no hidden file."
        )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def main():
    try:
        hook_input = json.load(sys.stdin)
    except ValueError:
        return 0
    out = decision(hook_input, audit.read_patterns(HERE / "hidden_paths.txt"))
    if out:
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
