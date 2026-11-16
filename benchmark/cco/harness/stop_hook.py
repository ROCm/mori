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
"""Stop hook that keeps a cco-fusion-loop session working until its task is finished.

Inert unless ``CCO_FUSION_RECORDS`` names the task's records directory. Then, each time
the main session is about to stop, it asks ``loop_state.py``: on ``continue`` it blocks the
stop and hands the session the next step; on ``done``, ``needs-owner`` or ``stopped`` it
lets the session end. Three brakes keep it from running away:

* the owner creates ``<task>/STOP``;
* at most ``CCO_FUSION_MAX_CONTINUES`` blocks per session (default 100);
* when nothing under the task directory changed across ``CCO_FUSION_MAX_FLAT``
  consecutive blocks (default 3), it writes ``<task>/BLOCKED`` saying so and lets the
  session end, for the owner to look.

Its bookkeeping is ``<task>/.driver.json``. Settings register it under ``hooks.Stop``;
``settings.example.json`` and the cases' settings do.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loop_state  # noqa: E402

BOOK = ".driver.json"


def fingerprint(task):
    """What changes when the session makes progress: every file's path, size and mtime."""
    h = hashlib.sha1()
    for p in sorted(Path(task).rglob("*")):
        if p.is_file() and p.name != BOOK:
            st = p.stat()
            h.update(
                f"{p.relative_to(task)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode()
            )
    return h.hexdigest()


def decide(hook_input, env):
    """(output dict or None, bookkeeping to save or None) for one Stop event."""
    records = env.get("CCO_FUSION_RECORDS")
    if not records:
        return None, None
    task = Path(records)
    if not task.is_dir():
        return {
            "systemMessage": f"cco-fusion driver: no records directory {task}"
        }, None
    s = loop_state.state(task)
    if s["state"] in loop_state.TERMINAL:
        return {
            "systemMessage": f"cco-fusion driver: {s['state']}: {'; '.join(s['reasons'])}"
        }, None

    book_path = task / BOOK
    try:
        book = json.loads(book_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        book = {}
    session = hook_input.get("session_id") or "?"
    if book.get("session") != session:
        book = {"session": session, "continues": 0, "flat": 0, "fingerprint": None}
    fp = fingerprint(task)
    book["flat"] = book["flat"] + 1 if fp == book["fingerprint"] else 0
    book["fingerprint"] = fp
    max_continues = int(env.get("CCO_FUSION_MAX_CONTINUES", 100))
    max_flat = int(env.get("CCO_FUSION_MAX_FLAT", 3))

    if book["continues"] >= max_continues or book["flat"] >= max_flat:
        why = (
            f"the driver continued this session {book['continues']} times"
            if book["continues"] >= max_continues
            else f"nothing in the records changed across {book['flat']} continuations"
        )
        (task / "BLOCKED").write_text(
            f"Stopped by the unattended driver: {why}. The last state was:\n\n"
            f"{loop_state.summary(s)}\n\nLook at the session, then delete this file and "
            "resume it.\n",
            encoding="utf-8",
        )
        return {"systemMessage": f"cco-fusion driver: stopping, {why}"}, book

    book["continues"] += 1
    reason = (
        f"cco-fusion driver (continuation {book['continues']}/{max_continues}): the task "
        f"in {task} is not finished.\n{loop_state.summary(s)}\n"
        "Carry on with the cco-fusion-loop skill and keep the records current. If you "
        f"need the owner (GPUs, a decision, a permission), write the question to "
        f"{task / 'BLOCKED'} and end the turn."
    )
    return {"decision": "block", "reason": reason}, book


def main():
    try:
        hook_input = json.load(sys.stdin)
    except ValueError:
        hook_input = {}
    out, book = decide(hook_input, os.environ)
    if book is not None:
        task = Path(os.environ["CCO_FUSION_RECORDS"])
        (task / BOOK).write_text(json.dumps(book) + "\n", encoding="utf-8")
    if out:
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
