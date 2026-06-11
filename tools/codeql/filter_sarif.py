#!/usr/bin/env python3
# Drop CodeQL results located in third-party / test / build-artifact paths.
#
# CodeQL's `paths-ignore` config is not reliably applied to *compiled* languages
# (C/C++): the extractor still compiles vendored/test sources pulled into the
# build, so their alerts appear in the SARIF. This post-processes the SARIF to
# remove any result whose location falls under an excluded path prefix, so the
# uploaded report reflects MoRI's own shippable code only.
#
# Usage: filter_sarif.py FILE.sarif [FILE2.sarif ...]   (edits files in place)

import json
import sys

# Repo-root-relative path prefixes / substrings that are not MoRI's own code.
EXCLUDE_PREFIXES = ("3rdparty/", "build/", "tests/")
EXCLUDE_SUBSTRINGS = ("/_deps/", "/googletest", "/3rdparty/")


def is_excluded(uri: str) -> bool:
    u = uri.lstrip("./")
    if u.startswith(EXCLUDE_PREFIXES):
        return True
    return any(s in u for s in EXCLUDE_SUBSTRINGS)


def result_uri(result: dict) -> str:
    try:
        return result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    except (KeyError, IndexError, TypeError):
        return ""


def main(paths):
    for path in paths:
        with open(path) as f:
            sarif = json.load(f)

        removed = 0
        for run in sarif.get("runs", []):
            kept = []
            for res in run.get("results", []):
                if is_excluded(result_uri(res)):
                    removed += 1
                else:
                    kept.append(res)
            run["results"] = kept

        with open(path, "w") as f:
            json.dump(sarif, f)
        print(f"{path}: removed {removed} third-party/test/build result(s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: filter_sarif.py FILE.sarif [...]", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1:])
