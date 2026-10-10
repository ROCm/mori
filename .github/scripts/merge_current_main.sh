#!/usr/bin/env bash
# Merge the current origin/main into the checked-out PR branch.
#
# Used by ci.yml and ci_cco.yml when ci-rerun.yml starts them with
# workflow_dispatch and merge_main=true: a pull_request run tests the PR merged
# with main, and this gives a re-run the same thing against the main of today.
set -euo pipefail

if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
  git fetch --no-tags --unshallow origin main
else
  git fetch --no-tags origin main
fi

head=$(git rev-parse --short HEAD)
main=$(git rev-parse --short FETCH_HEAD)
if ! git -c user.name="mori-ci" -c user.email="mori-ci@localhost" \
    merge --no-edit --no-ff FETCH_HEAD; then
  echo "::error::The PR branch (${head}) conflicts with main (${main}); resolve the conflict and push."
  exit 1
fi
git submodule update --init
echo "Testing PR branch ${head} merged with main ${main} as $(git rev-parse --short HEAD)"
