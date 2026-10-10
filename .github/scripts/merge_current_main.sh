#!/usr/bin/env bash
# Merge the current origin/main into the checkout of a re-run PR CI run.
#
# A pull_request run checks out the PR merged with main as of when the run was
# created, and a re-run ("Re-run all jobs" or /ci-rerun, see ci-rerun.yml)
# reuses that same merge commit. ci.yml and ci_cco.yml call this on re-runs
# (run_attempt > 1) so they test against the current main instead.
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
  echo "::error::The PR (${head}) conflicts with the current main (${main}); resolve the conflict and push."
  exit 1
fi
git submodule update --init
echo "Testing ${head} merged with the current main ${main} as $(git rev-parse --short HEAD)"
