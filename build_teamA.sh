#!/bin/bash
set -e
cd /apps/mingzliu/fsdp_sdma_team/worktrees/teamA
BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF python3 setup.py build_ext --inplace 2>&1 | tail -8
# Import check MUST point at the worktree python, else it resolves the in-image
# /root/wuyl/mori (which lacks HierAllGather) and falsely reports a build failure.
PYTHONPATH=/apps/mingzliu/fsdp_sdma_team/worktrees/teamA/python \
  python3 -c "import mori.ccl; print('HierAllGather', hasattr(mori.ccl,'HierAllGather'))"
