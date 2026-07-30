#!/bin/bash
# T21 discriminator: is the GPU memory-access fault that poisons the pool a bug
# in `test_finalized_getters_raise`, or is it the DESTROY+RECREATE path itself?
#
# T20's faulthandler stack named the abort site exactly:
#   test_dispatch_combine_role_switch.py:416 in _worker_finalized_getters_raise
#   -> _run_once -> run_test_once -> sync() -> torch.cuda.synchronize
# i.e. the fault is in the PRE-finalize dispatch of a FRESHLY CONSTRUCTED op,
# not in the finalized-getter probes the test is named for. And T8 measured that
# same test passing ALONE (`1 passed in 27.13s`).
#
# The one fact the T20 log adds is adjacency: the marker immediately before the
# fault is `[real-capacities] rank 0: RAN 4096 -> 128 -> 4096`, whose worker
# ends with `op.finalize()` at 4096. So the suspect is not the test, it is
# `finalize()` at a LARGE capacity followed by a construct+dispatch at a SMALL
# one in the same process -- which is precisely the flip COORD turn 19 tells
# Team S to perform.
#
# A FILE, not an ssh one-liner: the remote login shell is tcsh.
#
# $1 = mode: pair | alone | serial
MODE="${1:-pair}"
cd /home/mingzliu/pdrs_ext_team/worktrees/teamM || exit 99
export MORI_TEST_WORLD_SIZE=8
export MORI_SHMEM_HEAP_SIZE=8G
export MORI_TEST_RESULT_TIMEOUT=300
export MORI_TEST_SHUTDOWN_TIMEOUT=60

case "$MODE" in
  alone)
    # Control arm. Reproduces T8's green if the fault needs a predecessor.
    NAME=t21alone
    KEXPR="finalized_getters"
    ;;
  pair)
    # Test arm. Same two tests, same order pytest ran them in during T20.
    NAME=t21pair
    KEXPR="real_capacities or finalized_getters"
    ;;
  serial)
    # Localization arm: AMD_SERIALIZE_KERNEL makes the fault report AT the
    # offending launch instead of at the next synchronize, which is the only
    # way to convert "some kernel in this test faults" into a line number.
    NAME=t21serial
    KEXPR="real_capacities or finalized_getters"
    export AMD_SERIALIZE_KERNEL=3
    export HIP_LAUNCH_BLOCKING=1
    ;;
  # Prefix-bisect over the predecessors of `finalized_getters`: T21's `pair`
  # arm refuted adjacency (real_capacities -> finalized_getters passes in 39s),
  # so the latent state is left by a test FURTHER back in the file order.
  #
  # The -k expressions live HERE, not in the ssh argument, because the remote
  # login shell is tcsh: passing a quoted expression through
  # `ssh host 'docker exec ... bash -lc "... -k \"a or b\""'` died on
  # `Unmatched '"'`. Same quoting layer that made every published PYTEST_RC
  # measure `tail` (RESULTS_M T5a). One named mode per selection.
  inj)
    # The fault-injection half: these deliberately abort allocations partway
    # through buffer setup, so they are the prime suspects for leaving a
    # kernel-visible pointer table half-written.
    NAME=t21binj
    KEXPR="plain_device_oom or repeated_failed_flips or rank_asymmetric_failure or rank_asymmetric_unrecoverable or finalized_getters"
    ;;
  early)
    # The non-injection half that runs before finalized_getters.
    NAME=t21bearly
    KEXPR="flip_and_flip_back or leak_stress or finalize_returns or public_reject or max_total_recv or rejects_layout or noop_and_finalize or oom_rolls_back or real_capacities or finalized_getters"
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 98 ;;
esac

export MORI_TEST_FULL_LOG="/home/mingzliu/pdrs_ext_team/logs/mori_test_M_${NAME}_full.log"
bash tests/python/ops/run_role_switch_suite.sh test -k "$KEXPR"
echo "RUNNER_RC=$?"
