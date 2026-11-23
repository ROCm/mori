#!/bin/bash
# Run test_proxy_put across two nodes, one rank per node on GPU 0, from node1 of
# a CI internode job. Builds the test in its own cmake tree in node1's container
# (the wheel's tree was configured by pip's isolated cmake), copies the binary
# and the libraries it links from that tree to the same path on node2, and fails
# unless both ranks print PASSED; a SKIP fails too.
#
# Usage: run_cco_proxy_internode.sh [--cases LIST] [--negative LIST]
#   --cases     positive cases (MORI_CCO_PROXY_TEST_CASES); default "all",
#               "atomic_free" for NICs without atomics into GPU memory
#   --negative  negative cases (MORI_CCO_PROXY_TEST_NEGATIVE); default "all"
# Reads the internode jobs' env: CT, CONTAINER, SSH_OPTS, NODE1_IFNAME,
# NODE2_HOST, NODE2_PORT, GITHUB_WORKSPACE, RUNNER_TEMP.
set -uo pipefail

CASES=all
NEGATIVE=all
while [ $# -gt 0 ]; do
  case "$1" in
    --cases) CASES=$2; shift 2 ;;
    --negative) NEGATIVE=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

TREE=$GITHUB_WORKSPACE/build_proxy
BIN=$TREE/tests/cpp/cco/test_proxy_put
UID_FILE=$TREE/proxy_uid.bin
NODE2=(ssh $SSH_OPTS -p "$NODE2_PORT" "$(whoami)@$NODE2_HOST")

# GPU and NIC are detected in the container, as pip install does.
$CT exec "$CONTAINER" bash -c "cd $GITHUB_WORKSPACE && \
  cmake -S . -B $TREE -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_PYBINDS=OFF -DBUILD_UMBP=OFF \
    -DWITH_MPI=OFF > /dev/null && \
  cmake --build $TREE --target test_proxy_put -j \$(nproc)" || exit 1
LIBS=$($CT exec "$CONTAINER" ldd "$BIN" | awk -v d="$TREE/" 'index($3, d) == 1 {print $3}')
$CT exec "$CONTAINER" "$BIN" --gen-uid "$NODE1_IFNAME" 0 "$UID_FILE" || exit 1
tar -cPhf - "$BIN" $LIBS "$UID_FILE" | "${NODE2[@]}" "tar -xPf -" || exit 1

ENVS="-e HIP_VISIBLE_DEVICES=0 -e MORI_CCO_PROXY_TEST_CASES=$CASES -e MORI_CCO_PROXY_TEST_NEGATIVE=$NEGATIVE"
LOG1=$RUNNER_TEMP/proxy_node1.log
LOG2=$RUNNER_TEMP/proxy_node2.log
"${NODE2[@]}" "$CT exec $ENVS $CONTAINER timeout 600 $BIN --rank 1 --world 2 --uid-file $UID_FILE" \
  > "$LOG2" 2>&1 &
NODE2_PID=$!
RC=0
$CT exec $ENVS "$CONTAINER" timeout 600 "$BIN" --rank 0 --world 2 --uid-file "$UID_FILE" \
  > "$LOG1" 2>&1 || RC=1
wait "$NODE2_PID" || RC=1
for LOG in "$LOG1" "$LOG2"; do
  echo "=== $LOG ==="
  cat "$LOG"
  if grep -q SKIP "$LOG"; then echo "test_proxy_put SKIPped"; RC=1; fi
  grep -q '\] PASSED' "$LOG" || RC=1
done
exit $RC
