#!/bin/bash

debug=${debug:-0}
profile=${profile:-0}

GDB=
if [[ ${debug} -eq 1 ]]; then
  GDB="rocgdb --args "
fi

DUMP=uu_dump
rm -rf $DUMP
mkdir $DUMP

PYEXEC=$(which python3)

TEST=mori_playground.py
#TEST=examples/ops/dispatch_combine/test_dispatch_combine.py

export XLA_FLAGS="--xla_dump_to=$$DUMP"

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib
$GDB $PYEXEC $TEST