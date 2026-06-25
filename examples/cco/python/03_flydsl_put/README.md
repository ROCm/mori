# CCO Example 03 — FlyDSL GDA put + signal/wait

Device-initiated cross-node RDMA from a FlyDSL `@flyc.kernel`, using the cco
device bindings in `mori.cco.device.flydsl`.

## What it shows
- Passing a cco device communicator into a FlyDSL kernel as a **device-resident
  handle** (`dc.device_ptr`), and windows as their handles (`win.handle`).
- Calling the cco GDA device API from the kernel: `Gda.put(...)` with a
  completion `SignalOp.INC`, `Gda.flush(...)`, and `Gda.wait_signal(...)`.

## Prerequisites
- `mori` built with the cco host extension (Cython `mori.cco.cco`).
- The cco device bitcode `libmori_cco_device.bc`:
  ```
  bash tools/build_cco_bitcode.sh        # auto arch+NIC, cov=6, -> lib/
  ```
  (or set `MORI_CCO_BC=/path/to/libmori_cco_device.bc`)
- FlyDSL installed, `mpi4py`, and 2 GPUs across 2 nodes for CROSSNODE GDA.

## Run
```
mpirun -n 2 python main.py
```
Rank 0 fills its send window and issues one 1 MiB GDA put into rank 1's recv
window with a signal; rank 1 waits on the signal; the host validates the payload.
