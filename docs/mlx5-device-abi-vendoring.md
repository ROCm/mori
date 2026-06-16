# MLX5 device-ABI vendoring — make mlx5 device code `<infiniband/verbs.h>`-free

Handoff doc. **Do this on an MLX5 box** (so the mlx5 RDMA path can be runtime-verified).
These structs overlay NIC hardware memory — a wrong field offset silently corrupts
RDMA, so hardware validation is mandatory.

## Goal

`include/mori/core/transport/rdma/providers/mlx5/mlx5_device_primitives.hpp` currently
`#include "infiniband/mlx5dv.h"`, which pulls the system `<infiniband/verbs.h>` into
every mlx5 device translation unit. Vendoring the small, stable subset of mlx5 WQE/CQE
hardware-ABI structs + constants that the device code actually uses lets the mlx5 device
header drop mlx5dv.h → device code becomes verbs-free (like bnxt/ionic already are).

## Context — what's already done (branch `jiahzhou/core-include-refactor`, base `main`)

- A broad core/application include-hygiene + host/device-separation refactor (commit
  `af9b880d`).
- `WcStatus` decoupling (commit `1091c7a9`): added a device-safe `mori::core::WcStatus`
  enum mirroring `ibv_wc_status`, so the CQE error decoders no longer return
  `ibv_wc_status`. Result (verified with `hipcc -M`): **bnxt and ionic device headers no
  longer pull `infiniband/verbs.h` at all**. Only mlx5 still does — via mlx5dv.h. This
  doc removes that last one.
- Pattern to follow: bnxt/ionic get their WQE/CQE ABI from repo-vendored firmware headers
  (`bnxt_re_hsi.h`, `ionic_fw.h`) that have **zero** verbs.h dependency. mlx5 should do the
  same with a vendored `mlx5_defs.hpp`.

## Why the naive attempt failed (already tried + reverted)

Vendoring the structs/constants into `mlx5_defs.hpp` under `namespace mori::core` **with
the same names** (`mlx5_cqe64`, `MLX5_OPCODE_SEND`, …) broke ~35 targets with
**ambiguity / type-mismatch errors**. Root cause: many TUs (host `mlx5.hpp`/`mlx5.cpp`,
examples like `dist_write.cpp`, `shmem_ibgda_kernels.hpp`) see **both** the system
`::mlx5_*` (pulled via the application RDMA headers → `mlx5_dv.h` → system mlx5dv) **and**
the vendored `mori::core::mlx5_*` (via the core device path), and reference those names
unqualified → ambiguous. e.g. `shmem_ibgda_kernels.hpp:314`:
`cannot initialize 'struct mlx5_err_cqe *' with an rvalue of type 'mlx5_err_cqe *'`.

So same-name vendoring is unsafe in this codebase because system and core mlx5 names are
intermixed across host/device/example TUs.

## Recommended approach: mori-prefixed names (no clash with system `::mlx5_*`)

Vendor the structs/constants with **distinct names** so they can never collide with the
system header, regardless of which TUs see both:

- structs: `mori::core::Mlx5WqeCtrlSeg`, `Mlx5WqeDataSeg`, `Mlx5WqeRaddrSeg`,
  `Mlx5WqeAtomicSeg`, `Mlx5WqeInlDataSeg`, `Mlx5ErrCqe`, `Mlx5Cqe64`
- constants: `MORI_MLX5_OPCODE_SEND`, `MORI_MLX5_CQE_SYNDROME_*`, etc. (or a scoped enum)

Then update **`mlx5_device_primitives.hpp`** (~40 use sites) to use the mori names and drop
`#include "infiniband/mlx5dv.h"` (keep `mlx5_defs.hpp`). Device TUs that still transitively
pull system mlx5dv are then harmless — different names, no ambiguity.

(Alternative, more invasive: ensure device TUs never include the application RDMA host
headers that drag system mlx5dv. Harder given current entanglement; the rename is simpler.)

## Exact inventory to vendor

### Structs (verbatim layout from `/usr/include/infiniband/mlx5dv.h`; `__beN` → `uintN_t`, the device code byte-swaps explicitly)

```cpp
// 16 bytes, packed/aligned(4)
struct Mlx5WqeCtrlSeg {            // ::mlx5_wqe_ctrl_seg
  uint32_t opmod_idx_opcode;
  uint32_t qpn_ds;
  uint8_t  signature;
  uint16_t dci_stream_channel_id;
  uint8_t  fm_ce_se;
  uint32_t imm;
} __attribute__((__packed__)) __attribute__((__aligned__(4)));

struct Mlx5WqeDataSeg {            // ::mlx5_wqe_data_seg  (16 B)
  uint32_t byte_count; uint32_t lkey; uint64_t addr;
};
struct Mlx5WqeRaddrSeg {           // ::mlx5_wqe_raddr_seg (16 B)
  uint64_t raddr; uint32_t rkey; uint32_t reserved;
};
struct Mlx5WqeAtomicSeg {          // ::mlx5_wqe_atomic_seg (16 B)
  uint64_t swap_add; uint64_t compare;
};
struct Mlx5WqeInlDataSeg {         // ::mlx5_wqe_inl_data_seg (4 B)
  uint32_t byte_count;
};
struct Mlx5ErrCqe {                // ::mlx5_err_cqe (64 B)
  uint8_t  rsvd0[32];
  uint32_t srqn;
  uint8_t  rsvd1[18];
  uint8_t  vendor_err_synd;
  uint8_t  syndrome;
  uint32_t s_wqe_opcode_qpn;
  uint16_t wqe_counter;
  uint8_t  signature;
  uint8_t  op_own;
};
struct Mlx5Cqe64 {                 // ::mlx5_cqe64 (64 B)
  // mlx5dv's leading union { anon hdr / mlx5_tm_cqe / ibv_tmh } is 32 B; the
  // device code never reads its members (only the trailer + wqe_counter/op_own),
  // so keep it opaque — avoids vendoring mlx5_tm_cqe / ibv_tmh.
  uint8_t  rsvd_hdr[32];
  uint32_t srqn_uidx;
  uint32_t imm_inval_pkey;
  uint8_t  app;
  uint8_t  app_op;
  uint16_t app_info;
  uint32_t byte_cnt;
  uint64_t timestamp;
  uint32_t sop_drop_qpn;
  uint16_t wqe_counter;
  uint8_t  signature;
  uint8_t  op_own;
};
```

### Constants (values from system mlx5dv.h)

```
MLX5_SEND_WQE_BB = 64,  MLX5_SEND_WQE_SHIFT = 6
MLX5_RCV_DBR = 0,  MLX5_SND_DBR = 1
MLX5_CQ_SET_CI = 0,  MLX5_CQ_ARM_DB = 1            (already in mlx5_defs.hpp today)
MLX5_WQE_CTRL_CQ_UPDATE = (2 << 2)
MLX5_INLINE_SEG = 0x80000000u
MLX5_CQE_OWNER_MASK = 1, MLX5_CQE_REQ_ERR = 13, MLX5_CQE_RESP_ERR = 14, MLX5_CQE_INVALID = 15
MLX5_OPCODE_RDMA_WRITE=0x08, SEND=0x0a, RDMA_READ=0x10,
              ATOMIC_CS=0x11, ATOMIC_FA=0x12, ATOMIC_MASKED_CS=0x14, ATOMIC_MASKED_FA=0x15
MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR=0x01, LOCAL_QP_OP_ERR=0x02, LOCAL_PROT_ERR=0x04,
  WR_FLUSH_ERR=0x05, MW_BIND_ERR=0x06, BAD_RESP_ERR=0x10, LOCAL_ACCESS_ERR=0x11,
  REMOTE_INVAL_REQ_ERR=0x12, REMOTE_ACCESS_ERR=0x13, REMOTE_OP_ERR=0x14,
  TRANSPORT_RETRY_EXC_ERR=0x15, RNR_RETRY_EXC_ERR=0x16, REMOTE_ABORTED_ERR=0x22
```

Note: `MLX5_POST_ATOMIC_SPEC` used in the device code is **mori's own** (not in mlx5dv.h) —
leave it as-is.

## Parity guard (mandatory — this is the safety net)

The structs overlay hardware memory, so the vendored layout MUST equal the system one.
Add `static_assert`s in a **dedicated host .cpp** that includes BOTH the system mlx5dv and
the vendored header — keep it isolated (no other mlx5 code) so the dual visibility causes
no ambiguity. Suggested new file
`src/application/transport/rdma/providers/mlx5/mlx5_abi_parity.cpp`:

```cpp
#include <cstddef>
#include <infiniband/mlx5dv.h>                                   // system ::mlx5_*
#include "mori/core/transport/rdma/providers/mlx5/mlx5_defs.hpp" // vendored mori::core::Mlx5*
#define SZ(v, s)  static_assert(sizeof(::mori::core::v) == sizeof(::s), "size drift " #v)
#define OFF(v, s, vm, sm) \
  static_assert(offsetof(::mori::core::v, vm) == offsetof(::s, sm), "offset drift " #v "::" #vm)
SZ(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg);
SZ(Mlx5WqeDataSeg, mlx5_wqe_data_seg);
SZ(Mlx5WqeRaddrSeg, mlx5_wqe_raddr_seg);
SZ(Mlx5WqeAtomicSeg, mlx5_wqe_atomic_seg);
SZ(Mlx5WqeInlDataSeg, mlx5_wqe_inl_data_seg);
SZ(Mlx5ErrCqe, mlx5_err_cqe);
SZ(Mlx5Cqe64, mlx5_cqe64);
OFF(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg, fm_ce_se, fm_ce_se);
OFF(Mlx5WqeCtrlSeg, mlx5_wqe_ctrl_seg, imm, imm);
OFF(Mlx5WqeDataSeg, mlx5_wqe_data_seg, addr, addr);
OFF(Mlx5ErrCqe, mlx5_err_cqe, syndrome, syndrome);
OFF(Mlx5Cqe64, mlx5_cqe64, byte_cnt, byte_cnt);
OFF(Mlx5Cqe64, mlx5_cqe64, wqe_counter, wqe_counter);
OFF(Mlx5Cqe64, mlx5_cqe64, op_own, op_own);
#undef SZ
#undef OFF
```
Wire it into `src/application/CMakeLists.txt` (`MORI_APP_SOURCES`). If any assert fails,
the vendored layout is wrong — fix before trusting it.

## Step-by-step

1. Put the mori-prefixed structs + constants into
   `include/mori/core/transport/rdma/providers/mlx5/mlx5_defs.hpp` (`namespace mori::core`,
   `#include <stdint.h>` only — device-safe).
2. In `mlx5_device_primitives.hpp`: remove `#include "infiniband/mlx5dv.h"`; rename the ~40
   `mlx5_*` / `MLX5_*` use sites to the mori-prefixed names. (`mlx5_err_cqe` in
   `Mlx5HandleErrorCqe` too.)
3. Add the parity .cpp above + CMake wiring.
4. Build everything. Confirm the parity asserts pass.
5. Verify mlx5 is now verbs-free:
   `echo '#include "mori/core/transport/rdma/providers/mlx5/mlx5_device_primitives.hpp"' \
    | hipcc -x hip -std=c++17 -Iinclude -D__HIP_PLATFORM_AMD__ --offload-arch=<gfx> -M - \
    | grep -c infiniband/verbs.h`  → expect **0**.

## Verification (on the MLX5 box — the whole point of the handoff)

Run the CI mlx5/AINIC suite (see `.github/workflows/ci.yml`, `intranode-test`):
- `BUILD_UMBP=OFF BUILD_EXAMPLES=ON BUILD_BENCHMARK=ON pip install .`
- `pytest tests/python/ops/test_dispatch_combine_intranode.py`
- `MORI_ENABLE_SDMA=1 python -m tests.python.ccl.test_allgather/test_all2all --world-size 8 …`
- shmem C++: `mpirun -np 2 ./build/examples/concurrent_put_thread` (+ `MORI_DISABLE_P2P=ON`
  for the IBGDA/RDMA path — this is the one that exercises the vendored mlx5 WQE/CQE code).
- `pytest tests/python/shmem/test_api.py`

A layout bug shows up as IBGDA correctness failures / hangs, NOT a compile error — so the
RDMA data-path tests above are the real gate.

## Reference: how bnxt/ionic already did the device verbs-decoupling

See commit `1091c7a9` (WcStatus) and the bnxt/ionic provider device headers — they use
repo-vendored firmware headers and the `mori::core::WcStatus` mirror, and pull no verbs.h.
The mlx5 work is the same idea, just also vendoring the WQE/CQE structs (which mlx5 gets
from the system mlx5dv.h instead of a repo firmware header).
