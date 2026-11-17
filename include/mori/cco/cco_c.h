// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// MORI-CCO host C ABI.
//
// A plain C99 interface to the cco host control plane, for code that loads
// libmori_cco at run time instead of linking its C++ API. The ABI is one exported
// symbol, moriCcoGetApi, which fills a function table:
//
//   void* lib = dlopen(MORI_CCO_LIBRARY_SONAME, RTLD_NOW | RTLD_LOCAL);
//   moriCcoGetApiFn_t get = (moriCcoGetApiFn_t)dlsym(lib, MORI_CCO_GET_API_SYMBOL);
//   moriCcoApi_t api;
//   if (get(MORI_CCO_ABI_MAJOR, &api, sizeof(api)) != MORI_CCO_SUCCESS) { ... }
//
// Versioning. A major bump breaks compatibility and changes the SONAME;
// moriCcoGetApi refuses a major it does not implement. A minor bump only appends:
// entries at the end of moriCcoApi_t, fields at the end of the size-prefixed
// structs. A caller built against an older minor passes its smaller sizes and gets
// what it knows. A newer caller on an older library sees api.abiMinor, finds the
// entries the library lacks set to NULL, and gets MORI_CCO_ERROR_NOT_SUPPORTED if
// it sets a struct field the library does not know to anything but 0.
//
// Errors. Every entry except getLastError returns a MORI_CCO_* code, and on
// failure getLastError() says why. No entry ends the process on an error it can
// report. A collective that fails on some ranks can leave the others blocked in
// it; after a collective error, destroy the comm on every rank.
//
// Threading. All entries are thread-safe, and calls on one comm are serialised.
// Collective entries (commCreate, windowImport, devCommCreate) must be called by
// every rank of the comm in the same order, so issue them from one thread per
// comm. The other entries are local. Each entry runs on the comm's HIP device and
// restores the caller's current device before it returns. Entries make
// synchronous HIP calls, so none may be called during stream capture.
//
// Lifetime. Release windows and dev comms before their comm; commDestroy releases
// any that remain. Nothing may be released while GPU work that uses it is in
// flight. A dev comm's window table is fixed when it is created, so release dev
// comms that might look a window up before releasing that window. A window holds
// its own reference to the memory it imports, so the owner's unmap and release of
// its VMM handle and windowRelease may come in either order. A forked child must
// not use handles created by its parent.
#ifndef MORI_CCO_CCO_C_H_
#define MORI_CCO_CCO_C_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MORI_CCO_ABI_MAJOR 1
#define MORI_CCO_ABI_MINOR 0

// SONAME of the library that implements this ABI major, and its entry symbol.
#define MORI_CCO_LIBRARY_SONAME "libmori_cco.so.1"
#define MORI_CCO_GET_API_SYMBOL "moriCcoGetApi"

// Return codes.
enum {
  MORI_CCO_SUCCESS = 0,
  MORI_CCO_ERROR_INVALID_ARGUMENT = 1,  // bad handle, pointer, size or flag
  MORI_CCO_ERROR_ABI_MISMATCH = 2,      // ABI major not implemented
  MORI_CCO_ERROR_NOT_SUPPORTED = 3,     // valid request this library cannot serve
  MORI_CCO_ERROR_HIP = 4,               // HIP runtime call failed
  MORI_CCO_ERROR_INTERNAL = 5           // cco failed; see getLastError()
};

// Opaque handles. They are not pointers: never dereference them.
typedef struct moriCcoComm_st* moriCcoComm_t;
typedef struct moriCcoWindow_st* moriCcoWindow_t;

// Rank 0's rendezvous address, passed to every rank out of band.
#define MORI_CCO_UNIQUE_ID_BYTES 128
typedef struct moriCcoUniqueId {
  char internal[MORI_CCO_UNIQUE_ID_BYTES];
} moriCcoUniqueId_t;

// commCreate flags. They must match on all ranks.
// Leave the calling thread's CPU affinity alone. Without it commCreate pins the
// thread to its GPU's NUMA-local cores, and threads it starts later inherit that.
#define MORI_CCO_COMM_NO_CPU_AFFINITY ((uint64_t)1 << 0)
// Intra-node (LSA) only: open no RDMA device and skip GPU-NIC topology. Peers
// outside the caller's LSA team are unreachable (windowPeerPtr gives NULL).
// Required: ABI 1.0 has no RDMA, so commCreate refuses a config without it with
// MORI_CCO_ERROR_NOT_SUPPORTED.
#define MORI_CCO_COMM_LSA_ONLY ((uint64_t)1 << 1)

// Reserved for a caller-provided bootstrap (e.g. a c10d Store); a later minor
// defines it.
struct moriCcoBootstrap;

typedef struct moriCcoCommConfig {
  size_t size;            // sizeof(moriCcoCommConfig_t)
  uint64_t flags;         // MORI_CCO_COMM_*
  size_t perRankVaBytes;  // flat VA per rank, rounded up to 4 GiB; 0 = device memory size
  const struct moriCcoBootstrap* bootstrap;  // reserved: must be NULL
  int device;    // HIP device ordinal; -1 = the calling thread's current device
  int reserved;  // must be 0
} moriCcoCommConfig_t;

// What MORI_CCO_COMM_CONFIG_INIT and a NULL config use.
#define MORI_CCO_COMM_DEFAULT_FLAGS (MORI_CCO_COMM_LSA_ONLY | MORI_CCO_COMM_NO_CPU_AFFINITY)
#define MORI_CCO_COMM_CONFIG_INIT \
  {sizeof(moriCcoCommConfig_t), MORI_CCO_COMM_DEFAULT_FLAGS, 0, NULL, -1, 0}

typedef struct moriCcoCommInfo {
  size_t size;  // in: sizeof(moriCcoCommInfo_t)
  int rank;
  int nRanks;
  int lsaRank;            // index in the LSA team
  int lsaSize;            // ranks reachable by load/store: same node, or same vPOD
  int lsaFirstRank;       // the LSA team is ranks [lsaFirstRank, lsaFirstRank + lsaSize)
  int device;             // HIP device the comm runs on
  int memHandleType;      // hipMemAllocationHandleType windowImport exports: 1 POSIX fd, 8 fabric
  size_t perRankVaBytes;  // flat-VA distance between LSA peers' copies of a window
  uint64_t flags;         // flags the comm was created with
} moriCcoCommInfo_t;

typedef struct moriCcoWindowInfo {
  size_t size;         // in: sizeof(moriCcoWindowInfo_t)
  void* localPtr;      // this rank's flat-VA alias of the buffer
  void* deviceWindow;  // mori::cco::ccoWindow_t, for cco device code
  size_t bytes;        // bytes registered
} moriCcoWindowInfo_t;

typedef struct moriCcoDevCommRequirements {
  size_t size;          // sizeof(moriCcoDevCommRequirements_t)
  int lsaBarrierCount;  // LSA barriers (ccoLsaBarrierSession)
  int sdmaQueueCount;   // SDMA queues per peer; 0 = default. Needs MORI_ENABLE_SDMA=1
} moriCcoDevCommRequirements_t;

#define MORI_CCO_DEV_COMM_REQUIREMENTS_INIT {sizeof(moriCcoDevCommRequirements_t), 0, 0}

typedef struct moriCcoDevCommInfo {
  size_t size;             // in: sizeof(moriCcoDevCommInfo_t)
  size_t devCommBytes;     // sizeof(mori::cco::ccoDevComm) in this library
  uint32_t layoutVersion;  // mori::cco::CCO_DEV_COMM_LAYOUT_VERSION in this library
  int lsaBarrierCount;     // LSA barriers created
  int sdmaQueueCount;      // SDMA queues per peer created; 0 = no SDMA
} moriCcoDevCommInfo_t;

typedef struct moriCcoApi {
  size_t size;   // out: bytes the library filled
  int abiMajor;  // out
  int abiMinor;  // out

  // ---- ABI 1.0 ----

  // Any argument may be NULL. libraryVersion is static.
  int (*getVersion)(int* abiMajor, int* abiMinor, const char** libraryVersion);

  // This thread's last error; "" after a successful call. Valid until the
  // thread's next call into this ABI. Never NULL.
  const char* (*getLastError)(void);

  // Call on rank 0 only. Uses MORI_SOCKET_IFNAME, else the first non-loopback NIC.
  int (*getUniqueId)(moriCcoUniqueId_t* uniqueId);

  // Collective. A NULL config means MORI_CCO_COMM_CONFIG_INIT.
  int (*commCreate)(const moriCcoUniqueId_t* uniqueId, int nRanks, int rank,
                    const moriCcoCommConfig_t* config, moriCcoComm_t* comm);
  // Local. Also releases the comm's remaining windows and dev comms.
  int (*commDestroy)(moriCcoComm_t comm);
  int (*commGetInfo)(moriCcoComm_t comm, moriCcoCommInfo_t* info);

  // Collective. Registers [ptr, ptr + bytes) of an existing HIP VMM allocation
  // (hipMemCreate + hipMemMap) as a symmetric window, without copying. Every rank
  // passes the same bytes and the same offset of ptr into its allocation, which
  // must be created with requestedHandleType = commGetInfo's memHandleType.
  // flags must be 0. info may be NULL.
  int (*windowImport)(moriCcoComm_t comm, void* ptr, size_t bytes, uint64_t flags,
                      moriCcoWindow_t* window, moriCcoWindowInfo_t* info);
  // Local; ranks may release in any order.
  int (*windowRelease)(moriCcoWindow_t window);
  // Rank peer's copy of the window as a device address valid on this rank, or
  // NULL if peer is outside the LSA team. peer == own rank gives localPtr.
  int (*windowPeerPtr)(moriCcoWindow_t window, int peer, void** ptr);

  // Collective. Returns, in *devComm, device memory holding a
  // mori::cco::ccoDevComm for kernels. requirements may be NULL for defaults.
  int (*devCommCreate)(moriCcoComm_t comm, const moriCcoDevCommRequirements_t* requirements,
                       void** devComm);
  // Local.
  int (*devCommDestroy)(moriCcoComm_t comm, void* devComm);
  // Layout handshake for device code built against cco.hpp, and what was created.
  int (*devCommGetInfo)(moriCcoComm_t comm, const void* devComm, moriCcoDevCommInfo_t* info);
} moriCcoApi_t;

// Fills the first tableSize bytes of *table with the entries of ABI major
// abiMajor. Thread-safe. Returns MORI_CCO_ERROR_ABI_MISMATCH for an unknown
// major, and MORI_CCO_ERROR_INVALID_ARGUMENT for a table smaller than ABI 1.0's.
int moriCcoGetApi(int abiMajor, moriCcoApi_t* table, size_t tableSize);
typedef int (*moriCcoGetApiFn_t)(int abiMajor, moriCcoApi_t* table, size_t tableSize);

#ifdef __cplusplus
}
#endif

#endif  // MORI_CCO_CCO_C_H_
