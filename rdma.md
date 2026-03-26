# Mori RDMA Subsystem — AINIC (Ionic) Implementation

## 1. Overview

Mori's RDMA subsystem enables **GPU-initiated network I/O** — the GPU directly constructs and submits RDMA work requests without CPU involvement. This is the foundation for Mori-SHMEM `put`/`get`/`atomic` operations in GPU kernels.

Standard libibverbs requires CPU to post work requests. IBGDA (InfiniBand GPU Direct Async) bypasses this by exposing the NIC's hardware queues (SQ, RQ, CQ, Doorbell registers) directly to GPU memory space, allowing HIP kernels to construct WQEs and ring doorbells.

## 2. RDMA Class Hierarchy
See this pic ![rdma](./image.png)

```
struct RdmaStates // top-level structure of rdma subsystem
  |── application::Context* 


Context // RDMA subsystem 
  |── std::unique_ptr<RdmaContext>
  |── std::unique_ptr<RdmaDeviceContext>
  |── std::vector<RdmaEndpoint>
  |── std::unique_ptr<TopoSystem>


RdmaContext                       // top-level, enumerates all NICs 
  |──std::vector<RdmaDevice*>   // one per RNIC

/* RdmaDevice holds the identity and capabilities of a physical 
 * NIC (which device, how many ports, attributes of each port)
 * it describes the RDMA hardware but without QPs' abstractions
 */
RdmaDevice                          // single NIC
  ├── ibv_device*
  ├── ibv_context*
  ├── map<uint32_t, std::unique_ptr<ibv_port_attr>> // record attributes of each port
  └── std::unique_ptr<ibv_device_attr_ex> // device attribute

RdmaDeviceContext  //  Base class for all RDMA NIC providers (Ionic, MLX5, BNXT)
  ├── ibv_pd*
  ├── ibv_srq*
  ├── RdmaDevice* 
  └── map<void*, ibv_mr*>           // MR pool


/* Holds all QPs and CQs all Pesando RDMA NIC
 */
IonicDeviceContext : RdmaDeviceContext
  ├── std::unordered_map<uint32_t, IonicCqContainer*>            
  └── std::unordered_map<uint32_t, IonicQpContainer*>


IonicCqContainer     
  ├── uint32_t cqn
  ├── void* cqDbrUmemAddr
  ├── void* cqUmemAddr
  ├── void* cqUmem
  ├── void* cqDbrUmem
  ├── void* cqUar
  ├── void* cqUarPtr
  └── ibv_cq* cq              


/* List the part of members of this class
 */ 
IonicQpContainer
  ├── size_t qpn
  ├── uint16_t wqeNum
  ├── uint64_t* gpu_db_cq
  ├── uint64_t* gpu_db_sq
  ├── uint64_t* gpu_db_rq
  ├── size_t atomicIbufSize
  ├── void* atomicIbufAddr
  └── ibv_mr* atomicIbufMr atomicIbufAddr   // atomic result buffer


RdmaEndpoint // Pure POD data structure, can be copied with hipMemcy                  
  ├── core::WorkQueueHandle wqHandle               // SQ/RQ addrs, doorbell, tracking indices
  ├── core::CompletionQueueHandle cqHandle         // CQ addr, consumer index, doorbell
  ├── core::IBVerbsHandle ibvHandle;                // ibv_qp*/ibv_cq* (host fallback)
  └── core::IbufHandle atomicIbuf;                // atomic buffer addr + lkey
```