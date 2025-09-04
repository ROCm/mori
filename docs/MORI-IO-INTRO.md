# MORI-IO Introduction

MORI-IO is AMD's point-to-point communication library that leverages GDR (GPU Direct RDMA) to achieve low-latency and high-bandwidth. Its current main use case is KVCache transfer in LLM inference.

## Concepts
- **IOEgnine**: The primary interface for interacting with MORI-IO, it abstracts low-level details of p2p communications and provides high-level APIs for engine registration, memory registration, p2p transfer and etc.
- **Backend**: A backend represents and manages a specific transfer medium (e.g., PCIe, xGMI, IB). It must be created before any data transfer can occur over that medium.
- **Engine Registration**: Before two engines can communicate, the remote engine must be registered with the local engine. This establishes the necessary context for initiating data transfers between them.
- **Memory Registration**: Application memory must be registered with a local engine before it can participate in data transfer. This ensures the engine can access and manage the memory efficiently during communication.
- **Read/Write**: One-sided transfer operations initiated by the initiator engine without active involvement from the target engine. These operations can move data directly between registered memory regions.
- **Batch Read/Write**: A batched form of one-sided operations, where multiple transfers are grouped and launched together. Batching reduces per-operation launch overhead and improves bandwidth utilization.
- **Session**: A pre-established transfer context between a pair of MemoryDesc objects. Sessions eliminate repetitive overheads such as connection setup, metadata exchange, and resource management, providing a lightweight and efficient path for repeated transfers.

## Workflow

## Architecture
