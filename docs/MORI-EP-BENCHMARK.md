# MORI-EP Benchmark

## Table of Contents

- [Intra-node](#intra-node)
- [Inter-node](#inter-node)
- [NIC Selection](#nic-selection)

## Intra-node

```bash
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH

# Benchmark performance
python3 tests/python/ops/bench_dispatch_combine.py
```

## Inter-node

Run the following command on each node and replace `node_rank` with its actual rank. `master_addr` should be the IP of the rank 0 node. `GLOO_SOCKET_IFNAME` should be set to the TCP socket interface you want to use.

```bash
export GLOO_SOCKET_IFNAME=ens14np0
export MORI_RDMA_DEVICES=^mlx5_0,mlx5_1  # Optional: use `^` prefix to exclude specified devices

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --master_addr="10.194.129.65" --master_port=1234 \
    examples/ops/dispatch_combine/test_dispatch_combine_internode.py --bench
```

The output includes total number of tokens received, total number of RDMA tokens received, and total bandwidth (XGMI + RDMA combined). To calculate RDMA-only bandwidth:

```
RDMA BW = Total BW × (RDMA tokens / Total tokens)
```

## NIC Selection

For RoCE networks, you can specify which RDMA devices to use with the `MORI_RDMA_DEVICES` environment variable:

- **Include specific devices**: `MORI_RDMA_DEVICES=mlx5_0,mlx5_1`
- **Exclude devices**: `MORI_RDMA_DEVICES=^mlx5_2,mlx5_3` (use `^` prefix to exclude specified devices)
- **Default**: If not set, all available RDMA devices will be used
