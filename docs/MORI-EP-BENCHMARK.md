# MORI-EP Benchmark

## Intra-node
```
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH

# Benchmark performance
python3 tests/python/ops/bench_dispatch_combine.py
```

## Inter-node

Run the following command on each node and replace node_rank to its actual rank. Note that 'master_addr' should be the ip of rank 0 node. Environment variable 'GLOO_SOCKET_IFNAME' should be set to the tcp socket ifname you want to use.

```
export GLOO_SOCKET_IFNAME=ens14np0
export MORI_RDMA_DEVICES=^mlx5_0,mlx5_1  # Optional: use `^` prefix to exclude specified devices

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.194.129.65" --master_port=1234 examples/ops/dispatch_combine/test_dispatch_combine_internode.py --bench
```

The output of this scripit includes total number of tokens received, total number of RDMA tokens received and total bandwidth(include XGMI and RDMA). To calculate RDMA bandwidth, multiply the total bandwidth with (total # of RDMA tokens / total # of tokens);

## Others

### Select NICs by setting environment variable MORI_RDMA_DEVICES

For RoCE networks, you can specify which RDMA devices to use with the `MORI_RDMA_DEVICES` environment variable:

- **Include specific devices**: `MORI_RDMA_DEVICES=mlx5_0,mlx5_1`
- **Exclude devices**: `MORI_RDMA_DEVICES=^mlx5_2,mlx5_3` (use `^` prefix to exclude specified devices)
- **Default**: If not set, all available RDMA devices will be used
