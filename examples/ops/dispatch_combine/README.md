
## Test under DeepSeek-V3/R1 configuration
Test:
```
make -j32 && mpirun -np 8 --allow-run-as-root ./examples/test_dispatch_combine_ops --data_type fp8 --hdim=7168 --max_tokens=128 --expert_per_rank=32 --expert_per_token=8 --warp_per_blk=4 --block_num=128 --num=1 --cmd test
```

Intra-node Benchmark:
```
make -j32 && mpirun -np 8 --allow-run-as-root rocprofv3 --kernel-trace --stats -o dispatch -- ./examples/test_dispatch_combine_ops --data_type bf16 --hdim=7168 --max_tokens=512 --expert_per_rank=32 --expert_per_token=8 --warp_per_blk=4 --block_num=256 --num=10 --cmd bench --kernel_type=intra
```

Inter-node Benchmark:

Run the following command on each node and replace node_rank to its actual rank. Note that 'master_addr' should be the ip of rank 0 node. Environment variable 'GLOO_SOCKET_IFNAME' should be set to the tcp socket ifname you want to use.

```
export GLOO_SOCKET_IFNAME=ens14np0
export MORI_RDMA_DEVICES=^mlx5_0,mlx5_1  # Optional: use `^` prefix to exclude specified devices

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.194.132.29" --master_port=1234 examples/ops/dispatch_combine/test_dispatch_combine_internode.py --max-tokens 128 --cmd bench
```


GLOO_SOCKET_IFNAME=enp81s0f1 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.235.192.87" --master_port=1234 examples/ops/dispatch_combine/test_dispatch_combine_internode.py --max-tokens 128 --cmd bench
GLOO_SOCKET_IFNAME=enp81s0f1 torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 --master_addr="10.235.192.87" --master_port=1234 examples/ops/dispatch_combine/test_dispatch_combine_internode.py --max-tokens 128 --cmd bench

The output of this scripit includes total number of tokens received, total number of RDMA tokens received and total bandwidth(include XGMI and RDMA). To calculate RDMA bandwidth, multiply the total bandwidth with (total # of RDMA tokens / total # of tokens);

## JAX MoE block example

[`moe_block_jax.py`](moe_block_jax.py) is a self-contained, heavily-commented example
of an expert-parallel Mixture-of-Experts (MoE) layer written in JAX on top of MORI's
`EpDispatchCombineOp`. It is the JAX analogue of the MORI-EP inference path used by
MaxText.

Each rank owns a contiguous block of experts. The flow on every rank is: top-k gate
routing per token, then `op.dispatch` scatters each token to the rank(s) owning its
selected experts, the per-expert SwiGLU MLP runs locally on the received tokens, and
`op.combine` sums the partial results back into the original source-token order. The
round-trip output is checked against a plain-JAX reference MoE.

MORI bakes the rank into the FFI config as a static attribute, so the example needs one
GPU per process; it spawns one worker process per GPU, so just run it directly:

```
MORI_NUM_PROCS=8 MORI_TOTAL_GPUS=8 python examples/ops/dispatch_combine/moe_block_jax.py
```

Knobs such as `--hidden_dim`, `--inter_dim`, `--num_experts_per_rank`,
`--num_experts_per_token` and `--num_tokens` are available via `--help`.

Note: the expert weights are replicated on every rank only so the reference MoE can be
computed locally without collectives. A real model would shard the experts across ranks
and keep only its own block, which is exactly why dispatch/combine moves tokens to where
their experts live.

## Mori Environment Variables

### MORI_RDMA_DEVICES

For RoCE networks, you can specify which RDMA devices to use with the `MORI_RDMA_DEVICES` environment variable:

- **Include specific devices**: `MORI_RDMA_DEVICES=mlx5_0,mlx5_1`
- **Exclude devices**: `MORI_RDMA_DEVICES=^mlx5_2,mlx5_3` (use `^` prefix to exclude specified devices)
- **Default**: If not set, all available RDMA devices will be used
