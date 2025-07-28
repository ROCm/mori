


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
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.194.129.65" --master_port=1234 examples/ops/dispatch_combine/test_dispatch_combine_internode.py --bench
```

The output of this scripit includes total number of tokens received, total number of RDMA tokens received and total bandwidth(include XGMI and RDMA). To calculate RDMA bandwidth, multiply the total bandwidth with (total # of RDMA tokens / total # of tokens);