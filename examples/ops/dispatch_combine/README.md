


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
```
pip3 install .. && torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.194.128.135" --master_port=1234 ../examples/ops/dispatch_combine/test_dispatch_combine_internode.py
```