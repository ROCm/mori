


## Test under DeepSeek-V3/R1 configuration
```
make -j8 && mpirun -np 8 --allow-run-as-root ./examples/test_dispatch_combine --data_type fp8 --hdim=7168 --max_tokens=128 --expert_per_rank=32 --expert_per_token=8 --warp_per_blk=4 --block_num=128 --num=1 --cmd test
```

make -j8 && mpirun -np 8 --allow-run-as-root rocprofv3 --kernel-trace --stats -- ./examples/test_dispatch_combine --data_type fp8 --hdim=7168 --max_tokens=128 --expert_per_rank=32 --expert_per_token=4 --warp_per_blk=10 --block_num=64 --num=1 --cmd bench