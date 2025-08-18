# mori

MORI stands for Modular RDMA Interface

## Installation

### Prerequsites

- pytorch:rocm >= 6.4.0
- Linux packages
    
    ```apt-get install -y git cython3 ibverbs-utils openmpi-bin libopenmpi-dev cmake libdw1```

Or build docker image with:
```
cd mori && docker build -t rocm/mori:dev -f docker/Dockerfile.dev .
```

### Install with Python
```
cd mori && git submodule update --init --recursive && pip3 install .
```

### Test dispatch / combine
```
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH

# Test correctness
pytest tests/python/ops/

# Benchmark performance
python3 tests/python/ops/bench_dispatch_combine.py 
```

### Test MORI-IO
```
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH

# Test correctness
pytest tests/python/io/

# Benchmark performance
export GLOO_SOCKET_IFNAME=ens14np0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.194.129.65" --master_port=1234 tests/python/io/benchmark.py --host="10.194.129.65" --enable-batch-transfer --enable-sess --buffer-size 32768 --transfer-batch-size 128
```