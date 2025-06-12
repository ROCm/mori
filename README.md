# mori

MORI stands for Modular RDMA Interface

## Installation

### Prerequsites

- pytorch:rocm >= 6.3.0
- Linux packages
    
    ```apt-get install -y git cython3 ibverbs-utils openmpi-bin libopenmpi-dev cmake libdw1```

Or build docker image with:
```
cd mori && docker build -t rocm/mori:dev -f docker/Dockerfile.dev .
```

### Install with Python
```
cd mori && pip3 install .
```

### Test dispatch / combine
```
python mori/examples/ops/dispatch_combine/test_dispatch_combine.py 
```