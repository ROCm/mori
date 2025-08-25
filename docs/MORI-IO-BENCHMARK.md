# MORI-IO Benchmark

## Benchmark Commands
```
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH
export GLOO_SOCKET_IFNAME=ens14np0 # set to your nic interface

# Benchmark performance, run the following command on two nodes
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.194.129.65" --master_port=1234 tests/python/io/benchmark.py --host="10.194.129.65"
```

Arguments:
- **buffer-size:** message size per transfer
- **all**: sweep from 8B to 1MB message size
- **transfer-batch-size**: # of consecutive transfers
- **enable-batch-transfer**: whether to enable batch transfer
- **enable-sess**: whether to enable session transfer, session transfer has lower latency
- **num-initiator-dev**: # of initiator devices
- **num-target-dev**: # of target devices
- **num-qp-per-transfer**: # of queue pair used

## Results

### Thor2 Result (RDMA read)
Command:
```
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 --master_addr="10.235.192.57" --master_port=1234 tests/python/io/benchmark.py --host="10.235.192.60" --all --enable-sess --enable-batch-transfer --num-qp-per-transfer 2 --num-target-dev 8 --num-initiator-dev 8
```

Output:
```
MORI-IO Benchmark Configurations:
  host: 10.235.192.57
  port: 38553
  node_rank: 0
  role: EngineRole.INITIATOR
  role_rank: 4
  num_initiator_dev: 8
  num_target_dev: 8
  buffer_size: 1048576 B
  transfer_batch_size: 256
  enable_batch_transfer: True
  enable_sess: True
  num_qp_per_transfer: 2
+--------------------------------------------------------------------------------------------+
|                                      Initiator Rank 7                                      |
+-------------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+----------------+---------------+---------------+--------------+--------------+
|      8      |      0.00      |      0.02     |      0.02     |    113.73    |    120.65    |
|      16     |      0.00      |      0.04     |      0.03     |    113.96    |    118.57    |
|      32     |      0.01      |      0.07     |      0.07     |    113.49    |    118.69    |
|      64     |      0.02      |      0.14     |      0.14     |    114.44    |    118.73    |
|     128     |      0.03      |      0.29     |      0.27     |    114.68    |    119.34    |
|     256     |      0.07      |      0.57     |      0.55     |    114.44    |    118.67    |
|     512     |      0.13      |      1.15     |      1.11     |    113.49    |    118.18    |
|     1024    |      0.26      |      2.30     |      2.23     |    114.20    |    117.78    |
|     2048    |      0.52      |      4.47     |      4.31     |    117.30    |    121.52    |
|     4096    |      1.05      |      8.31     |      8.01     |    126.12    |    130.95    |
|     8192    |      2.10      |     14.19     |     13.77     |    147.82    |    152.34    |
|    16384    |      4.19      |     22.16     |     21.56     |    189.30    |    194.58    |
|    32768    |      8.39      |     30.44     |     29.84     |    275.61    |    281.09    |
|    65536    |     16.78      |     37.51     |     36.55     |    447.27    |    458.96    |
|    131072   |     33.55      |     42.93     |     41.65     |    781.54    |    805.60    |
|    262144   |     67.11      |     45.66     |     45.08     |   1469.85    |   1488.69    |
|    524288   |     134.22     |     46.99     |     46.81     |   2856.02    |   2867.27    |
|   1048576   |     268.44     |     47.88     |     47.75     |   5605.94    |   5622.15    |
+-------------+----------------+---------------+---------------+--------------+--------------+
```
