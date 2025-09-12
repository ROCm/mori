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
- **all-batch**: sweep batch size from 1 to 32768
- **transfer-batch-size**: # of consecutive transfers
- **enable-batch-transfer**: whether to enable batch transfer
- **enable-sess**: whether to enable session transfer, session transfer has lower latency
- **num-initiator-dev**: # of initiator devices
- **num-target-dev**: # of target devices
- **num-qp-per-transfer**: # of queue pair used
- **poll_cq_mode**: mode of polling cqe, ['polling', 'event']

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

### Thor2 Result (RDMA Write) at 9.12
#### Performance of sweeping message size
```
numactl --cpunodebind=0 --membind=0 --physcpubind=0-47,96-143 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.235.192.60" --master_port=1234 tests/python/io/benchmark.py --host="10.235.192.60" --enable-batch-transfer --enable-sess --buffer-size 1024 --transfer-batch-size 128 --num-initiator-dev 1 --num-target-dev 1 --num-qp-per-transfer 4 --all --num-worker-threads 1 --log-level info --op-type write --poll_cq_mode polling
```

```
+--------------------------------------------------------------------------------------------------------+
|                                            Initiator Rank 0                                            |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
|      8      |    128    |      0.00      |      0.03     |      0.03     |    33.38     |    36.33     |
|      16     |    128    |      0.00      |      0.06     |      0.06     |    34.09     |    36.35     |
|      32     |    128    |      0.00      |      0.12     |      0.11     |    34.57     |    36.33     |
|      64     |    128    |      0.01      |      0.24     |      0.23     |    33.62     |    36.33     |
|     128     |    128    |      0.02      |      0.49     |      0.45     |    33.62     |    36.49     |
|     256     |    128    |      0.03      |      0.94     |      0.89     |    34.81     |    36.99     |
|     512     |    128    |      0.07      |      1.86     |      1.77     |    35.29     |    37.01     |
|     1024    |    128    |      0.13      |      3.84     |      3.53     |    34.09     |    37.09     |
|     2048    |    128    |      0.26      |      7.33     |      6.96     |    35.76     |    37.65     |
|     4096    |    128    |      0.52      |     12.94     |     12.46     |    40.53     |    42.09     |
|     8192    |    128    |      1.05      |     20.75     |     20.12     |    50.54     |    52.11     |
|    16384    |    128    |      2.10      |     29.03     |     28.33     |    72.24     |    74.02     |
|    32768    |    128    |      4.19      |     36.50     |     35.91     |    114.92    |    116.81    |
|    65536    |    128    |      8.39      |     41.74     |     41.39     |    200.99    |    202.70    |
|    131072   |    128    |     16.78      |     45.14     |     44.85     |    371.69    |    374.10    |
|    262144   |    128    |     33.55      |     46.93     |     46.76     |    715.02    |    717.56    |
|    524288   |    128    |     67.11      |     47.94     |     47.81     |   1399.99    |   1403.64    |
|   1048576   |    128    |     134.22     |     48.44     |     48.32     |   2770.90    |   2777.76    |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
```

#### Performance of sweeping batch size
```
numactl --cpunodebind=0 --membind=0 --physcpubind=0-47,96-143 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr="10.235.192.60" --master_port=1234 tests/python/io/benchmark.py --host="10.235.192.60" --enable-batch-transfer --enable-sess --buffer-size 1024 --transfer-batch-size 128 --num-initiator-dev 1 --num-target-dev 1 --num-qp-per-transfer 16 --all-batch --num-worker-threads 1 --log-level info --op-type write --poll_cq_mode pol
ling
```

```
+--------------------------------------------------------------------------------------------------------+
|                                            Initiator Rank 0                                            |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
| MsgSize (B) | BatchSize | TotalSize (MB) | Max BW (GB/s) | Avg Bw (GB/s) | Min Lat (us) | Avg Lat (us) |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
|     1024    |     1     |      0.00      |      0.10     |      0.09     |    10.25     |    11.10     |
|     1024    |     2     |      0.00      |      0.19     |      0.18     |    10.73     |    11.33     |
|     1024    |     4     |      0.00      |      0.34     |      0.33     |    12.16     |    12.60     |
|     1024    |     8     |      0.01      |      0.58     |      0.54     |    14.07     |    15.07     |
|     1024    |     16    |      0.02      |      0.83     |      0.72     |    19.79     |    22.87     |
|     1024    |     32    |      0.03      |      1.48     |      1.30     |    22.17     |    25.19     |
|     1024    |     64    |      0.07      |      2.45     |      2.14     |    26.70     |    30.61     |
|     1024    |    128    |      0.13      |      3.39     |      3.06     |    38.62     |    42.87     |
|     1024    |    256    |      0.26      |      4.43     |      4.19     |    59.13     |    62.53     |
|     1024    |    512    |      0.52      |      5.08     |      4.84     |    103.24    |    108.37    |
|     1024    |    1024   |      1.05      |      5.79     |      5.43     |    181.20    |    192.99    |
|     1024    |    2048   |      2.10      |      6.47     |      6.24     |    324.01    |    336.00    |
|     1024    |    4096   |      4.19      |      7.00     |      6.85     |    599.15    |    611.94    |
|     1024    |    8192   |      8.39      |      7.15     |      6.76     |   1173.02    |   1241.83    |
|     1024    |   16384   |     16.78      |      7.29     |      7.02     |   2301.69    |   2390.62    |
|     1024    |   32768   |     33.55      |      7.32     |      7.27     |   4585.50    |   4617.83    |
+-------------+-----------+----------------+---------------+---------------+--------------+--------------+
```
