import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
# from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.experimental import multihost_utils
from jax._src import xla_bridge as xb
import numpy as np
import argparse, os, time, functools


def init_distributed():
    
  parser = argparse.ArgumentParser()
  parser.add_argument("--world_size", type=int, default=1)
  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--coordination_service", type=str, default="")
  args, _ = parser.parse_known_args()
  
  print(f"{args.rank} --- start")

  jax.distributed.initialize(
      coordinator_address=args.coordination_service or "localhost:12341",
      num_processes=args.world_size,
      process_id=args.rank)
  print(f"[rank {args.rank}] devices = {jax.local_devices()}", flush=True)
  
#   Z = xb.backends()["rocm"]
#   print(f"xla backend {Z} -- {dir(Z)}")
  
  unique_id = np.full((100,),args.rank+1)
#   if args.rank == 0:
  print(f"XX {args.rank} --- id: {unique_id}", flush=True)
#   else:
#     unique_id = np.arange(100)

  unique_id = multihost_utils.broadcast_one_to_all(unique_id)
  
  print(f"{args.rank} --- id: {unique_id}")
  
  return (args.rank, args.world_size)

if __name__ == "__main__":
    print(f" --- start", flush=True)
    world_config = init_distributed()
