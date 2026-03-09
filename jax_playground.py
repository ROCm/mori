import jax
import jax.numpy as jnp
import mori
from jax.sharding import PartitionSpec as P
# from jax.experimental.pjit import pjit
from jax._src.lib import _jax
import numpy as np
import argparse, os, time, functools
import gc

def get_distributed_client(): # -> _jax.DistributedRuntimeClient:
  from jax._src.distributed import global_state
  assert isinstance(global_state.client, _jax.DistributedRuntimeClient)
  return global_state.client
  
def mori_shmem_init_attr(rank, world_size, sync_name="mori/unique_id", 
            timeout_ms=5_000):
  
  client = get_distributed_client()
  if rank == 0:
    unique_id = mori.shmem.shmem_get_unique_id()
    client.key_value_set_bytes(
       sync_name, unique_id
       #devs.key, pickle.dumps(nccl_id)
    )
  else:
    unique_id = client.blocking_key_value_get_bytes(
      sync_name, timeout_ms)
  
  #print(f"{rank} unique ID initattr {unique_id}", flush=True)
  mori.shmem.shmem_init_attr(mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
                   rank, world_size, unique_id)
  print(f"{rank} unique ID initattr OK", flush=True)
  
def setup(rank, world_size):
  return mori_shmem_init_attr(rank, world_size)

def cleanup():
  jax.clear_caches() # this is needed since EpDispatchCombineState can be
                     # destroyed later on
  gc.collect()
  mori.shmem.shmem_finalize()
  
def run_test(rank, world_size):
  try:
    jax.ffi.register_ffi_target("mori_ep", mori.cpp.mori_ep_handler(), platform="ROCM")
    jax.ffi.register_ffi_type_id("mori_ep", mori.cpp.mori_ep_type_id(), platform="ROCM")
  except Exception:
    # Already registered in this process.
    pass
  print(f"[rank {rank}] devices = {jax.local_devices()}", flush=True)

  cfg = mori.cpp.EpDispatchCombineConfig(
      rank=rank,
      world_size=world_size,
      hidden_dim=4096,
      scale_dim=32,
      scale_type_size=1,
      max_token_type_size=4,
      max_num_inp_token_per_rank=128,
      num_experts_per_rank=1,
      num_experts_per_token=2,
      warp_num_per_block=1,
      block_num=1,
      use_external_inp_buf=True,
      kernel_type=mori.cpp.EpDispatchCombineKernelType.IntraNode,
      gpu_per_node=8,
      rdma_block_num=1,
      num_qp_per_pe=1,
      quant_type=mori.cpp.EpDispatchCombineQuantType.None_,
  )

  launch_test = jax.ffi.ffi_call("mori_ep", (), has_side_effect=True)
  launch_test(ep_config=np.asarray(cfg.to_packed_array(), dtype=np.int32),
              reset_op=True)
  jax.block_until_ready(jnp.array(0))
  print(f"[rank {rank}] launch_test submitted", flush=True)

def init_distributed(): 
  parser = argparse.ArgumentParser()
  parser.add_argument("--world_size", type=int, default=1)
  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--coordination_service", type=str, default="")
  args, _ = parser.parse_known_args()
  
  jax.distributed.initialize(
      coordinator_address=args.coordination_service or "localhost:12341",
      num_processes=args.world_size,
      process_id=args.rank)
  print(f"[rank {args.rank}] devices = {jax.local_devices()}", flush=True)
  return (args.rank, args.world_size)

if __name__ == "__main__":
    rank, world_size = init_distributed()
    setup(rank, world_size)
    run_test(rank, world_size)
    cleanup()
