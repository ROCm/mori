#!/usr/bin/env python3
# Probe: does num_blocks=1 vs num_blocks=2 (same world_size=G, first_block=0)
# change slot0 correctness? Isolates whether the numBlocks loop-count alone is
# the num_blocks=1 scatter bug. Single-node, nproc=4.
import os, traceback
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import IntraNodeSubGroupAllgatherSdma

import sys
NP = int(os.environ.get("PROBE_NP", "2"))
DT = torch.bfloat16 if os.environ.get("PROBE_BF16", "0") == "1" else torch.float32
RAMP = os.environ.get("PROBE_RAMP", "0") == "1"
SCALE = int(os.environ.get("PROBE_SCALE", "1"))
SP = [x*SCALE for x in [1048576, 524288, 262144, 131072, 65536][:NP]]
REPS = 3

def run(nb):
    rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"])
    G = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
    dev = torch.device(f"cuda:{rank}")
    count = sum(SP)
    # input has nb blocks; block i = this rank's shard.
    inp = torch.empty(count * nb, dtype=DT, device=dev)
    for i in range(nb):
        base = (rank+1)*17 + i*1000
        if RAMP:
            inp[i*count:(i+1)*count] = ((torch.arange(count) % 64) + base).to(DT).to(dev)
        else:
            inp[i*count:(i+1)*count] = float(base)
    out = torch.empty(count * G, dtype=DT, device=dev)
    elem = inp.element_size()
    offs, acc = [], 0
    for e in SP:
        offs.append(acc); acc += e
    ss = torch.tensor([(e*elem)//4 for e in SP], dtype=torch.int64, device=dev)
    so = torch.tensor([(o*elem)//4 for o in offs], dtype=torch.int64, device=dev)
    blk = (count*elem)//4
    h = IntraNodeSubGroupAllgatherSdma(my_pe=rank, npes=ws,
        out_buffer_bytes=count*4*G+4096, group_size=G, group_pos=rank%G,
        pe_base=(rank//G)*G, pe_stride=1)
    h.register_output_buffer(out); torch.cuda.synchronize(); dist.barrier()
    st = torch.cuda.current_stream()
    for _ in range(REPS):
        out.zero_(); torch.cuda.synchronize(); dist.barrier()
        h.gather_kernel_direct_param_contiguous(inp, out, blk, nb, G, ss, so,
            stream=st, prepare_barrier=True, first_block=0)
        h.finish_direct_stream(stream=st, barrier=True); st.synchronize(); torch.cuda.synchronize()
    # full bit-exact reference [param][rank] over the sub-group (num_blocks=1)
    sub = None
    for n in range(int(os.environ["WORLD_SIZE"])//G):
        pass
    import torch.distributed as _d
    grp = _d.new_group(ranks=list(range((rank//G)*G, (rank//G)*G+G)))
    rank_major = torch.empty(count*G, dtype=DT, device=dev)
    _d.all_gather_into_tensor(rank_major, inp[:count], group=grp)
    ref = torch.empty(count*G, dtype=DT, device=dev)
    o = 0
    for e in SP:
        for g in range(G):
            ref[o*G+g*e:o*G+g*e+e] = rank_major[g*count+o:g*count+o+e]
        o += e
    okeq = torch.equal(out, ref)
    if rank == 0:
        E0 = SP[0]
        slots = [float(out[g*E0].item()) for g in range(G)]
        print(f"NB={nb} SCALE={SCALE} bf16={DT==torch.bfloat16} bitexact_ALLRANKS={okeq} "
              f"rank0 slot0-vals={slots}")
    # reduce okeq across ranks
    t = torch.tensor([1 if okeq else 0], device=dev)
    _d.all_reduce(t)
    if rank == 0:
        print(f"NB={nb} ranks_passing={int(t.item())}/{int(os.environ['WORLD_SIZE'])}")
    dist.barrier(); h.deregister_output_buffer(out); del h

def main():
    os.environ.setdefault("MORI_ENABLE_SDMA","1"); os.environ.setdefault("MORI_SDMA_NUM_CHANNELS","1")
    rank=int(os.environ["RANK"]); ws=int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank%8)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=ws,
        device_id=torch.device(f"cuda:{rank%8}"))
    torch._C._distributed_c10d._register_process_group("default", torch.distributed.group.WORLD)
    shmem.shmem_torch_process_group_init("default")
    G=int(os.environ.get("LOCAL_WORLD_SIZE",ws))
    for n in range(ws//G):
        dist.new_group(ranks=list(range(n*G,n*G+G)))
    try:
        run(1)
    finally:
        dist.barrier(); shmem.shmem_finalize(); dist.destroy_process_group()

if __name__=="__main__":
    try: main()
    except Exception: traceback.print_exc(); raise SystemExit(1)
