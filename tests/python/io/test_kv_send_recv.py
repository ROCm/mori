import pytest
import os
import torch
import torch.distributed as dist
import socket
from multiprocessing import Queue
import sys
from torch.multiprocessing import Pipe
import mori
import traceback

import torch
import mori
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    StatusCode
)


shape = [1024,1024]
sync_port = [5656,5657,5658,5659,5660,5661,5662,5663,5664,5665,5666,5667,5668,5669]
sync_time = 0
kv_provider_ip = "10.235.192.56"
kv_consumer_ip = "10.235.192.57"

def run_get_output(cmd):
    import subprocess
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,shell=True)
    return result.stdout
  

def send_data(pybytes, receiver_ip, receiver_port):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((receiver_ip, receiver_port))
                s.sendall(pybytes)
                print(f"Sent: {pybytes}")
                s.close()
                return
        except ConnectionRefusedError:
            pass


def receive_data(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = conn.recv(2048)
            print(f"Received: {data}")
            return data
        s.close()

def exchange_engine_meta_data(io_engine_obj:IOEngine,exchange_ip,self_role):
    port = 41630
    if self_role == "kv_provider":
        provider_engine_metadata = io_engine_obj.get_engine_desc()
        provider_engine_metadata_packed = provider_engine_metadata.pack()
        send_data(provider_engine_metadata_packed, exchange_ip, port)

        port += 1
        consumer_engine_metadata = EngineDesc.unpack(receive_data(port))
        io_engine_obj.register_remote_engine(consumer_engine_metadata)
        return consumer_engine_metadata

    elif self_role == "kv_consumer":
        provider_engine_metadata = EngineDesc.unpack(receive_data(port))
        io_engine_obj.register_remote_engine(provider_engine_metadata)

        port += 1
        consumer_engine_metadata = io_engine_obj.get_engine_desc().pack()
        send_data(consumer_engine_metadata, exchange_ip, port)
        return provider_engine_metadata

    else:
        raise Exception()

def exchange_tensor_meta_data(io_engine_obj:IOEngine,tensor,exchange_ip,role):
    port = 41632
    if role == "kv_provider":
        provider_mem_metadata = io_engine_obj.register_torch_tensor(tensor)
        provider_mem_metadata_packed = provider_mem_metadata.pack()
        send_data(provider_mem_metadata_packed, exchange_ip, port)

        port +=1 
        consumer_mem_metadata = MemoryDesc.unpack(receive_data(port))
        return provider_mem_metadata,consumer_mem_metadata
    
    elif role == "kv_consumer":
        provider_mem_metadata = MemoryDesc.unpack(receive_data(port))

        port += 1
        consumer_mem_metadata = io_engine_obj.register_torch_tensor(tensor)
        consumer_mem_metadata_packed = consumer_mem_metadata.pack()
        send_data(consumer_mem_metadata_packed, exchange_ip, port)
        return consumer_mem_metadata,provider_mem_metadata
    
    else:
        raise Exception()
    
def exchange_transfer_uid(uid,exchange_ip,role):
    port = 41632
    if role == "kv_provider":
        uid = int.from_bytes(receive_data(port), byteorder='big')
        return uid
    elif role == "kv_consumer":
        send_data(int(uid).to_bytes(4, byteorder='big'),exchange_ip, port)
        return None
    else:
        raise Exception()

def waiting_for_transfer_complete(transfer_status,role):
    global sync_port
    global sync_time
    
    if role == "kv_provider":
        pass
        # ok = receive_data(sync_port[sync_time%len(sync_port)])

    elif role == "kv_consumer":
        while transfer_status.Code() == StatusCode.INIT:
            pass
        # send_data("OK".encode("utf-8"), kv_provider_ip, sync_port[sync_time%len(sync_port)])
    else:
        raise Exception()
    sync_time += 1
    

def generate_kv_cache():
    bs,seq,head,headsize = 1,1024,128,128
    torch.manual_seed(11)
    tensor_list = []
    for idx in range(seq*bs):
        tensor_list.append(idx*torch.ones([head,headsize],dtype=torch.half,device=torch.device("cuda",0)))
    cache = torch.stack(tensor_list,dim=0)
    return cache


def send_kv_cache():
    kv_provider_config = IOEngineConfig(
        host=kv_provider_ip,
        port=39876,
    )
    kv_provider = IOEngine(key="kv_provider", config=kv_provider_config)
    kv_provider.create_backend(BackendType.RDMA)
    remote_engine_meta_data = exchange_engine_meta_data(kv_provider,kv_consumer_ip,"kv_provider")
    tensor = generate_kv_cache()
    local_mem_meta_data,remote_mem_meta_data = exchange_tensor_meta_data(kv_provider,tensor,kv_consumer_ip,"kv_provider")
    ok = receive_data(sync_port[0])
    kv_provider.deregister_memory(local_mem_meta_data)
    kv_provider.deregister_remote_engine(remote_engine_meta_data)

    del kv_provider

def recv_kv_cache():
    kv_consumer_config = IOEngineConfig(
        host=kv_consumer_ip,
        port=39877,
    )
    kv_consumer = IOEngine(key="kv_consumer", config=kv_consumer_config)
    kv_consumer.create_backend(BackendType.RDMA)
    remote_engine_meta_data = exchange_engine_meta_data(kv_consumer,kv_provider_ip,"kv_consumer")
    tensor = generate_kv_cache() * 0 # empty space
    local_mem_meta_data,remote_mem_meta_data = exchange_tensor_meta_data(kv_consumer,tensor,kv_provider_ip,"kv_consumer")
    # transfer_uid = kv_consumer.allocate_transfer_uid()
    # uid = exchange_transfer_uid(transfer_uid,kv_consumer_ip,"kv_consumer")
    test_pass = True
    for idx in range(512):
        transfer_status = kv_consumer.read(
            local_mem_meta_data, 0, 
            remote_mem_meta_data, 128*128*idx* 2*2, 
            # local_mem_meta_data.size, 
            128*128*2,
            kv_consumer.allocate_transfer_uid())
        waiting_for_transfer_complete(transfer_status,"kv_consumer")   
        test_pass &= torch.allclose(tensor[0],idx*2*torch.ones_like(tensor[0]))
    send_data("OK".encode("utf-8"), kv_provider_ip, sync_port[0])
    if test_pass:
        print("all test pass!")
    kv_consumer.deregister_memory(local_mem_meta_data)
    kv_consumer.deregister_remote_engine(remote_engine_meta_data)

    del kv_consumer


if __name__=="__main__":
    machine = run_get_output(r'''   ibv_devices | awk '/bnxt_re_bond0/ {print $2}'    ''')
    is_provider = False
    print(machine)
    if machine.find("0c6e4cfffe7117d7")>=0:
        is_provider = True
    
    if is_provider:
        print("provider")
        send_kv_cache()
    else:
        print("consumer")
        recv_kv_cache()