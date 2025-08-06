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
port_list = [39876,39877,39878,39879,39880,39881,39882]
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


def test_kv_provider():
    kv_provider_config = IOEngineConfig(
        host=kv_provider_ip,
        port=port_list[0],
    )
    kv_provider = IOEngine(key="kv_provider", config=kv_provider_config)
    kv_provider.create_backend(BackendType.RDMA)
    
    # provider  send to consumer
    print("sending......")
    provider_engine_metadata = kv_provider.get_engine_desc()
    provider_engine_metadata_packed = provider_engine_metadata.pack()
    send_data(provider_engine_metadata_packed, kv_consumer_ip, port_list[2])


    tensor = torch.randn(shape).to(torch.device("cuda", 0))
    provider_mem_metadata = kv_provider.register_torch_tensor(tensor)
    provider_mem_metadata_packed = provider_mem_metadata.pack()
    send_data(provider_mem_metadata_packed, kv_consumer_ip, port_list[3])

    # provider recv from consumer
    print("receving.....")
    consumer_engine_metadata = EngineDesc.unpack(receive_data(port_list[4]))
    kv_provider.register_remote_engine(consumer_engine_metadata)


    consumer_mem_metadata = MemoryDesc.unpack(receive_data(port_list[5]))
    print(f"handshake success! I am kv provider,{consumer_engine_metadata = },{consumer_mem_metadata = }")
    print(f"testing kv consumer read..........")
    transfer_uid = int.from_bytes(receive_data(port_list[6]), byteorder='big')

    while True:
        target_side_status = kv_provider.pop_inbound_transfer_status(
            consumer_engine_metadata.key, transfer_uid
        )
        if target_side_status:
            break
    print(
        f"read finished at target {transfer_uid=}, {target_side_status.Code()} {target_side_status.Message()}"
    )
    print(tensor)
    kv_provider.deregister_memory(provider_mem_metadata)
    kv_provider.deregister_remote_engine(consumer_engine_metadata)

    del kv_provider

def test_kv_consumer():
    kv_consumer_config = IOEngineConfig(
        host=kv_consumer_ip,
        port=port_list[1],
    )
    kv_consumer = IOEngine(key="kv_consumer", config=kv_consumer_config)
    kv_consumer.create_backend(BackendType.RDMA)

    # consumer recv from provider
    print("receving.....")
    provider_engine_metadata = EngineDesc.unpack(receive_data(port_list[2]))
    kv_consumer.register_remote_engine(provider_engine_metadata)

    provider_mem_metadata = MemoryDesc.unpack(receive_data(port_list[3]))

    # consumer send to provider
    print("sending......")
    consumer_engine_metadata = kv_consumer.get_engine_desc().pack()
    send_data(consumer_engine_metadata, kv_provider_ip, port_list[4])

    tensor = torch.randn(shape).to(torch.device("cuda", 0))
    consumer_mem_metadata = kv_consumer.register_torch_tensor(tensor)
    consumer_mem_metadata_packed = consumer_mem_metadata.pack()
    send_data(consumer_mem_metadata_packed, kv_provider_ip, port_list[5])

    print(f"handshake success! I am kv consumer,{provider_engine_metadata = },{provider_mem_metadata = }")
    print(f"testing kv consumer read..........")
    transfer_uid = kv_consumer.allocate_transfer_uid()
    send_data(int(transfer_uid).to_bytes(4, byteorder='big'),kv_provider_ip, port_list[6])

    transfer_status = kv_consumer.read(
        consumer_mem_metadata, 0, provider_mem_metadata, 0, consumer_mem_metadata.size, transfer_uid
    )

    while transfer_status.Code() == StatusCode.INIT:
        pass
    print(
        f"read finished at initiator {transfer_uid=},{transfer_status.Code()} {transfer_status.Message()}"
    )
    print(tensor)
    kv_consumer.deregister_memory(consumer_mem_metadata)
    kv_consumer.deregister_remote_engine(provider_engine_metadata)

    del kv_consumer


if __name__=="__main__":
    machine = run_get_output(r'''   ibv_devices | awk '/bnxt_re_bond0/ {print $2}'    ''')
    is_provider = False
    print(machine)
    if machine.find("0c6e4cfffe7117d7")>=0:
        is_provider = True
    
    if is_provider:
        print("provider")
        test_kv_provider()
    else:
        print("consumer")
        test_kv_consumer()