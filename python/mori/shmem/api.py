from mori import cpp as mori_cpp


def shmem_torch_process_group_init(group_name: str):
    return mori_cpp.shmem_torch_process_group_init(group_name)


def shmem_finalize():
    return mori_cpp.shmem_finalize()


def shmem_mype():
    return mori_cpp.shmem_mype()


def shmem_npes():
    return mori_cpp.shmem_npes()
