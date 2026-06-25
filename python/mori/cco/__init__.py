# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License

from .cco import (
    # Low-level Cython classes (prefer high-level wrappers below)
    UniqueId as _CyUniqueId,
    Comm,
    DevCommRequirements,
    DevComm,
    # Low-level lifecycle
    get_unique_id as _cy_get_unique_id,
    comm_create,
    comm_destroy,
    # VMM allocation
    mem_alloc,
    mem_free,
    # Window registration
    window_register,
    window_register_ptr,
    window_deregister,
    # Device communicator
    dev_comm_create,
    dev_comm_destroy,
    # Barrier
    barrier_all,
    # GdaConnectionType constants
    GDA_CONNECTION_NONE,
    GDA_CONNECTION_FULL,
    GDA_CONNECTION_CROSSNODE,
    GDA_CONNECTION_RAIL,
)

# High-level OO API (mirrors nccl4py style)
from .communicator import (
    Communicator,
    CCODevCommRequirements,
    UniqueId,
    get_unique_id,
    CCOResource,
    AllocatedMemory,
    RegisteredWindow,
    AllocatedWindow,
    DevCommHandle,
)
