# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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

# High-level OO API
from .communicator import (
    Communicator,
    CCODevCommRequirements,
    UniqueId,
    get_unique_id,
    CCOResource,
    AllocatedMemory,
    RegisteredWindow,
    DevCommHandle,
)
