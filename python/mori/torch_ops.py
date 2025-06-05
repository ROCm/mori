import logging
import os

import torch

try:
    # TODO: install shared library
    # _lib_path = os.path.join(os.path.dirname(__file__), "libmori_ops.so")
    _lib_path = "/home/ditian12/mori/build/src/pybind/libmori_pybinds.so"
    torch.ops.load_library(_lib_path)
    torch_mori_ops = torch.ops.mori_ops
    torch_mori_shmem = torch.ops.mori_shmem
except OSError:
    logging.exception("Error loading libmori_pybinds")
