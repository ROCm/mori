import logging
import os

import torch

try:
    # _lib_path = os.path.join(os.path.dirname(__file__), "libmori_ops.so")
    _lib_path = "/home/ditian12/mori/build/src/pybinds/libmori_pybinds.so"
    torch.ops.load_library(_lib_path)
    torch_mori_ops = torch.ops.mori_ops
except OSError:
    from types import SimpleNamespace

    torch_mori_ops = SimpleNamespace()
    logging.exception("Error loading libmori_pybinds")
