import os
import sys
import importlib.util

cur_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.abspath(os.path.join(cur_dir, "../libmori_pybinds.so"))

spec = importlib.util.spec_from_file_location("libmori_pybinds", lib_path)
module = importlib.util.module_from_spec(spec)
sys.modules["libmori_pybinds"] = module
spec.loader.exec_module(module)

from libmori_pybinds import *
