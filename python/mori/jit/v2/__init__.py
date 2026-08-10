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
"""JIT v2: the Python binding to the C++ Cfg-as-NTTP / content-addressed JIT
framework (libmori_jit.so, ``include/mori/jit`` + ``src/jit``).

Distinct from the v1 ``mori.jit`` bitcode compiler (``mori.jit.core`` /
``mori.jit.cache``): v1 drives hipcc from Python and assembles the cache key by
hand; v2 renders a Cfg struct to source in C++ and the rendered text *is* the
key. The two share the ``mori.jit`` namespace and the ``~/.mori/jit`` cache root
on purpose (see docs/MORI_JIT_V2_DESIGN.md).

Op-agnostic: an op registers its kernels by loading its library
(``load_library("libmori_ops_v2.so")``), then ``make_plan("<kernel>")`` builds a
Plan class from the C++-published schema.
"""

from mori.jit.v2.plan_api import (
    DTYPES,
    library_path,
    load_library,
    make_plan,
    precompile,
    registered_plans,
)

__all__ = [
    "DTYPES",
    "library_path",
    "load_library",
    "make_plan",
    "precompile",
    "registered_plans",
]
