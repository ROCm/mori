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
"""EP kernels' JIT plans.

The only EP-specific part of the C++/JIT binding: load ``libmori_ops_v2.so`` so
its kernels register into the shared JIT registry, then expose the two Plan
classes. Everything else -- the ABI, the Plan factory, schema-driven arg structs
-- is generic and lives in ``mori.jit.v2.plan_api``.

Importing this module loads the library (raising OSError if it is not built),
which is what lets the binding test skip cleanly when the .so is absent.
``MORI_V2_LIB_DIR`` points at a build tree; the package dir is the fallback.
"""

from __future__ import annotations

import os

from mori.jit.v2 import plan_api

# EP naming/library live here; the generic layer stays op-agnostic.
_LIB_NAME = "libmori_ops_v2.so"
DTYPES = plan_api.DTYPES
make_plan = plan_api.make_plan
registered_plans = plan_api.registered_plans
precompile = plan_api.precompile
library_path = plan_api.library_path


def _extra_dirs() -> list[str]:
    env = os.environ.get("MORI_V2_LIB_DIR")
    return [env] if env else []


plan_api.load_library(_LIB_NAME, extra_dirs=_extra_dirs())

EpDispatchPlan = make_plan("ep_dispatch")
EpCombinePlan = make_plan("ep_combine")
