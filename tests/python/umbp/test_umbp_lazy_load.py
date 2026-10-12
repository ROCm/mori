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
"""mori.cpp loads UMBP's module (libmori_umbp_pybinds) lazily, and only for names it lacks.

UMBP links gRPC/Protobuf, so mori.ops/io/shmem must import without loading it.
Each case runs in a fresh interpreter, because what matters is which modules
end up in sys.modules.
"""

import subprocess
import sys
import textwrap

import pytest

mori_cpp = pytest.importorskip("mori.cpp")

_UMBP = mori_cpp._UMBP_LIB
try:
    mori_cpp._load_umbp()
    _skip_reason = None
except ImportError as e:  # built without UMBP, or no gRPC runtime on this host
    _skip_reason = str(e)
needs_umbp = pytest.mark.skipif(
    _skip_reason is not None, reason=f"UMBP unavailable: {_skip_reason}"
)


def _run(code: str) -> None:
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_ops_io_shmem_do_not_load_umbp():
    _run(
        f"""
        import sys
        import mori.ops, mori.io, mori.shmem
        assert {_UMBP!r} not in sys.modules
    """
    )


def test_unknown_names_are_attribute_errors():
    _run(
        f"""
        import sys
        import mori.cpp as c
        assert not hasattr(c, "_private_probe")
        assert {_UMBP!r} not in sys.modules  # private names never trigger a load
        assert not hasattr(c, "NoSuchThing")
        assert getattr(c, "NoSuchThing", None) is None
    """
    )


def test_unloadable_umbp_degrades_to_attribute_error():
    _run(
        """
        import mori.cpp as c
        c._UMBP_LIB = "libmori_no_such_module"
        assert not hasattr(c, "UMBPClient")
        try:
            import mori.umbp
        except ImportError as e:
            assert "built without UMBP" in str(e), e
        else:
            raise AssertionError("import mori.umbp succeeded without its module")
    """
    )


@needs_umbp
def test_umbp_names_resolve_through_mori_cpp():
    _run(
        f"""
        import sys
        from mori.cpp import UMBPClient
        import mori.cpp as c
        import mori.umbp
        assert UMBPClient is mori.umbp.UMBPClient
        assert c.HBM == c.UMBPTierType.HBM  # export_values() names
        assert {_UMBP!r} in sys.modules
    """
    )


@needs_umbp
def test_cross_module_default_resolves():
    _run(
        """
        import mori.umbp
        doc = mori.umbp.UMBPClient.register_memory.__doc__
        assert "MemoryLocationType.CPU" in doc, doc
    """
    )


@needs_umbp
def test_concurrent_first_access_loads_once():
    _run(
        """
        import threading
        import mori.cpp as c
        barrier = threading.Barrier(8)
        got, errors = [], []
        def touch():
            barrier.wait()
            try:
                got.append(c.UMBPConfig)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=touch) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert len(got) == 8 and all(g is got[0] for g in got)
    """
    )
