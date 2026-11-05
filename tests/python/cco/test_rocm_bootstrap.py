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

from types import SimpleNamespace

import pytest

import mori._rocm_bootstrap as bootstrap


@pytest.fixture(autouse=True)
def reset_bootstrap(monkeypatch):
    monkeypatch.setattr(bootstrap, "_bootstrap_complete", False)
    monkeypatch.delenv("MORI_DISABLE_ROCM_SDK", raising=False)


def test_reuses_an_already_loaded_hip_runtime(monkeypatch):
    loaded = ("/runtime/libamdhip64.so.7",)
    monkeypatch.setattr(bootstrap, "_loaded_libamdhip64_paths", lambda: loaded)

    def unexpected_import(name):
        pytest.fail(f"unexpected import: {name}")

    monkeypatch.setattr(bootstrap.importlib, "import_module", unexpected_import)

    assert bootstrap.ensure_rocm_runtime() == loaded


def test_falls_back_to_system_rocm_when_sdk_is_absent(monkeypatch):
    monkeypatch.setattr(bootstrap, "_loaded_libamdhip64_paths", lambda: ())

    def missing_sdk(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(bootstrap.importlib, "import_module", missing_sdk)

    assert bootstrap.ensure_rocm_runtime() == ()


def test_preloads_sdk_runtime_with_build_version_check(monkeypatch):
    loaded = False
    calls = []

    def loaded_paths():
        return ("/sdk/libamdhip64.so.7",) if loaded else ()

    def initialize_process(**kwargs):
        nonlocal loaded
        calls.append(kwargs)
        loaded = True

    sdk = SimpleNamespace(initialize_process=initialize_process)
    monkeypatch.setattr(bootstrap, "_loaded_libamdhip64_paths", loaded_paths)
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: sdk)
    monkeypatch.setattr(bootstrap, "ROCM_BUILD_VERSION", "7.14.0")
    monkeypatch.setattr(bootstrap, "ROCM_VERSION_PATTERN", "7.14.*")

    assert bootstrap.ensure_rocm_runtime() == ("/sdk/libamdhip64.so.7",)
    assert calls == [
        {
            "preload_shortnames": ["amd_comgr", "amdhip64"],
            "check_version": "7.14.*",
            "fail_on_version_mismatch": False,
        }
    ]

    assert bootstrap.ensure_rocm_runtime() == ("/sdk/libamdhip64.so.7",)
    assert len(calls) == 1


def test_explicit_disable_keeps_system_rocm_fallback(monkeypatch):
    monkeypatch.setenv("MORI_DISABLE_ROCM_SDK", "1")
    monkeypatch.setattr(bootstrap, "_loaded_libamdhip64_paths", lambda: ())

    def unexpected_import(name):
        pytest.fail(f"unexpected import: {name}")

    monkeypatch.setattr(bootstrap.importlib, "import_module", unexpected_import)

    assert bootstrap.ensure_rocm_runtime() == ()


def test_sdk_initialization_failure_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(bootstrap, "_loaded_libamdhip64_paths", lambda: ())
    monkeypatch.setattr(bootstrap, "ROCM_BUILD_VERSION", "7.14.0")

    def fail_initialization(**kwargs):
        raise OSError("missing sdk library")

    sdk = SimpleNamespace(initialize_process=fail_initialization)
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: sdk)

    with pytest.raises(RuntimeError, match="built against ROCm 7.14.0"):
        bootstrap.ensure_rocm_runtime()
    assert bootstrap._bootstrap_complete is False


def test_sdk_must_actually_load_hip(monkeypatch):
    monkeypatch.setattr(bootstrap, "_loaded_libamdhip64_paths", lambda: ())
    sdk = SimpleNamespace(initialize_process=lambda **kwargs: None)
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: sdk)

    with pytest.raises(RuntimeError, match="without loading libamdhip64"):
        bootstrap.ensure_rocm_runtime()


def test_baked_version_uses_major_minor_compatibility_pattern():
    if bootstrap.ROCM_BUILD_VERSION is None:
        pytest.skip("ROCm build metadata is generated while building the package")

    major, minor, *_ = bootstrap.ROCM_BUILD_VERSION.split(".")
    assert bootstrap.ROCM_VERSION_PATTERN == f"{major}.{minor}.*"
