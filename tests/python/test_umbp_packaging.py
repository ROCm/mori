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
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UMBP_INIT = REPO_ROOT / "python/mori/umbp/__init__.py"


def _write_fake_mori_package(tmp_path: Path, proxy_mode: int) -> Path:
    pkg_root = tmp_path / "mori"
    (pkg_root / "cpp").mkdir(parents=True)
    (pkg_root / "umbp").mkdir(parents=True)

    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "cpp" / "__init__.py").write_text(
        "\n".join(
            [
                "class UMBPClient: pass",
                "class UMBPConfig: pass",
                "class UMBPCopyPipelineConfig: pass",
                "class UMBPDramConfig: pass",
                "class UMBPDurabilityMode: pass",
                "class UMBPIoBackend: pass",
                "class UMBPIoConfig: pass",
                "class UMBPRole: pass",
                "class UMBPSsdConfig: pass",
                "class UMBPEvictionConfig: pass",
                "class UMBPDurabilityConfig: pass",
            ]
        ),
        encoding="utf-8",
    )
    (pkg_root / "umbp" / "__init__.py").write_text(
        UMBP_INIT.read_text(encoding="utf-8"), encoding="utf-8"
    )

    proxy_path = pkg_root / "spdk_proxy"
    proxy_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    proxy_path.chmod(proxy_mode)
    return proxy_path


def _import_umbp_in_subprocess(
    tmp_path: Path, extra_env: dict[str, str]
) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("UMBP_SPDK_PROXY_BIN", None)
    env.update(extra_env)
    env["PYTHONPATH"] = str(tmp_path)

    script = """
import json
import os
import mori.umbp

proxy_bin = os.getenv("UMBP_SPDK_PROXY_BIN")
print(json.dumps({
    "proxy_bin": proxy_bin,
    "is_exec": bool(proxy_bin and os.access(proxy_bin, os.X_OK)),
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(proc.stdout)


def test_packaged_spdk_proxy_auto_configures_env(tmp_path: Path) -> None:
    proxy_path = _write_fake_mori_package(tmp_path, 0o644)

    result = _import_umbp_in_subprocess(tmp_path, {})

    assert result["proxy_bin"] == str(proxy_path)
    assert result["is_exec"] is True
    mode = proxy_path.stat().st_mode
    assert mode & stat.S_IXUSR


def test_packaged_spdk_proxy_does_not_override_explicit_env(tmp_path: Path) -> None:
    _write_fake_mori_package(tmp_path, 0o755)
    explicit_proxy = tmp_path / "user_proxy"
    explicit_proxy.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    explicit_proxy.chmod(0o755)

    result = _import_umbp_in_subprocess(
        tmp_path, {"UMBP_SPDK_PROXY_BIN": str(explicit_proxy)}
    )

    assert result["proxy_bin"] == str(explicit_proxy)
    assert result["is_exec"] is True
