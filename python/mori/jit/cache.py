# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""JIT cache directory management and content hashing."""

import hashlib
import os
from pathlib import Path


def get_cache_root() -> Path:
    """Return the JIT cache root directory.

    Default: ``~/.mori/jit/``.  Override with ``MORI_JIT_CACHE_DIR``.
    """
    env = os.environ.get("MORI_JIT_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".mori" / "jit"


def _hash_tree(paths: list[Path]) -> str:
    """Compute a short content hash over files and directories.

    For directories, all ``.hpp``, ``.h``, and ``.cpp`` files are included.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        if p.is_file():
            h.update(p.read_bytes())
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix in (".hpp", ".h", ".cpp"):
                    h.update(f.read_bytes())
    return h.hexdigest()[:12]


def get_cache_dir(
    arch: str, source_paths: list[Path], nic: str = "mlx5", *, cov: int | None = None
) -> Path:
    """Return the cache directory for a specific arch + NIC + content combo.

    Structure: ``<cache_root>/<arch>_<nic>[_cov<N>]/<content_hash>/``

    Args:
        cov: AMDGPU code object version. When specified, the version is
             included in the directory name to separate bitcode compiled
             with different ABI versions (e.g. cov5 for Triton, cov6 for
             FlyDSL).  ``None`` omits the suffix for backward compatibility.
    """
    content_hash = _hash_tree(source_paths)
    dir_name = f"{arch}_{nic}" if cov is None else f"{arch}_{nic}_cov{cov}"
    d = get_cache_root() / dir_name / content_hash
    d.mkdir(parents=True, exist_ok=True)
    return d
