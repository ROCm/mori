Installation
============

Requirements
------------

* Python 3.10 or later
* ROCm 6.4 or later
* PyTorch with ROCm support
* AMD Instinct GPU (MI308X, MI300X, MI325X, MI355X)

Installation Methods
--------------------

From Source
^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone --recursive https://github.com/ROCm/mori.git
   cd mori

   # Install PyTorch with ROCm support first
   # See https://pytorch.org for ROCm installation instructions

   # Build and install MORI
   pip install .
   # For venv build: pip install . --no-build-isolation

No hipcc is needed at install time — host code compiles with a standard C++
compiler. GPU kernels are JIT-compiled on first use and cached to ``~/.mori/jit/``.

To manually precompile all kernels (e.g. in a Docker image build):

.. code-block:: bash

   MORI_PRECOMPILE=1 python -c "import mori"

Environment Variables
---------------------

Common environment variables:

.. code-block:: bash

   # Symmetric heap size (required for shmem/EP)
   export MORI_SHMEM_HEAP_SIZE=6G

   # Log verbosity
   export MORI_GLOBAL_LOG_LEVEL=INFO  # TRACE, DEBUG, INFO

Verification
------------

Verify the installation:

.. code-block:: python

   import mori
   import torch

   # Check if modules loaded successfully
   print("MORI modules available:")
   print(f"  - mori.shmem: {hasattr(mori, 'shmem')}")
   print(f"  - mori.ops: {hasattr(mori, 'ops')}")
   print(f"  - mori.io: {hasattr(mori, 'io')}")
   print(f"  - mori.kernel_profiler: {hasattr(mori, 'kernel_profiler')}")

   # Check ROCm availability via PyTorch
   print(f"\nPyTorch version: {torch.__version__}")
   print(f"ROCm available: {torch.cuda.is_available()}")
   print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

Troubleshooting
---------------

**ImportError: No module named 'mori'**
   Ensure ROCm libraries are in your library path:

   .. code-block:: bash

      export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

**RuntimeError: No AMD GPU found**
   Verify GPU is accessible:

   .. code-block:: bash

      rocm-smi
      rocminfo | grep gfx
