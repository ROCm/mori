"""Engine-owned KV memory primitives.

The control-plane client is added in M2. M1 exposes the VMM allocator used by
``torch.cuda.use_mem_pool`` plus allocation introspection/export.
"""

from .kv_allocator import allocator_stats, export_allocation, kv_mem_pool, lookup_allocation

__all__ = ["allocator_stats", "export_allocation", "kv_mem_pool", "lookup_allocation"]
