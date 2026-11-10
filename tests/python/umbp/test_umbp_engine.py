"""CPU-only smoke tests for the M1 Python surface."""


def test_engine_binding_loads_without_initializing_a_gpu():
    from mori.umbp.engine import allocator_stats

    stats = allocator_stats()
    assert stats["allocations"] == 0
    assert stats["quarantined"] == 0
    assert stats["bytes"] == 0
    assert stats["failures"] == 0


def test_unknown_pointer_is_not_an_engine_allocation():
    from mori.umbp.engine import lookup_allocation

    assert lookup_allocation(1) is None
