"""CPU-only smoke tests for the engine allocation and control surface."""


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


def test_engine_control_value_types_and_cpu_executor_need_no_gpu():
    import ctypes

    from mori.umbp.engine import (
        CompletionProof,
        CancelLoadResult,
        DelegateResult,
        EngineClient,
        EngineExecutor,
        LoadOutcome,
        LocalPageState,
        PageGeometry,
        PageRegion,
        PoolEpoch,
        PublishReceiptState,
        PublishResult,
        ReservationPurpose,
        ReadSourceKind,
    )

    epoch = PoolEpoch(b"e" * 16, 3)
    assert epoch.boot_id == b"e" * 16
    assert epoch.reset_seq == 3
    assert CompletionProof.Completed != CompletionProof.Unknown
    assert LocalPageState.Cached != LocalPageState.Quarantined
    assert ReservationPurpose.Load != ReservationPurpose.Extend
    assert ReadSourceKind.LocalHbm != ReadSourceKind.LocalDram
    assert LoadOutcome.Completed != LoadOutcome.Unknown
    assert CancelLoadResult.NeverSubmitted != CancelLoadResult.InFlight
    assert PublishReceiptState.Sent != PublishReceiptState.Accepted
    delegated = DelegateResult()
    assert delegated.success is False
    assert delegated.handles == []
    published = PublishResult()
    assert published.success is False
    assert published.accepted == []
    assert hasattr(EngineClient, "lookup")
    assert hasattr(EngineClient, "prepare_load")
    assert hasattr(EngineClient, "cancel_load")
    assert hasattr(EngineClient, "publish_status")
    assert hasattr(EngineClient, "poll_publish_receipts")
    assert hasattr(EngineClient, "consume_event_overflow")
    assert hasattr(EngineClient, "acknowledge_publish_receipt")

    storage = (ctypes.c_ubyte * 96)()
    region = PageRegion(
        "kv",
        ctypes.addressof(storage),
        len(storage),
        32,
        1,
        0,
        component="kv",
    )
    geometry = PageGeometry()
    geometry.page_tokens = 1
    geometry.num_pages = 2
    geometry.object_bytes = 32
    geometry.layout_id = "cpu-test-layout"
    geometry.regions = [region]
    assert geometry.validate()

    executor = EngineExecutor("cpu-executor", -1)
    assert executor.register_original_region("kv", ctypes.addressof(storage), len(storage))
    assert not executor.register_original_region("kv", ctypes.addressof(storage), len(storage))
    assert executor.map_peer_region(
        "peer-executor", "kv", ctypes.addressof(storage), len(storage)
    )
    assert hasattr(executor, "submit_load")
    assert executor.poll_loads() == []
    executor.close()
