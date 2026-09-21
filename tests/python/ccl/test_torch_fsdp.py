import unittest
import warnings
import weakref
from unittest.mock import MagicMock, call, patch

import torch

import mori.ccl.torch_fsdp as torch_fsdp
from mori.ccl.torch_fsdp import MoriSdmaAllGather, MoriSdmaAllGatherPool
from torch.distributed.fsdp._fully_shard._all_gather_layout import (
    AllGatherInputMetadata,
    AllGatherLayout,
    DefaultAllGatherLayout,
)


def _prepare(
    layout,
    split_sizes: list[int],
    *,
    dtype: torch.dtype = torch.float32,
    eligible: bool = True,
    world_size: int = 2,
    device: torch.device = torch.device("cpu"),
) -> object | None:
    return layout.prepare_output(
        AllGatherInputMetadata(
            input_split_sizes=split_sizes,
            input_numel=sum(split_sizes),
            world_size=world_size,
            dtype=dtype,
            device=device,
            can_use_param_contiguous_output=eligible,
        )
    )


class TestMoriSdmaAllGather(unittest.TestCase):
    def test_zero_copy_disabled_uses_default_output(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=False)
        self.assertIsNone(_prepare(comm.layout, [2, 4]))

    def test_rank_major_backend_cannot_be_shared_between_groups(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=False)
        comm.layout._bind_owner(object())
        with self.assertRaisesRegex(ValueError, "cannot be shared"):
            comm.layout._bind_owner(object())

    def test_fallback_selects_default_finalizer(self) -> None:
        comm = MoriSdmaAllGather()
        copy_in, selected_layout, metadata = comm.layout.prepare(
            AllGatherInputMetadata(
                input_split_sizes=[2, 4], input_numel=6, world_size=2,
                dtype=torch.float32, device=torch.device("cpu"),
                can_use_param_contiguous_output=False,
            )
        )
        self.assertIs(copy_in, torch.ops.fsdp.all_gather_copy_in)
        self.assertIsInstance(selected_layout, DefaultAllGatherLayout)
        self.assertIsNone(metadata)
        self.assertIsNone(comm._param_contiguous_split_sizes)

    def test_uses_safe_base_copy_in(self):
        comm = MoriSdmaAllGather()
        self.assertIs(type(comm.layout).copy_in, AllGatherLayout.copy_in)
        _prepare(comm.layout, [2, 4])
        output = torch.full((12,), -1.0)
        source, returned = comm.layout.copy_in(
            [torch.arange(2.0), torch.arange(4.0)], output, [2, 4], 6, 1
        )
        self.assertIs(returned, output)
        self.assertEqual(output.tolist(), [-1.0] * 12)
        self.assertNotEqual(source.untyped_storage().data_ptr(), output.untyped_storage().data_ptr())
        self.assertEqual(source.tolist(), [0, 1, 0, 1, 2, 3])

    def test_prepare_zero_copy_metadata(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=True)
        layout = comm.layout
        self.assertIsNotNone(layout)
        assert layout is not None
        metadata = _prepare(layout, [2, 4], dtype=torch.bfloat16)
        self.assertIsNotNone(metadata)
        split_sizes, split_offsets = metadata
        self.assertEqual(split_sizes.tolist(), [1, 2])
        self.assertEqual(split_offsets.tolist(), [0, 1])

    def test_metadata_cache_reuses_after_release(self) -> None:
        comm = MoriSdmaAllGather()
        first = _prepare(comm.layout, [2, 4], dtype=torch.bfloat16)
        comm.release_output()
        self.assertIsNone(comm._param_contiguous_split_sizes)
        with patch.object(torch, "tensor", side_effect=AssertionError("cache miss")), \
                patch.object(torch, "empty", side_effect=AssertionError("allocation")):
            second = _prepare(comm.layout, [2, 4], dtype=torch.bfloat16)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertEqual(comm._param_contiguous_input_nbytes, 12)

    def test_metadata_cache_invalidates_complete_key(self) -> None:
        cases = [
            ([4, 2], torch.float32, 2, torch.device("cpu")),
            ([2, 6], torch.float32, 2, torch.device("cpu")),
            ([2, 4], torch.bfloat16, 2, torch.device("cpu")),
            ([2, 4], torch.float32, 4, torch.device("cpu")),
            ([2, 4], torch.float32, 2, torch.device("meta")),
        ]
        for splits, dtype, world_size, device in cases:
            with self.subTest(splits=splits, dtype=dtype, world_size=world_size, device=device):
                comm = MoriSdmaAllGather()
                first = _prepare(comm.layout, [2, 4])
                with patch.object(torch, "tensor", wraps=torch.tensor) as create:
                    second = _prepare(
                        comm.layout, splits, dtype=dtype, world_size=world_size, device=device
                    )
                self.assertEqual(create.call_count, 2)
                self.assertIsNot(first[0], second[0])
                self.assertEqual(second[0].device, device)
                self.assertEqual(first[0].tolist(), [2, 4])
                self.assertEqual(first[1].tolist(), [0, 2])

    def test_metadata_cache_does_not_bypass_fallback(self) -> None:
        comm = MoriSdmaAllGather()
        first = _prepare(comm.layout, [2, 4], dtype=torch.bfloat16)
        self.assertIsNone(_prepare(comm.layout, [2, 4], dtype=torch.bfloat16, eligible=False))
        self.assertIsNone(comm._param_contiguous_split_sizes)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(_prepare(comm.layout, [1, 5], dtype=torch.bfloat16))
        self.assertEqual(len(caught), 1)
        self.assertIn("4-byte aligned", str(caught[0].message))
        with patch.object(torch, "tensor", side_effect=AssertionError("cache miss")):
            restored = _prepare(comm.layout, [2, 4], dtype=torch.bfloat16)
        self.assertIs(restored[0], first[0])

    def test_metadata_cache_rejects_invalid_input_size(self) -> None:
        comm = MoriSdmaAllGather()
        _prepare(comm.layout, [2, 4])
        with self.assertRaisesRegex(RuntimeError, "do not match input numel"):
            comm.layout.prepare_output(AllGatherInputMetadata(
                input_split_sizes=[2, 4], input_numel=7, world_size=2,
                dtype=torch.float32, device=torch.device("cpu"),
                can_use_param_contiguous_output=True,
            ))
        self.assertIsNone(comm._param_contiguous_split_sizes)

    def test_metadata_cache_retains_only_latest_entry(self) -> None:
        comm = MoriSdmaAllGather()
        references = []
        for size in range(2, 34, 2):
            metadata = _prepare(comm.layout, [size, 4])
            references.append(weakref.ref(metadata[0]))
        del metadata
        comm.release_output()
        self.assertTrue(all(reference() is None for reference in references[:-1]))
        self.assertIsNotNone(references[-1]())

    def test_metadata_cache_miss_failure_is_atomic(self) -> None:
        comm = MoriSdmaAllGather()
        first = _prepare(comm.layout, [2, 4])
        with patch.object(torch, "tensor", side_effect=[torch.zeros(2), RuntimeError("upload")]):
            with self.assertRaisesRegex(RuntimeError, "upload"):
                _prepare(comm.layout, [4, 2])
        self.assertIsNone(comm._param_contiguous_split_sizes)
        with patch.object(torch, "tensor", side_effect=AssertionError("cache miss")):
            restored = _prepare(comm.layout, [2, 4])
        self.assertIs(restored[0], first[0])
        self.assertIs(restored[1], first[1])

    def test_metadata_cache_survives_failed_call_without_selecting_layout(self) -> None:
        comm = MoriSdmaAllGather()
        first = _prepare(comm.layout, [2, 4])
        with patch.object(comm, "_validate_tensors", side_effect=RuntimeError("input")):
            with self.assertRaisesRegex(RuntimeError, "input"):
                comm(torch.empty(12), torch.empty(6), MagicMock())
        self.assertFalse(comm._can_call_param_contiguous(torch.empty(6)))
        with patch.object(torch, "tensor", side_effect=AssertionError("cache miss")):
            restored = _prepare(comm.layout, [2, 4])
        self.assertIs(restored[0], first[0])

    def test_unaligned_split_warns_once_and_falls_back(self) -> None:
        comm = MoriSdmaAllGather()
        layout = comm.layout
        self.assertIsNotNone(layout)
        assert layout is not None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                self.assertIsNone(_prepare(layout, [1], dtype=torch.bfloat16))
        self.assertEqual(len(caught), 1)
        self.assertIn("falling back", str(caught[0].message))

    def test_layout_fallback_clears_collective_metadata(self) -> None:
        comm = MoriSdmaAllGather()
        layout = comm.layout
        self.assertIsNotNone(layout)
        assert layout is not None
        _prepare(layout, [1])
        self.assertIsNone(_prepare(layout, [1], eligible=False))
        self.assertFalse(comm._can_call_param_contiguous(torch.empty(1)))

    def test_metadata_validation_does_not_read_gpu_values(self) -> None:
        comm = MoriSdmaAllGather()
        _prepare(comm.layout, [2, 4], dtype=torch.bfloat16)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError):
            self.assertTrue(
                comm._can_call_param_contiguous(torch.empty(6, dtype=torch.bfloat16))
            )
            self.assertFalse(
                comm._can_call_param_contiguous(torch.empty(7, dtype=torch.bfloat16))
            )
        self.assertIsNone(comm._param_contiguous_split_sizes)
        self.assertEqual(comm._param_contiguous_input_nbytes, 0)

    def test_allocate_reuses_persistent_output(self) -> None:
        comm = MoriSdmaAllGather()
        output = comm.allocate(
            (16,), dtype=torch.bfloat16, device=torch.device("cpu")
        )
        reused = comm.allocate(
            (8,), dtype=torch.bfloat16, device=torch.device("cpu")
        )
        self.assertEqual(output.data_ptr(), reused.data_ptr())

    def test_persistent_release_orders_packing_and_collective(self) -> None:
        for zero_copy in (False, True):
            with self.subTest(zero_copy=zero_copy):
                comm = MoriSdmaAllGather(zero_copy_output=zero_copy)
                device = torch.device("cpu")
                output = comm.allocate((4,), dtype=torch.float32, device=device)
                consumer, packer, gather = MagicMock(), MagicMock(), MagicMock()
                gather.device = device
                group = MagicMock()
                group.size.return_value = 2
                with patch.object(torch.cuda, "current_stream", return_value=consumer):
                    comm.release_output()
                    comm.release_output()
                consumer.record_event.assert_called_once()
                event = consumer.record_event.return_value
                with patch.object(torch.cuda, "current_stream", return_value=packer):
                    reused = comm.allocate((4,), dtype=torch.float32, device=device)
                self.assertEqual(output.data_ptr(), reused.data_ptr())
                packer.wait_event.assert_called_once_with(event)
                order = []
                gather.wait_event.side_effect = lambda e: order.append("wait")
                collective = MagicMock()
                collective.enqueue.side_effect = lambda source, *a, **kw: order.append(
                    "ready" if source.numel() == 1 else "write"
                )
                with (
                    patch.object(torch.cuda, "current_stream", return_value=gather),
                    patch.object(comm, "_get_collective", return_value=collective),
                    patch.object(comm, "_validate_tensors"),
                    patch.object(torch.Tensor, "record_stream"),
                ):
                    comm(reused, torch.ones(2), group)
                self.assertEqual(order, ["wait", "ready", "write"])
                collective.register_output_buffer.assert_called_once_with(comm._ready_buffer)
                gather.wait_event.assert_called_once_with(event)
                packer.synchronize.assert_not_called()
                gather.synchronize.assert_not_called()

    def test_release_before_allocation_is_noop(self) -> None:
        comm = MoriSdmaAllGather()
        with patch.object(torch.cuda, "current_stream") as current_stream:
            comm.release_output()
        current_stream.assert_not_called()

    def test_backends_order_writes_on_the_same_process_group(self):
        group = MagicMock()
        streams = [MagicMock(), MagicMock()]
        order = []
        for index, stream in enumerate(streams):
            comm = MoriSdmaAllGather()
            collective = MagicMock()
            stream.wait_event.side_effect = lambda event: order.append("wait")
            collective.enqueue.side_effect = lambda *args, **kwargs: order.append("write")
            with (
                patch.object(comm, "_get_collective", return_value=collective),
                patch.object(comm, "_wait_for_peer_consumers"),
                patch.object(comm, "_validate_tensors"),
                patch.object(torch.cuda, "current_stream", return_value=stream),
                patch.object(torch.Tensor, "record_stream"),
            ):
                comm(torch.empty(4), torch.empty(2), group)
            stream.record_event.assert_called_once()
            if index:
                stream.wait_event.assert_called_once_with(streams[0].record_event.return_value)
        self.assertEqual(order, ["write", "wait", "write"])

    def test_sync_call_uses_original_enqueue_path(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=False)
        collective = MagicMock()
        group = MagicMock()
        output = torch.empty(4)
        input = torch.empty(2)
        stream = MagicMock()
        with (
            patch.object(comm, "_get_collective", return_value=collective),
            patch.object(comm, "_wait_for_peer_consumers"),
            patch.object(comm, "_validate_tensors"),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.Tensor, "record_stream", autospec=True),
        ):
            self.assertIsNone(comm(output, input, group, async_op=False))
        collective.enqueue.assert_called_once_with(
            input, output, input.numel(), stream=stream
        )

    def test_unaligned_input_uses_process_group_fallback(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=False)
        group = MagicMock()
        output = torch.empty(2, dtype=torch.bfloat16)
        input = torch.empty(1, dtype=torch.bfloat16)
        work = MagicMock()
        with (
            patch.object(comm, "_validate_tensors"),
            patch.object(torch.cuda, "current_stream"),
            patch.object(
                torch.distributed, "all_gather_into_tensor", return_value=work
            ) as fallback,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            self.assertIs(comm(output, input, group, async_op=True), work)
        fallback.assert_called_once_with(
            output, input, group=group, async_op=True
        )

    def test_param_contiguous_call_preserves_stream_lifetimes(self) -> None:
        comm = MoriSdmaAllGather()
        layout = comm.layout
        self.assertIsNotNone(layout)
        assert layout is not None
        _prepare(layout, [2], dtype=torch.bfloat16)
        collective = MagicMock()
        group = MagicMock()
        output = torch.empty(4, dtype=torch.bfloat16)
        input = torch.empty(2, dtype=torch.bfloat16)
        stream = MagicMock()
        with (
            patch.object(comm, "_get_collective", return_value=collective),
            patch.object(comm, "_wait_for_peer_consumers"),
            patch.object(comm, "_validate_tensors"),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.Tensor, "record_stream") as record_stream,
        ):
            work = comm(output, input, group, async_op=False)
            self.assertIsNone(work)
        collective.enqueue_param_contiguous.assert_called_once()
        self.assertEqual(record_stream.call_args_list, [call(stream)] * 4)
        collective.start_async_param_contiguous.assert_not_called()
        collective.wait_async.assert_not_called()
        stream.synchronize.assert_not_called()

    def test_next_call_without_prepare_uses_rank_major(self) -> None:
        comm = MoriSdmaAllGather()
        metadata = _prepare(comm.layout, [2, 4])
        collective = MagicMock()
        group = MagicMock()
        output = torch.empty(12)
        input = torch.arange(6.0)
        stream = MagicMock()

        def param_contiguous(input, output, count, sizes, offsets, *, stream):
            output.copy_(torch.cat([t.repeat(2) for t in input.split([2, 4])]))

        def rank_major(input, output, count, *, stream):
            output.copy_(input.repeat(2))

        collective.enqueue_param_contiguous.side_effect = param_contiguous
        collective.enqueue.side_effect = rank_major
        with (
            patch.object(comm, "_get_collective", return_value=collective),
            patch.object(comm, "_wait_for_peer_consumers"),
            patch.object(comm, "_validate_tensors"),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.Tensor, "record_stream"),
        ):
            comm(output, input, group)
            self.assertEqual(output.tolist(), [0, 1, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5])
            self.assertIsNone(comm._param_contiguous_split_sizes)
            self.assertIsNone(comm._param_contiguous_split_offsets)
            self.assertEqual(comm._param_contiguous_input_nbytes, 0)
            # A custom input hook may return rank-major inputs without prepare.
            comm(output, input + 10, group)
        collective.enqueue_param_contiguous.assert_called_once()
        collective.enqueue.assert_called_once()
        self.assertEqual(output.tolist(), (input + 10).repeat(2).tolist())
        self.assertEqual(metadata[0].tolist(), [2, 4])

    def test_async_work_orders_each_consumer_after_completion(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=False)
        collective = MagicMock()
        group = MagicMock()
        output = torch.empty(4)
        input = torch.empty(2)
        stream = MagicMock()
        consumer = MagicMock()
        other_consumer = MagicMock()
        order = MagicMock()
        order.attach_mock(collective.enqueue, "enqueue")
        order.attach_mock(stream.record_event, "record_event")
        with (
            patch.object(comm, "_get_collective", return_value=collective),
            patch.object(comm, "_wait_for_peer_consumers"),
            patch.object(comm, "_validate_tensors"),
            patch.object(
                torch.cuda, "current_stream",
                side_effect=[stream, consumer, other_consumer],
            ),
            patch.object(torch.Tensor, "record_stream"),
        ):
            work = comm(output, input, group, async_op=True)
            self.assertIsInstance(work, torch.distributed.Work)
            self.assertTrue(work.wait())
            self.assertTrue(work.wait())
        collective.enqueue.assert_called_once_with(
            input, output, input.numel(), stream=stream
        )
        self.assertEqual(
            order.mock_calls,
            [call.enqueue(input, output, input.numel(), stream=stream),
             call.record_event()],
        )
        consumer.wait_event.assert_called_once_with(stream.record_event.return_value)
        other_consumer.wait_event.assert_called_once_with(stream.record_event.return_value)
        collective.wait_async.assert_not_called()
        stream.synchronize.assert_not_called()

    def test_param_contiguous_async_call_returns_event_work(self) -> None:
        comm = MoriSdmaAllGather()
        _prepare(comm.layout, [2], dtype=torch.bfloat16)
        collective = MagicMock()
        stream = MagicMock()
        consumer = MagicMock()
        with (
            patch.object(comm, "_get_collective", return_value=collective),
            patch.object(comm, "_wait_for_peer_consumers"),
            patch.object(comm, "_validate_tensors"),
            patch.object(torch.Tensor, "record_stream"),
            patch.object(torch.cuda, "current_stream", side_effect=[stream, consumer]),
        ):
            work = comm(
                torch.empty(4, dtype=torch.bfloat16),
                torch.empty(2, dtype=torch.bfloat16),
                MagicMock(), async_op=True,
            )
            self.assertIsInstance(work, torch.distributed.Work)
            self.assertTrue(work.wait())
        collective.enqueue_param_contiguous.assert_called_once()
        consumer.wait_event.assert_called_once_with(stream.record_event.return_value)
        collective.wait_async.assert_not_called()

    def test_missing_mori_dependency_has_actionable_error(self) -> None:
        def import_module(name: str):
            raise ModuleNotFoundError("No module named 'mori'", name=name)

        comm = MoriSdmaAllGather()
        group = MagicMock()
        group.rank.return_value = 0
        group.size.return_value = 1
        with patch.object(
            torch_fsdp.importlib, "import_module", side_effect=import_module
        ), self.assertRaisesRegex(RuntimeError, "optional ROCm MORI Python package"):
            comm._get_collective(group)

    def test_persistent_collective_is_bound_to_process_group_identity(self):
        comm = MoriSdmaAllGather()
        first, second = MagicMock(), MagicMock()
        for group in (first, second):
            group.rank.return_value = 0
            group.size.return_value = 2
        with patch.object(comm, "_make_collective", return_value=MagicMock()) as make:
            comm._get_collective(first)
            comm._get_collective(first)
            with self.assertRaisesRegex(ValueError, "different process group"):
                comm._get_collective(second)
        make.assert_called_once_with(first)


class TestMoriSdmaAllGatherPool(unittest.TestCase):
    def test_pool_resolves_device_and_selects_its_allocator(self):
        for device in ("cuda", "cuda:3"):
            with self.subTest(device=device):
                arena = MagicMock()
                arena.device = torch.device("cuda:3")
                group = MagicMock()
                group.size.return_value = 2
                with patch.object(torch.cuda, "current_device", return_value=3), \
                        patch.object(torch.cuda, "device") as device_guard, \
                        patch.object(torch.cuda, "MemPool"), \
                        patch.object(torch.cuda, "use_mem_pool") as use_pool, \
                        patch.object(torch.cuda, "current_stream"), \
                        patch.object(torch, "empty", return_value=arena), \
                        patch.object(torch, "zeros"):
                    pool = MoriSdmaAllGatherPool([128], group=group, device=torch.device(device))
                self.assertEqual(pool._device, torch.device("cuda:3"))
                device_guard.assert_called_once_with(torch.device("cuda:3"))
                use_pool.assert_called_once_with(pool._mem_pool, device=torch.device("cuda:3"))

    def _initialize(self, pool, mutate=None):
        def exchange(outputs, config, **kwargs):
            outputs[:] = [config] * pool._group.size()
            if mutate is not None:
                outputs[-1] = mutate(config)

        with patch.object(dist := torch.distributed, "get_process_group_ranks", return_value=[0, 1]), \
                patch.object(dist, "all_gather_object", side_effect=exchange):
            pool.initialize()

    def _pool(self, sizes=(128,)):
        group = MagicMock()
        group.size.return_value = 2
        group.rank.return_value = 0
        with patch.object(torch.cuda, "device"), patch.object(torch.cuda, "MemPool"), \
                patch.object(torch.cuda, "use_mem_pool"), \
                patch.object(torch.distributed, "new_group"), patch.object(
            torch.distributed, "get_process_group_ranks", return_value=[0, 1]
        ):
            return MoriSdmaAllGatherPool(
                sizes, group=group, device=torch.device("cpu")
            )

    def test_reuse_preserves_saved_views_and_bounds_storage(self):
        pool = self._pool()
        first, second = [MoriSdmaAllGather(output_pool=pool) for _ in range(2)]
        self._initialize(pool)
        output = first.allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        output.copy_(torch.arange(8.0))
        param = torch.nn.Parameter(output[:4])
        loss = param.square().sum()
        with patch.object(torch.cuda, "current_stream"):
            first.release_output()
        with patch.object(torch.cuda, "current_stream"):
            other = second.allocate((16,), dtype=torch.float32, device=torch.device("cpu"))
        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(other):
            other.fill_(100)
        with patch.object(torch.cuda, "current_stream"):
            second.release_output()
        with patch.object(torch.cuda, "current_stream"):
            restored = first.allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(restored):
            restored.copy_(torch.arange(8.0))
        loss.backward()
        self.assertEqual(param.grad.tolist(), [0, 2, 4, 6])
        self.assertEqual(output.data_ptr(), other.data_ptr())
        self.assertEqual(output.data_ptr(), restored.data_ptr())
        self.assertEqual(pool.allocated_bytes, 144)
        self.assertIsNone(first._output_buffer)
        self.assertIsNone(second._output_buffer)

    def test_live_slot_conflict_and_capacity_are_explicit(self):
        pool = self._pool((32, 64))
        first = MoriSdmaAllGather(output_pool=pool)
        conflict = MoriSdmaAllGather(output_pool=pool)
        other = MoriSdmaAllGather(output_pool=pool, buffer_index=1)
        self._initialize(pool)
        first.allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "still leased"):
            conflict.allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "slot has 64"):
            other.allocate((17,), dtype=torch.float32, device=torch.device("cpu"))
        other.allocate((16,), dtype=torch.float32, device=torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "reshard all groups"):
            pool.close()

    def test_remote_readiness_follows_previous_compute(self):
        pool = self._pool()
        first, second = [MoriSdmaAllGather(output_pool=pool) for _ in range(2)]
        self._initialize(pool)
        first.allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        compute, gather = MagicMock(), MagicMock()
        with patch.object(torch.cuda, "current_stream", return_value=compute):
            first.release_output()
        with patch.object(torch.cuda, "current_stream") as packing_stream:
            second.allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        packing_stream.return_value.wait_event.assert_called_once_with(
            compute.record_event.return_value
        )
        order = []
        gather.wait_event.side_effect = lambda e: order.append("compute_done")
        collective = MagicMock()
        collective.enqueue.side_effect = lambda *args, **kwargs: order.append("readiness_collective")
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
                patch.object(second, "_make_collective", return_value=collective):
            pool._before_write(second, gather)
        self.assertEqual(order, ["compute_done", "readiness_collective"])
        gather.wait_event.assert_called_once_with(compute.record_event.return_value)
        collective.enqueue.assert_called_once_with(pool._ready_input, pool._ready_output, 1, stream=gather)
        self.assertGreaterEqual(pool._ready_output.data_ptr(), pool._slots[-1].buffer.data_ptr() + pool._slots[-1].buffer.numel())
        gather.synchronize.assert_not_called()

    def test_slot_writes_are_serialized_across_streams(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        comm.allocate((8,), dtype=torch.float32, device=torch.device("cpu"))
        previous, current = MagicMock(), MagicMock()
        pool._after_write(previous)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
                patch.object(comm, "_make_collective", return_value=MagicMock()):
            pool._before_write(comm, current)
        current.wait_event.assert_called_once_with(previous.record_event.return_value)

    def test_shared_slot_is_registered_once(self):
        pool = self._pool((128, 128))
        first, second = [MoriSdmaAllGather(output_pool=pool, buffer_index=i) for i in range(2)]
        self._initialize(pool)
        collective = MagicMock()
        with patch.object(first, "_make_collective", return_value=collective), \
                patch.object(second, "_make_collective") as make_second:
            self.assertIs(first._get_collective(pool._group), collective)
            self.assertIs(second._get_collective(pool._group), collective)
        collective.register_output_buffer.assert_called_once_with(pool._buffer)
        make_second.assert_not_called()

    def test_arena_slots_are_aligned_and_nonoverlapping(self):
        pool = self._pool((4, 20, 32))
        self.assertEqual(pool.allocated_bytes, 96)
        base = pool._buffer.data_ptr()
        self.assertEqual([slot.buffer.data_ptr() - base for slot in pool._slots], [0, 16, 48])
        self.assertEqual([slot.buffer.numel() for slot in pool._slots], [4, 20, 32])

    def test_fallback_keeps_independent_input_and_default_copy_out(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        metadata = _prepare(comm.layout, [2, 4], eligible=False)
        self.assertIs(metadata, torch_fsdp._POOLED_RANK_MAJOR)
        output = comm.allocate((12,), dtype=torch.float32, device=torch.device("cpu"))
        source, returned = comm.layout.copy_in(
            [torch.arange(2.0), torch.arange(4.0)], output, [2, 4], 6, 0
        )
        self.assertIs(returned, output)
        self.assertNotEqual(source.untyped_storage().data_ptr(), output.untyped_storage().data_ptr())
        self.assertEqual(source.tolist(), [0, 1, 0, 1, 2, 3])
        params = [
            torch_fsdp.AllGatherParamMetadata([n], [torch.float32], 0, torch.Size([n]), [], False)
            for n in [2, 4]
        ]
        output.copy_(source.repeat(2))
        with patch.object(torch.cuda, "current_stream"):
            result = comm.layout.finalize_outputs(output, params, 2, metadata)
        self.assertIsNone(pool._slots[0].owner)
        self.assertFalse(result.backend_owned)
        self.assertEqual(result.tensors[0][0].tolist(), [0, 1, 0, 1])
        self.assertEqual(result.tensors[1][0].tolist(), [0, 1, 2, 3, 0, 1, 2, 3])

    def test_static_configuration_is_validated_before_use(self):
        for field in range(3):
            with self.subTest(field=field):
                pool = self._pool()
                comm = MoriSdmaAllGather(output_pool=pool, group_key="block")
                with self.assertRaisesRegex(RuntimeError, "initialize"):
                    comm.allocate((4,), dtype=torch.float32, device=torch.device("cpu"))

                def mutate(config):
                    values = list(config)
                    values[field] = ()
                    return tuple(values)

                with self.assertRaisesRegex(ValueError, "differ across ranks"):
                    self._initialize(pool, mutate)
                self.assertIsNone(pool._collective)
                self.assertIsNone(pool._slots[0].owner)
                self._initialize(pool)
                with self.assertRaisesRegex(RuntimeError, "before pool.initialize"):
                    MoriSdmaAllGather(output_pool=pool)

    def test_mixed_modes_and_duplicate_keys_are_rejected(self):
        pool = self._pool()
        MoriSdmaAllGather(output_pool=pool, group_key="block")
        with self.assertRaisesRegex(ValueError, "same zero_copy_output"):
            MoriSdmaAllGather(False, output_pool=pool)
        with self.assertRaisesRegex(ValueError, "unique strings"):
            MoriSdmaAllGather(output_pool=pool, group_key="block")

    def test_wrong_group_is_rejected_before_readiness_and_releases_lease(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        output = comm.allocate((4,), dtype=torch.float32, device=torch.device("cpu"))
        other = MagicMock()
        other.rank.return_value = 0
        with patch.object(torch.distributed, "get_process_group_ranks", return_value=[0, 1]), \
                patch.object(pool, "_before_write") as before, \
                patch.object(torch.cuda, "current_stream"):
            with self.assertRaisesRegex(ValueError, "bound process group"):
                comm(output, torch.ones(2), other)
        before.assert_not_called()
        self.assertFalse(pool._failed)
        self.assertIsNone(pool._slots[0].owner)

    def test_subgroup_uses_independent_staging_without_pool_handshake(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        cached = _prepare(comm.layout, [2])
        subgroup = MagicMock()
        subgroup.rank.return_value = 0
        subgroup.size.return_value = 1
        _prepare(comm.layout, [2], world_size=1, eligible=False)
        output = comm.allocate((2,), dtype=torch.float32, device=torch.device("cpu"))
        self.assertNotEqual(output.untyped_storage().data_ptr(), pool._buffer.data_ptr())
        with patch.object(torch.distributed, "get_process_group_ranks", return_value=[0]), \
                patch.object(comm, "_validate_tensors"), \
                patch.object(pool, "_before_write") as before, \
                patch.object(torch.distributed, "all_gather_into_tensor") as native:
            comm(output, torch.ones(2), subgroup, async_op=True)
        before.assert_not_called()
        native.assert_called_once()
        self.assertIsNone(pool._collective)
        self.assertIsNone(pool._slots[0].owner)
        with patch.object(torch, "tensor", side_effect=AssertionError("cache miss")):
            restored = _prepare(comm.layout, [2])
        self.assertIs(restored[0], cached[0])
        self.assertFalse(comm._native_fallback)

    def test_unprepared_full_gather_after_subgroup_uses_pool(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        subgroup = MagicMock()
        subgroup.rank.return_value = 0
        subgroup.size.return_value = 1
        _prepare(comm.layout, [2], world_size=1, eligible=False)
        output = comm.allocate((2,), dtype=torch.float32, device=torch.device("cpu"))
        with patch.object(torch.distributed, "get_process_group_ranks", return_value=[0]), \
                patch.object(comm, "_validate_tensors"), \
                patch.object(torch.distributed, "all_gather_into_tensor"):
            comm(output, torch.ones(2), subgroup)
        output = comm.allocate((4,), dtype=torch.float32, device=torch.device("cpu"))
        self.assertEqual(output.untyped_storage().data_ptr(), pool._buffer.data_ptr())
        self.assertIs(pool._slots[0].owner, comm)
        with patch.object(torch.cuda, "current_stream"):
            comm.release_output()

    def test_failed_call_discards_prepared_layout(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        _prepare(comm.layout, [2, 4])
        output = comm.allocate((12,), dtype=torch.float32, device=torch.device("cpu"))
        with patch.object(comm, "_validate_tensors", side_effect=RuntimeError("invalid input")), \
                patch.object(torch.cuda, "current_stream"):
            with self.assertRaisesRegex(RuntimeError, "invalid input"):
                comm(output, torch.ones(6), pool._group)
        self.assertFalse(comm._can_call_param_contiguous(torch.ones(6)))
        self.assertIsNone(pool._slots[0].owner)
        self.assertFalse(pool._failed)

    def test_release_discards_prepared_subgroup_after_copy_in_failure(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        _prepare(comm.layout, [2], world_size=1, eligible=False)
        comm.allocate((2,), dtype=torch.float32, device=torch.device("cpu"))
        comm.release_output()
        output = comm.allocate((4,), dtype=torch.float32, device=torch.device("cpu"))
        self.assertEqual(output.untyped_storage().data_ptr(), pool._buffer.data_ptr())
        with patch.object(torch.cuda, "current_stream"):
            comm.release_output()

    def test_collective_failure_disables_pool_reuse(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        output = comm.allocate((4,), dtype=torch.float32, device=torch.device("cpu"))
        with patch.object(comm, "_validate_tensors"), patch.object(torch.cuda, "current_stream"), \
                patch.object(pool, "_before_write", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                comm(output, torch.ones(2), pool._group)
        self.assertIsNone(pool._slots[0].owner)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            comm.allocate((4,), dtype=torch.float32, device=torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "failed"):
            pool.close()

    def test_copy_in_failure_releases_real_pool_lease(self):
        from torch.distributed.fsdp._fully_shard._fsdp_collectives import foreach_all_gather

        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        stream = torch.cpu.current_stream()

        def prepare(params, group, device, backend):
            backend.allocate((4,), dtype=torch.float32, device=device).fill_(1)
            raise RuntimeError("copy-in failed")

        with patch.object(torch.cuda, "current_stream"):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "copy-in failed"):
                    foreach_all_gather([], pool._group, False, stream, stream,
                                       torch.device("cpu"), comm, all_gather_input_fn=prepare)
                self.assertIsNone(pool._slots[0].owner)
                self.assertFalse(pool._failed)
            comm.release_output()

    def test_fallback_retains_lease_for_existing_parameter_aliases(self):
        pool = self._pool()
        comm = MoriSdmaAllGather(output_pool=pool)
        self._initialize(pool)
        output = comm.allocate((4,), dtype=torch.float32, device=torch.device("cpu"))
        output.copy_(torch.arange(4.0))
        params = [torch_fsdp.AllGatherParamMetadata(
            [2], [torch.float32], 0, torch.Size([2]), [output], True
        )]
        comm.layout.finalize_outputs(output, params, 2, torch_fsdp._POOLED_RANK_MAJOR)
        self.assertIs(pool._slots[0].owner, comm)


if __name__ == "__main__":
    unittest.main()
