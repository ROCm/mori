import unittest
import warnings
from unittest.mock import MagicMock, call, patch

import torch

import mori.ccl.torch_fsdp as torch_fsdp
from mori.ccl.torch_fsdp import MoriSdmaAllGather
from torch.distributed.fsdp._fully_shard._all_gather_layout import DefaultAllGatherLayout


def _prepare(
    layout,
    split_sizes: list[int],
    *,
    dtype: torch.dtype = torch.float32,
    eligible: bool = True,
) -> object | None:
    return layout.prepare_output(
        split_sizes,
        sum(split_sizes),
        2,
        dtype,
        torch.device("cpu"),
        [[dtype] for _ in split_sizes],
        [[size] for size in split_sizes],
        eligible,
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
            [2, 4], 6, 2, torch.float32, torch.device("cpu"),
            [[torch.float32], [torch.float32]], [[2], [4]], False,
        )
        self.assertIs(copy_in, torch.ops.fsdp.all_gather_copy_in)
        self.assertIsInstance(selected_layout, DefaultAllGatherLayout)
        self.assertIsNone(metadata)
        self.assertIsNone(comm._param_contiguous_split_sizes)

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

    def test_sync_call_uses_original_enqueue_path(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=False)
        collective = MagicMock()
        group = MagicMock()
        output = torch.empty(4)
        input = torch.empty(2)
        stream = MagicMock()
        with (
            patch.object(comm, "_get_collective", return_value=collective),
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


if __name__ == "__main__":
    unittest.main()
