import unittest
import warnings
from unittest.mock import MagicMock, patch

import torch

import mori.ccl.torch_fsdp as torch_fsdp
from mori.ccl.torch_fsdp import MoriSdmaAllGather


def _prepare(
    layout,
    split_sizes: list[int],
    *,
    dtype: torch.dtype = torch.float32,
    eligible: bool = True,
    owner_token: int = 1,
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
        owner_token,
    )


class TestMoriSdmaAllGather(unittest.TestCase):
    def test_zero_copy_disabled_uses_default_output(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=False)
        self.assertIsNone(comm.layout)

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

    def test_layout_rejects_sharing_across_parameter_groups(self) -> None:
        comm = MoriSdmaAllGather()
        layout = comm.layout
        self.assertIsNotNone(layout)
        assert layout is not None
        _prepare(layout, [1], owner_token=1)
        with self.assertRaisesRegex(RuntimeError, "cannot be shared"):
            _prepare(layout, [1], owner_token=2)

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
        ):
            self.assertIsNone(comm(output, input, group, async_op=False))
        collective.enqueue.assert_called_once_with(
            input, output, input.numel(), stream=stream
        )

    def test_async_work_waits_and_synchronizes_once(self) -> None:
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
        ):
            work = comm(output, input, group, async_op=True)
            self.assertIsInstance(work, torch.distributed.Work)
            self.assertTrue(work.wait())
            self.assertTrue(work.wait())
        collective.start_async.assert_called_once_with(
            input, output, input.numel(), stream=stream
        )
        collective.wait_async.assert_called_once_with(stream=stream)
        stream.synchronize.assert_called_once_with()

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
