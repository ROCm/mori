import unittest
from unittest.mock import MagicMock, patch

import torch

import mori.ccl.torch_fsdp as torch_fsdp
from mori.ccl.torch_fsdp import MoriSdmaAllGather


class TestMoriSdmaAllGather(unittest.TestCase):
    def test_zero_copy_disabled_uses_default_output(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=False)
        metadata = comm.prepare_output(
            [2],
            2,
            2,
            torch.float32,
            torch.device("cpu"),
            [],
            [],
            [],
        )
        self.assertIsNone(metadata)

    def test_prepare_zero_copy_metadata(self) -> None:
        comm = MoriSdmaAllGather(zero_copy_output=True)
        with patch.object(
            comm, "can_use_param_contiguous_output", return_value=True
        ):
            metadata = comm.prepare_output(
                [2, 4],
                6,
                8,
                torch.bfloat16,
                torch.device("cpu"),
                [],
                [],
                [],
            )
        self.assertIsNotNone(metadata)
        split_sizes, split_offsets = metadata
        self.assertEqual(split_sizes.tolist(), [1, 2])
        self.assertEqual(split_offsets.tolist(), [0, 1])

    def test_unaligned_split_is_rejected(self) -> None:
        comm = MoriSdmaAllGather()
        with patch.object(
            comm, "can_use_param_contiguous_output", return_value=True
        ), self.assertRaisesRegex(RuntimeError, "4-byte aligned"):
            comm.prepare_output(
                [1],
                1,
                2,
                torch.bfloat16,
                torch.device("cpu"),
                [],
                [],
                [],
            )

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
