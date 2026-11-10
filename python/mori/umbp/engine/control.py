"""Rank-side UMBP engine control and execution bindings."""

from mori.cpp import _load_umbp


def _binding():
    return _load_umbp().engine


# These are resolved lazily at module import without creating a socket, GPU
# context, executor thread, or memory pool.
EngineClient = _binding().EngineClient
EngineExecutor = _binding().EngineExecutor
EnginePageAccounting = _binding().EnginePageAccounting
LocalPageState = _binding().LocalPageState
LocalHoldId = _binding().LocalHoldId
CompletionProof = _binding().CompletionProof
ReadSourceKind = _binding().ReadSourceKind
LoadOutcome = _binding().LoadOutcome
CancelLoadResult = _binding().CancelLoadResult
DelegateResult = _binding().DelegateResult
PublishResult = _binding().PublishResult
PublishReceipt = _binding().PublishReceipt
PublishReceiptState = _binding().PublishReceiptState
ReadPlan = _binding().ReadPlan
PreparedLoad = _binding().PreparedLoad
LoadHandle = _binding().LoadHandle
PoolEpoch = _binding().PoolEpoch
PageRegion = _binding().PageRegion
PageGeometry = _binding().PageGeometry
AllocationHandle = _binding().AllocationHandle
ResidentRecord = _binding().ResidentRecord
RetireState = _binding().RetireState
ResetLease = _binding().ResetLease
ReservationPurpose = _binding().ReservationPurpose

__all__ = [
    "AllocationHandle",
    "CompletionProof",
    "CancelLoadResult",
    "DelegateResult",
    "EngineClient",
    "EngineExecutor",
    "EnginePageAccounting",
    "LocalHoldId",
    "LocalPageState",
    "LoadHandle",
    "LoadOutcome",
    "PageGeometry",
    "PageRegion",
    "PoolEpoch",
    "ResidentRecord",
    "ReadPlan",
    "ReadSourceKind",
    "PreparedLoad",
    "PublishReceipt",
    "PublishReceiptState",
    "PublishResult",
    "ResetLease",
    "ReservationPurpose",
    "RetireState",
]
