// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>

#include "src/pybind/mori.hpp"
#include "umbp/engine/engine_client.h"
#include "umbp/engine/engine_executor.h"
#include "umbp/engine/kv_allocator.h"

namespace py = pybind11;

namespace mori {

void RegisterMoriUmbpEngine(py::module_& parent) {
  auto m = parent.def_submodule("engine", "UMBP engine-owned KV memory primitives");
  using namespace umbp::engine;

  py::enum_<VmmHandleKind>(m, "VmmHandleKind")
      .value("PosixFd", VmmHandleKind::kPosixFd)
      .value("Fabric", VmmHandleKind::kFabric);

  py::class_<PoolEpoch>(m, "PoolEpoch")
      .def(py::init([](py::bytes boot_id, uint64_t reset_seq) {
             std::string bytes = boot_id;
             if (bytes.size() != 16) throw py::value_error("boot_id must be 16 bytes");
             PoolEpoch out;
             std::memcpy(out.boot_id.data(), bytes.data(), bytes.size());
             out.reset_seq = reset_seq;
             return out;
           }),
           py::arg("boot_id"), py::arg("reset_seq") = 0)
      .def_property_readonly("boot_id",
                             [](const PoolEpoch& value) {
                               return py::bytes(reinterpret_cast<const char*>(value.boot_id.data()),
                                                value.boot_id.size());
                             })
      .def_readwrite("reset_seq", &PoolEpoch::reset_seq);

  py::class_<PageRegion>(m, "PageRegion")
      .def(py::init([](std::string region_id, uintptr_t address, uint64_t size, uint64_t row_bytes,
                       uint64_t rows_per_page, uint64_t logical_offset, int layer_id,
                       std::string component, int location, int device) {
             PageRegion out;
             out.region_id = std::move(region_id);
             out.buffer = umbp::TransferRef::HostBytes(
                 reinterpret_cast<void*>(address), size,
                 static_cast<mori::io::MemoryLocationType>(location), device);
             out.row_bytes = row_bytes;
             out.rows_per_page = rows_per_page;
             out.logical_offset = logical_offset;
             out.layer_id = layer_id;
             out.component = std::move(component);
             return out;
           }),
           py::arg("region_id"), py::arg("address"), py::arg("size"), py::arg("row_bytes"),
           py::arg("rows_per_page"), py::arg("logical_offset"), py::arg("layer_id") = -1,
           py::arg("component") = "", py::arg("location") = 1, py::arg("device") = -1)
      .def_readwrite("region_id", &PageRegion::region_id)
      .def_readwrite("row_bytes", &PageRegion::row_bytes)
      .def_readwrite("rows_per_page", &PageRegion::rows_per_page)
      .def_readwrite("logical_offset", &PageRegion::logical_offset)
      .def_readwrite("layer_id", &PageRegion::layer_id)
      .def_readwrite("component", &PageRegion::component);

  py::class_<PageGeometry>(m, "PageGeometry")
      .def(py::init<>())
      .def_readwrite("page_tokens", &PageGeometry::page_tokens)
      .def_readwrite("num_pages", &PageGeometry::num_pages)
      .def_readwrite("object_bytes", &PageGeometry::object_bytes)
      .def_readwrite("layout_id", &PageGeometry::layout_id)
      .def_readwrite("regions", &PageGeometry::regions)
      .def("validate", &PageGeometry::Validate);

  py::class_<AllocationHandle>(m, "AllocationHandle")
      .def(py::init<>())
      .def_readwrite("pool_id", &AllocationHandle::pool_id)
      .def_readwrite("epoch", &AllocationHandle::epoch)
      .def_readwrite("backend_id", &AllocationHandle::backend_id)
      .def_readwrite("page_id", &AllocationHandle::page_id)
      .def_readwrite("generation", &AllocationHandle::generation);

  py::class_<OperationId>(m, "OperationId")
      .def_property_readonly("session_id",
                             [](const OperationId& value) {
                               return py::bytes(
                                   reinterpret_cast<const char*>(value.session_id.data()),
                                   value.session_id.size());
                             })
      .def_readonly("seq", &OperationId::seq);

  py::class_<umbp::ResidentRecord>(m, "ResidentRecord")
      .def(py::init<>())
      .def_readwrite("key", &umbp::ResidentRecord::key)
      .def_readwrite("parent_key", &umbp::ResidentRecord::parent_key)
      .def_readwrite("handle", &umbp::ResidentRecord::handle);

  py::enum_<RetireState>(m, "RetireState")
      .value("WaitingHolds", RetireState::kWaitingHolds)
      .value("TransitionPending", RetireState::kTransitionPending)
      .value("Ready", RetireState::kReady)
      .value("Released", RetireState::kReleased)
      .value("Quarantined", RetireState::kQuarantined)
      .value("Busy", RetireState::kBusy);

  py::enum_<CompletionProof>(m, "CompletionProof")
      .value("Completed", CompletionProof::kCompleted)
      .value("NeverSubmitted", CompletionProof::kNeverSubmitted)
      .value("Unknown", CompletionProof::kUnknown);

  py::enum_<ReadSourceKind>(m, "ReadSourceKind")
      .value("Missing", ReadSourceKind::kMissing)
      .value("LocalHbm", ReadSourceKind::kLocalHbm)
      .value("LocalDram", ReadSourceKind::kLocalDram)
      .value("RemoteHbm", ReadSourceKind::kRemoteHbm)
      .value("RemoteDram", ReadSourceKind::kRemoteDram);

  py::enum_<LoadOutcome>(m, "LoadOutcome")
      .value("Pending", LoadOutcome::kPending)
      .value("Completed", LoadOutcome::kCompleted)
      .value("Failed", LoadOutcome::kFailed)
      .value("Unknown", LoadOutcome::kUnknown)
      .value("Canceled", LoadOutcome::kCanceled);

  py::enum_<CancelLoadResult>(m, "CancelLoadResult")
      .value("Rejected", CancelLoadResult::kRejected)
      .value("NeverSubmitted", CancelLoadResult::kNeverSubmitted)
      .value("InFlight", CancelLoadResult::kInFlight)
      .value("Completed", CancelLoadResult::kCompleted);

  py::class_<ReadSourceHint>(m, "ReadSourceHint")
      .def_readonly("key", &ReadSourceHint::key)
      .def_readonly("kind", &ReadSourceHint::kind)
      .def_readonly("backend_id", &ReadSourceHint::backend_id)
      .def_readonly("node_id", &ReadSourceHint::node_id)
      .def_readonly("peer_address", &ReadSourceHint::peer_address)
      .def_readonly("executor_id", &ReadSourceHint::executor_id)
      .def_readonly("layout_id", &ReadSourceHint::layout_id)
      .def_readonly("handle", &ReadSourceHint::handle);

  py::class_<ReadPlan>(m, "ReadPlan")
      .def_readonly("success", &ReadPlan::success)
      .def_readonly("id", &ReadPlan::id)
      .def_readonly("epoch", &ReadPlan::epoch)
      .def_readonly("namespace", &ReadPlan::name_space)
      .def_readonly("layout_id", &ReadPlan::layout_id)
      .def_readonly("requester_backend_id", &ReadPlan::requester_backend_id)
      .def_readonly("keys", &ReadPlan::keys)
      .def_readonly("sources", &ReadPlan::sources)
      .def_readonly("valid_prefix", &ReadPlan::valid_prefix)
      .def_readonly("legal_boundaries", &ReadPlan::legal_boundaries)
      .def_readonly("error", &ReadPlan::error);

  py::class_<PreparedLoad>(m, "PreparedLoad")
      .def_readonly("success", &PreparedLoad::success)
      .def_readonly("id", &PreparedLoad::id)
      .def_readonly("request_id", &PreparedLoad::request_id)
      .def_readonly("targets", &PreparedLoad::targets)
      .def_readonly("error", &PreparedLoad::error)
      .def_property_readonly("layer_count",
                             [](const PreparedLoad& load) { return load.layers.size(); })
      .def_property_readonly("layer_ids", [](const PreparedLoad& load) {
        std::vector<int32_t> out;
        for (const auto& layer : load.layers) out.push_back(layer.layer_id);
        return out;
      });

  py::class_<RetireTicket>(m, "RetireTicket")
      .def_readonly("id", &RetireTicket::id)
      .def_readonly("handles", &RetireTicket::handles)
      .def_readonly("state", &RetireTicket::state);

  py::class_<ResetLease>(m, "ResetLease")
      .def_readonly("id", &ResetLease::id)
      .def_readonly("old_epoch", &ResetLease::old_epoch)
      .def_readonly("new_epoch", &ResetLease::new_epoch)
      .def_readonly("ready", &ResetLease::ready)
      .def_readonly("approved", &ResetLease::approved);

  py::enum_<PublishReceiptState>(m, "PublishReceiptState")
      .value("Queued", PublishReceiptState::kQueued)
      .value("Ready", PublishReceiptState::kReady)
      .value("Sent", PublishReceiptState::kSent)
      .value("Accepted", PublishReceiptState::kAccepted)
      .value("Rejected", PublishReceiptState::kRejected)
      .value("Cancelled", PublishReceiptState::kCancelled)
      .value("Unknown", PublishReceiptState::kUnknown);

  py::class_<PublishReceipt>(m, "PublishReceipt")
      .def_readonly("ticket", &PublishReceipt::ticket)
      .def_readonly("state", &PublishReceipt::state)
      .def_readonly("accepted", &PublishReceipt::accepted);

  py::class_<SharedDramRegion>(m, "SharedDramRegion")
      .def_readonly("region_id", &SharedDramRegion::region_id)
      .def_readonly("epoch", &SharedDramRegion::epoch)
      .def_readonly("backend_id", &SharedDramRegion::backend_id)
      .def_readonly("buffer_index", &SharedDramRegion::buffer_index)
      .def_readonly("fd", &SharedDramRegion::fd)
      .def_readonly("fd_offset", &SharedDramRegion::fd_offset)
      .def_readonly("length", &SharedDramRegion::length)
      .def_readonly("mapped_length", &SharedDramRegion::mapped_length)
      .def_readonly("numa_node", &SharedDramRegion::numa_node)
      .def_readonly("alignment", &SharedDramRegion::alignment);

  py::class_<PoolRegistration>(m, "PoolRegistration")
      .def_readonly("success", &PoolRegistration::success)
      .def_readonly("backend_id", &PoolRegistration::backend_id)
      .def_readonly("epoch", &PoolRegistration::epoch)
      .def_readonly("dram_regions", &PoolRegistration::dram_regions)
      .def_readonly("error", &PoolRegistration::error);

  py::enum_<ReservationPurpose>(m, "ReservationPurpose")
      .value("Load", ReservationPurpose::kLoad)
      .value("Extend", ReservationPurpose::kExtend);

  py::enum_<LocalPageState>(m, "LocalPageState")
      .value("Cached", LocalPageState::kCached)
      .value("Reserved", LocalPageState::kReserved)
      .value("InUse", LocalPageState::kInUse)
      .value("PendingFree", LocalPageState::kPendingFree)
      .value("PublishSent", LocalPageState::kPublishSent)
      .value("Published", LocalPageState::kPublished)
      .value("Retiring", LocalPageState::kRetiring)
      .value("Quarantined", LocalPageState::kQuarantined);

  py::class_<EnginePageAccounting>(m, "EnginePageAccounting")
      .def_readonly("cached", &EnginePageAccounting::cached)
      .def_readonly("reserved", &EnginePageAccounting::reserved)
      .def_readonly("in_use", &EnginePageAccounting::in_use)
      .def_readonly("pending_free", &EnginePageAccounting::pending_free)
      .def_readonly("publish_sent", &EnginePageAccounting::publish_sent)
      .def_readonly("published", &EnginePageAccounting::published)
      .def_readonly("retiring", &EnginePageAccounting::retiring)
      .def_readonly("quarantined", &EnginePageAccounting::quarantined)
      .def_readonly("available", &EnginePageAccounting::available)
      .def_readonly("total", &EnginePageAccounting::total);

  py::class_<LocalHoldId>(m, "LocalHoldId").def_readonly("value", &LocalHoldId::value);

  py::class_<ReservationResult>(m, "ReservationResult")
      .def_readonly("success", &ReservationResult::success)
      .def_readonly("id", &ReservationResult::id)
      .def_readonly("shortfall", &ReservationResult::shortfall);

  py::class_<umbp::DelegateResult>(m, "DelegateResult")
      .def(py::init<>())
      .def_readonly("success", &umbp::DelegateResult::success)
      .def_readonly("handles", &umbp::DelegateResult::handles);

  py::class_<umbp::PublishResult>(m, "PublishResult")
      .def(py::init<>())
      .def_readonly("success", &umbp::PublishResult::success)
      .def_readonly("accepted", &umbp::PublishResult::accepted);

  py::class_<EngineClient>(m, "EngineClient")
      .def(py::init([](std::string node_address, std::string executor_id, uint32_t heartbeat_ms,
                       uint32_t heartbeat_timeout_ms) {
             return std::make_unique<EngineClient>(
                 EngineClientConfig{std::move(node_address), std::move(executor_id), heartbeat_ms,
                                    heartbeat_timeout_ms});
           }),
           py::arg("node_address"), py::arg("executor_id"), py::arg("heartbeat_ms") = 500,
           py::arg("heartbeat_timeout_ms") = 5000)
      .def("register_pool", &EngineClient::RegisterPool, py::call_guard<py::gil_scoped_release>(),
           py::arg("rank"), py::arg("pool_id"), py::arg("epoch"), py::arg("geometry"))
      .def("attach_executor", &EngineClient::AttachExecutor, py::keep_alive<1, 2>())
      .def("request_delegation", &EngineClient::RequestDelegation,
           py::call_guard<py::gil_scoped_release>())
      .def("reserve", &EngineClient::Reserve, py::call_guard<py::gil_scoped_release>(),
           py::arg("request_id"), py::arg("load_pages"), py::arg("extend_pages"))
      .def("take", &EngineClient::Take, py::call_guard<py::gil_scoped_release>(),
           py::arg("reservation_id"), py::arg("purpose"), py::arg("count"))
      .def("refund", &EngineClient::Refund, py::call_guard<py::gil_scoped_release>(),
           py::arg("reservation_id"))
      .def("return_pages", &EngineClient::ReturnPages, py::call_guard<py::gil_scoped_release>())
      .def("queue_publish", &EngineClient::QueuePublish, py::arg("records"),
           py::arg("ready") = false, py::call_guard<py::gil_scoped_release>())
      .def("mark_publish_ready", &EngineClient::MarkPublishReady,
           py::call_guard<py::gil_scoped_release>())
      .def("cancel_publish", &EngineClient::CancelPublish, py::call_guard<py::gil_scoped_release>())
      .def("publish_status", &EngineClient::PublishStatus, py::call_guard<py::gil_scoped_release>())
      .def("poll_publish_receipts", &EngineClient::PollPublishReceipts,
           py::call_guard<py::gil_scoped_release>())
      .def("acknowledge_publish_receipt", &EngineClient::AcknowledgePublishReceipt,
           py::call_guard<py::gil_scoped_release>())
      .def("consume_event_overflow", &EngineClient::ConsumeEventOverflow,
           py::call_guard<py::gil_scoped_release>())
      .def("publish", &EngineClient::Publish, py::call_guard<py::gil_scoped_release>())
      .def("retire", &EngineClient::Retire, py::call_guard<py::gil_scoped_release>())
      .def("poll_retires", &EngineClient::PollRetires, py::call_guard<py::gil_scoped_release>())
      .def("acknowledge_released", &EngineClient::AcknowledgeReleased,
           py::call_guard<py::gil_scoped_release>())
      .def("quiesce_reset", &EngineClient::QuiesceReset, py::call_guard<py::gil_scoped_release>())
      .def("finish_reset", &EngineClient::FinishReset, py::call_guard<py::gil_scoped_release>(),
           py::arg("lease"), py::arg("commit"))
      .def("hold_io", &EngineClient::HoldIO, py::call_guard<py::gil_scoped_release>())
      .def("complete_io", &EngineClient::CompleteIO, py::call_guard<py::gil_scoped_release>(),
           py::arg("hold"), py::arg("proof"))
      .def("hold_reuse_guard", &EngineClient::HoldReuseGuard,
           py::call_guard<py::gil_scoped_release>())
      .def("complete_reuse_guard", &EngineClient::CompleteReuseGuard,
           py::call_guard<py::gil_scoped_release>())
      .def("lookup", &EngineClient::Lookup, py::arg("keys"), py::arg("namespace"),
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_load", &EngineClient::PrepareLoad, py::arg("plan"), py::arg("request_id"),
           py::arg("targets"), py::arg("n_pages"), py::call_guard<py::gil_scoped_release>())
      .def("cancel_load", &EngineClient::CancelLoad, py::call_guard<py::gil_scoped_release>())
      .def("accounting", &EngineClient::Accounting, py::call_guard<py::gil_scoped_release>())
      .def("heartbeat", &EngineClient::Heartbeat, py::call_guard<py::gil_scoped_release>())
      .def("close", &EngineClient::Close, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("healthy", &EngineClient::healthy)
      .def_property_readonly("backend_id", &EngineClient::backend_id)
      .def_property_readonly("last_received_seq", &EngineClient::last_received_seq);

  py::class_<EngineLoadHandle, std::shared_ptr<EngineLoadHandle>>(m, "LoadHandle")
      .def_property_readonly("id", &EngineLoadHandle::id)
      .def_property_readonly("layer_count", &EngineLoadHandle::layer_count)
      .def("layer_done", &EngineLoadHandle::layer_done)
      .def("layer_outcome", &EngineLoadHandle::layer_outcome)
      .def("layer_error", &EngineLoadHandle::layer_error)
      .def("layer_event", &EngineLoadHandle::layer_event)
      .def_property_readonly("outcome", &EngineLoadHandle::outcome)
      .def_property_readonly("error", &EngineLoadHandle::error)
      .def("wait", &EngineLoadHandle::Wait, py::call_guard<py::gil_scoped_release>());

  py::class_<EngineExecutor>(m, "EngineExecutor")
      .def(py::init([](std::string executor_id, int device, std::string io_host, uint16_t io_port,
                       uint64_t staging_bytes) {
             return std::make_unique<EngineExecutor>(EngineExecutorConfig{
                 std::move(executor_id), device, std::move(io_host), io_port, staging_bytes});
           }),
           py::arg("executor_id"), py::arg("device") = -1, py::arg("io_host") = "",
           py::arg("io_port") = 0, py::arg("staging_bytes") = 0)
      .def(
          "register_original_region",
          [](EngineExecutor& executor, std::string id, uintptr_t address, size_t size, int location,
             int device) {
            return executor.RegisterOriginalRegion(
                std::move(id), umbp::TransferRef::HostBytes(
                                   reinterpret_cast<void*>(address), size,
                                   static_cast<mori::io::MemoryLocationType>(location), device));
          },
          py::arg("region_id"), py::arg("address"), py::arg("size"), py::arg("location") = 1,
          py::arg("device") = -1)
      .def(
          "map_peer_region",
          [](EngineExecutor& executor, std::string executor_id, std::string region_id,
             uintptr_t address, size_t size, int location, int device) {
            return executor.MapPeerRegion(
                std::move(executor_id), std::move(region_id),
                umbp::TransferRef::HostBytes(reinterpret_cast<void*>(address), size,
                                             static_cast<mori::io::MemoryLocationType>(location),
                                             device));
          },
          py::arg("executor_id"), py::arg("region_id"), py::arg("address"), py::arg("size"),
          py::arg("location") = 1, py::arg("device") = -1)
      .def("map_shared_dram",
           [](EngineExecutor& executor, const PoolRegistration& registration) {
             std::string error;
             if (!executor.MapSharedDram(registration.dram_regions, &error)) {
               throw std::runtime_error(error);
             }
           })
      .def("bootstrap_remote",
           [](EngineExecutor& executor, std::string executor_id, py::bytes engine_desc) {
             std::string packed = engine_desc;
             return executor.BootstrapRemote(std::move(executor_id),
                                             std::vector<uint8_t>(packed.begin(), packed.end()));
           })
      .def("submit_load", &EngineExecutor::SubmitLoad, py::call_guard<py::gil_scoped_release>())
      .def("poll_loads", &EngineExecutor::PollLoads, py::call_guard<py::gil_scoped_release>())
      .def("close", &EngineExecutor::Close, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("mapped_region_count", &EngineExecutor::mapped_region_count);

  m.def("allocator_stats", [] {
    auto& allocator = GlobalKvVmmAllocator();
    py::dict out;
    out["allocations"] = allocator.AllocationCount();
    out["quarantined"] = allocator.QuarantinedCount();
    out["bytes"] = allocator.AllocatedBytes();
    out["failures"] = allocator.FailureCount();
    return out;
  });
  m.def(
      "lookup_allocation",
      [](uintptr_t ptr, size_t length) -> py::object {
        auto allocation = GlobalKvVmmAllocator().Lookup(reinterpret_cast<void*>(ptr), length);
        if (!allocation.has_value()) return py::none();
        py::dict out;
        out["base"] = reinterpret_cast<uintptr_t>(allocation->base);
        out["requested_size"] = allocation->requested_size;
        out["mapped_size"] = allocation->mapped_size;
        out["device"] = allocation->device;
        out["handle_kind"] = allocation->handle_kind;
        return std::move(out);
      },
      py::arg("ptr"), py::arg("length") = 1);
  m.def(
      "export_allocation",
      [](uintptr_t ptr) -> py::object {
        auto exported = GlobalKvVmmAllocator().Export(reinterpret_cast<void*>(ptr));
        if (!exported.has_value()) return py::none();
        py::dict out;
        out["handle_kind"] = exported->kind;
        out["fd"] = exported->fd;
        out["fabric"] = py::bytes(reinterpret_cast<const char*>(exported->fabric.data()),
                                  exported->fabric.size());
        out["mapped_size"] = exported->mapped_size;
        return std::move(out);
      },
      py::arg("ptr"));
}

}  // namespace mori
